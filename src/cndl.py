import copy

import numpy as np
from scipy.optimize import fsolve
from tqdm import tqdm

from src.setting import *
from src.endf_io import *
from src.algorithm import *
from src.gendf import GendfInterface, Reaction, MF3, MF6Like, MF16
from src.binary_io import NdlBinary
from src.physics import *


class CNDL(GendfInterface):
    def __init__(self, endf, gendf, verbose=False, MF7=None):
        super().__init__()

        if verbose:
            print("*** CONVERT GENDF TO CNDL ***")
            
        self._endf = endf
        self._gendf = gendf

        self.za = copy.deepcopy(self._gendf.za)
        self.egn = copy.deepcopy(self._gendf.egn)
        self.egg = copy.deepcopy(self._gendf.egg)

        secondary_unresolved_reactions = []

        # compressing GENDF library
        secondary_unresolved_reactions += self._secondary(verbose, MF7)
        self._absorp(secondary_unresolved_reactions)
        if verbose:
            print("All reactions that don't generate secondary particle are merged to MT=27")

        # sorting mt and mf values
        for mt in self.reactions:
            self.reactions[mt].mf = dict(sorted(self.reactions[mt].mf.items(), key=lambda item: item[0]))
        self.reactions = dict(sorted(self.reactions.items(), key=lambda item: item[0]))
        
    def _secondary(self, verbose, MF7):
        """
            processing all reactions that have any secondary particles
        """

        unresolved = []

        # elastic scattering. it should be integrated with (MT = 221)
        if 2 in self._gendf.reactions:
            if 221 in self._gendf.reactions:
                self._scattering(verbose, MF7)
                if verbose:
                    print("Low energy thermal scattering is integrated to MT=2")
            else:
                raise Exception("Thermal neutron scattering (MT=221) not found!")

        # radiative capture
        if 102 in self._gendf.reactions:
            self.reactions[102] = copy.deepcopy(self._gendf.reactions[102])
            for mf in (22, 23, 24, 25, 26):
                if mf in self.reactions[102].mf.keys(): # residual particle energy -> dose
                    self.reactions[102].mf[27] = self.reactions[102].mf[mf]
                    del self.reactions[102].mf[mf]

        # integrate (n,n') reactions (MT 50-91) to MT 4
        target_reaction = []
        for mt in self._gendf.reactions:
            if mt in reaction_secondary_neutron:
                target_reaction += [mt]
        if len(target_reaction) > 0:
            self._mergeInelastic(4, 6, target_reaction)
        elif 4 in self._gendf.reactions:
            if 6 in self._gendf.reactions[4].mf:
                self.reactions[4] = copy.deepcopy(self._gendf.reactions[4])
        if verbose:
            print("all neutron inelastic scattering reactions are integrated to MT4")

        # integrate (n,p) reactions (MT 600-650) to MT 103
        target_reaction = []
        for mt in self._gendf.reactions:
            if mt in reaction_secondary_proton:
                target_reaction += [mt]
        if len(target_reaction) > 0:
            self._mergeInelastic(103, 21, target_reaction)
        elif 103 in self._gendf.reactions:
            if 21 in self._gendf.reactions[103].mf:
                self.reactions[103] = copy.deepcopy(self._gendf.reactions[103])  
            else:
                if verbose:
                    print("proton energy-angle distribution is not in MT103")
                    print("consider MT103 as full energy absorption reaction")
                unresolved += [103] 
        if verbose:
            print("all (n,p') reactions are integrated to MT103")

        # change MF26 to MF27   
        for mt in (4, 103):
            if mt in self.reactions:
                if 26 in self.reactions[mt].mf.keys(): # residual particle energy -> dose
                    self.reactions[mt].mf[27] = self.reactions[mt].mf[26]
                    del self.reactions[mt].mf[26]

        # other reactions
        target_reaction = []
        for mt in self._gendf.reactions:
            if mt in reaction_secondary:
                target_reaction += [mt]
        if len(target_reaction) == 0:
            return unresolved

        for mt in target_reaction:
            self.reactions[mt] = copy.deepcopy(self._gendf.reactions[mt])
            if getCutoffParticleNumber(mt):
                # check resnuc id
                ptype = checkResNucType(self.za, mt)
                # check whether this mt reaction is unresolved or not
                is_unresolved = False
                if 26 not in self.reactions[mt].mf and ptype not in self.reactions[mt].mf:
                    is_unresolved = True
                for particle in reaction_multiplicity[mt]:
                    if particle not in self.reactions[mt].mf:
                        is_unresolved = True
                
                # calculate residual dose for resolved reaction
                if is_unresolved:
                    if verbose:
                        print("{} reaction is unresolved".format(self.reactions[mt].__repr__()))
                else:
                    self._mergeResidual(mt)
                    if verbose:
                        print("All residual heavy ions of {} reaction are merged to MF=27".format(self.reactions[mt].__repr__()))

            else:
                if 26 in self.reactions[mt].mf.keys(): # residual particle energy -> dose
                    self.reactions[mt].mf[27] = self.reactions[mt].mf[26]
                    del self.reactions[mt].mf[26]

        return unresolved

    def _scattering(self, verbose, MF7):
        """
            generate elastic scattering (MT = 2) by free-gas thermal scattering law
            and MF4 scattering law
        """
        if verbose:
            print("Generate elastic scattering (MT = 2)")
            if MF7 is None:
                print("by free-gas thermal scattering law and MF4 scattering law")
            else:
                print("by MF7 tabulated S(a,b) kernel and MF4 scattering law")
        
        self.reactions[2] = Reaction(2)
        self.reactions[2].mf[3] = copy.deepcopy(self._gendf.reactions[2].mf[3])
        
        vfunc = np.vectorize(logMean)
        energy_mean = vfunc(self.egn[1:], self.egn[:-1])

        thermal_thres = np.argmin(self._gendf.reactions[221].mf[6].target_tape[:,0] > -1)
        energy_thermal = float(ENV["thermal_thres"])

        # calculate physical quantities
        A = self._endf.target["mass"] / self._endf.projectile["mass"]
        temp = float(ENV["temperature"])
        alpha = (A-1)**2/(A+1)**2

        data = []
        label = []

        # equiprob bin setting
        equi_steps = int(ENV["equibin_steps"])
        nbin = int(ENV["equiprob_nbin"])
        trans_steps = int(ENV["enetrans_steps"])
        equi_probs = np.linspace(0, 1, nbin+1)

        # get elastic scattering angular distribution
        ad = MF4AngularDistribution(A, self._endf.reactions[2].angular_distribution)
        ksteps = int(ENV["kernel_steps"])
        if MF7 is not None: # MF7 S(a,b) kernel mode
            MF7_mode = True
            tsk = ThermalScatteringKernel(A, temp, MF7)
        else:
            MF7_mode = False

        # build energy transition probability map
        for i in tqdm(range(len(energy_mean))):
            if i < thermal_thres: # thermal scattering
                if MF7_mode: # S(a,b) kernel mode
                    data_seg = np.zeros((thermal_thres+1, 1))
                    for egroup in range(thermal_thres):
                        # get alpha-beta map (linear)
                        beta_min = tsk.beta(energy_mean[i], self.egn[egroup], None)
                        beta_max = tsk.beta(energy_mean[i], self.egn[egroup+1], None)
                        beta_list = np.linspace(beta_min, beta_max, ksteps)
                        beta_list_mean = (beta_list[1:] + beta_list[:-1]) / 2
                        amax_list = tsk.amax(energy_mean[i], beta_list_mean)
                        amin_list = tsk.amin(energy_mean[i], beta_list_mean)

                        bmap = np.expand_dims(beta_list_mean, axis=0)
                        bmap = np.broadcast_to(bmap, (ksteps-1, ksteps-1))
                        amap = np.linspace(amin_list, amax_list, ksteps)
                        area_map = (amap[1:] - amap[:-1]) * (beta_max - beta_min) / ksteps
                        amap = (amap[1:] + amap[:-1])/2

                        vfunc = np.vectorize(lambda a, b: tsk.get(a, b))
                        v = vfunc(amap, bmap) * area_map

                        data_seg[egroup + 1, 0] = np.sum(v)

                else: # free gas scattering
                    ts = ThermalScattering2D(A, temp, energy_mean[i], energy_thermal)
                    data_seg = np.zeros((thermal_thres+1, 1))
                    for egroup in range(thermal_thres):
                        data_seg[egroup + 1, 0] = ts.getProb(self.egn[egroup], self.egn[egroup + 1])

                label += [[len(label), i + 1, 1]]
                data += [data_seg]

            else: # fast neutron elastic scattering
                esample = np.logspace(np.log10(self.egn[i]), np.log10(self.egn[i+1]), trans_steps+2)[1:-1]
                #esample = np.linspace(self.egn[i], self.egn[i+1], trans_steps+2)[1:-1]
                for j in range(trans_steps):
                    emin = alpha * esample[j]
                    gmin = max(np.argmax(self.egn > emin) - 1, 0)
                    gmax = i                
                    ebin_target = np.copy(self.egn[gmin:gmax+2])
                    ebin_target[0] = max(emin, ebin_target[0])
                    ebin_target[-1] = esample[j]
                    pyfunc = lambda inc_ene, elow, eup: ad.getArea(inc_ene, elow, eup)
                    vfunc = np.vectorize(pyfunc)
                    if j == 0:
                        prob = vfunc(esample[j], ebin_target[:-1], ebin_target[1:])
                        prob /= np.sum(prob)
                        gmin_lowest = gmin
                    else:
                        prob_seg = vfunc(esample[j], ebin_target[:-1], ebin_target[1:])
                        prob_seg /= np.sum(prob_seg)
                        prob += np.pad(prob_seg, (len(prob) - len(prob_seg),0))
                prob /= trans_steps
                data_seg = np.zeros((len(prob) + 1, 1))
                data_seg[1:, 0] = prob
                label += [[len(label), i + 1, gmin_lowest + 1]]
                data += [data_seg]

        self.reactions[2].mf[6] = MF6Like(data, label, len(self.egn) - 1, 6)

        # build equiprobable angular distribution
        equiprob_data = np.empty((len(self.reactions[2].mf[6].prob_map), nbin + 1), dtype=np.float64)
        pointer = 0
        for i in tqdm(range(len(self.reactions[2].mf[6].target_tape))):
            target_start, group_start = self.reactions[2].mf[6].target_tape[i]
            emin = alpha * energy_mean[i]
            emax = energy_mean[i]
            elast = emin
            if i == len(self.reactions[2].mf[6].target_tape) - 1:
                target_end = len(self.reactions[2].mf[6].prob_map)
            else:
                target_end = self.reactions[2].mf[6].target_tape[i+1,0]
            for pos, j in enumerate(range(group_start, group_start + target_end - target_start)):
                equiprob_seg = np.ones((1, nbin + 1), dtype=np.float64)
                if i < thermal_thres: # thermal scattering (free gas)
                    if MF7_mode: # S(a,b) kernel mode
                        beta = tsk.beta(energy_mean[i], energy_mean[j], None)
                        amax = tsk.amax(energy_mean[i], beta)
                        amin = tsk.amin(energy_mean[i], beta)

                        alist = np.linspace(amin, amax, equi_steps)
                        amean = (alist[1:] + alist[:-1]) / 2
                        # get total area
                        area = np.empty(amean.shape)
                        for k in range(len(amean)):
                            area[k] = tsk.get(amean[k], beta)
                        

                        area = np.append(0, area)
                        area_tot = np.sum(area)
                        interp = interp1d(np.cumsum(area) / area_tot, alist, 2)
                        area_target = np.linspace(0, 1, nbin+1)[1:-1]
                        alpha_target = interp.get(area_target)
                        alpha_target = np.append(amin, alpha_target)
                        alpha_target = np.append(alpha_target, amax)

                        equiprob_seg = tsk.getMu(energy_mean[i], energy_mean[j], alpha_target)
                        
                    else:
                        ts = ThermalScattering1D(A, temp, energy_mean[i], energy_mean[j])
                        bin_pointer = 0
                        prob_last = 0
                        for mu in np.linspace(-1, 1, equi_steps):
                            prob = ts.getProb(-1, mu)
                            while prob > equi_probs[bin_pointer]:
                                # do linear interpolation
                                mu_pos = mu_last + (equi_probs[bin_pointer]-prob_last)/(prob-prob_last)*(mu-mu_last)
                                equiprob_seg[0,bin_pointer] = mu_pos
                                bin_pointer += 1
                                if bin_pointer >= nbin + 1:
                                    break
                            if bin_pointer >= nbin + 1:
                                break

                            prob_last = prob
                            mu_last = mu

                else: # fast neutron elastic scattering
                    area = self.reactions[2].mf[6].prob_map[target_start + pos,0]
                    etarget = ad.getCumulEnergy(energy_mean[i], area, 50)
                    equiprob_seg = ad.getEquiAngularBin(energy_mean[i], elast, etarget, nbin)
                    elast = etarget
                equiprob_data[pointer] = equiprob_seg
                pointer += 1
                
        self.reactions[2].mf[6].equiprob_map = \
            np.append(self.reactions[2].mf[6].prob_map[:,:1],
                      equiprob_data,
                      axis = 1)

    def _mergeInelastic(self, mt, mf, target_reaction):
        """
            merging inelastic scattering (n,n') and (n,p')
        """

        for gendf_mt in target_reaction:
            if 26 not in self._gendf.reactions[gendf_mt].mf: # no resnuc information
                data = []
                label = []
                xs = self._gendf.reactions[gendf_mt].mf[3].xs
                for egroup in range(len(self.egn) - 1):
                    edist = np.zeros(len(self.egn) - 1, dtype=np.float64)
                    if xs[egroup] == 0:
                        continue
                    residual_energy = logMean(self.egn[egroup], self.egn[egroup+1])
                    residual_energy += self._endf.reactions[gendf_mt].Q_reaction
                    for gendf_mf in self._gendf.reactions[gendf_mt].mf:
                        if gendf_mf not in (3, 16):
                            residual_energy -= self._gendf.getHadronMeanEnergy(gendf_mt, gendf_mf, egroup)
                    if 16 in self._gendf.reactions[gendf_mt].mf: # photon is generated
                        residual_energy -= self._gendf.getGammaMeanEnergy(gendf_mt, egroup)
                    if residual_energy > self.egn[0]:
                        edist[np.argmax(self.egn > residual_energy) - 1] += xs[egroup]
                    else:
                        edist[0] += xs[egroup]
                    # write data and label
                    lower_e = np.argmax(edist > 0)
                    upper_e = len(self.egn) - np.argmax(np.flip(edist) > 0) - 1
                    data_seg = np.copy(edist[lower_e:upper_e])
                    # flux dummy
                    data_seg = np.append(0, data_seg)
                    label += [[len(data), egroup + 1, lower_e + 1]]
                    data += [np.expand_dims(data_seg, axis=1)]
                self._gendf.reactions[gendf_mt].mf[26] = MF6Like(data, label, len(self.egn)-1, 26)

        self.reactions[mt] = Reaction(mt)
        self.reactions[mt].mf[3] = MF3(np.zeros(len(self.egn)-1))
        for gendf_mt in target_reaction: # merge XS (MF=3)
            self.reactions[mt].mf[3].xs += self._gendf.reactions[gendf_mt].mf[3].xs

        for gendf_mf in (mf, 26): # merge secondary particle and resnuc

            order = -1
            ngn = -1
            for gendf_mt in target_reaction: # check largest Legendre order
                if gendf_mf in self._gendf.reactions[gendf_mt].mf:
                    order = max(order, self._gendf.reactions[gendf_mt].mf[gendf_mf].prob_map.shape[1])
                    ngn = len(self._gendf.reactions[gendf_mt].mf[gendf_mf].target_tape)
            
            if ngn == -1: # no mf value in gendf_mt reaction
                continue
            matrix = np.zeros((ngn, ngn, order), dtype=np.float64)
            for gendf_mt in target_reaction:
                seg = self._gendf.reactions[gendf_mt].mf[gendf_mf]._getTransMatrix()
                gmin = np.argmax(self._gendf.reactions[gendf_mt].mf[3].xs > 0)
                mul = self._gendf.reactions[gendf_mt].mf[3].xs[gmin:]
                mul = np.expand_dims(mul, axis=1)
                mul = np.expand_dims(mul ,axis=2)
                seg[gmin:,:,:] *= np.broadcast_to(mul, seg[gmin:,:,:].shape)
                matrix[:,:,:seg.shape[2]] += seg

            # normalize probability distribution   
            gmin = np.argmax(self._gendf.reactions[mt].mf[3].xs > 0)
            div = np.sum(matrix[gmin:,:,0], axis=1)
            div = np.expand_dims(div, axis=1)
            matrix[gmin:,:,0] = \
                np.divide(matrix[gmin:,:,0],
                          np.broadcast_to(div, matrix[gmin:,:,0].shape),
                          out=np.zeros_like(matrix[gmin:,:,0]),
                          where=np.broadcast_to(div, matrix[gmin:,:,0].shape)!=0)

            # normalize Legendre polynomials
            mask = matrix[:,:,0] > 0
            matrix[mask,1:] /= np.broadcast_to(np.expand_dims(matrix[mask,1], axis=1), matrix[mask,1:].shape)

            self.reactions[mt].mf[gendf_mf] = MF6Like([], [], len(self.egn)-1, gendf_mf)
            self.reactions[mt].mf[gendf_mf].setFromMatrix(matrix)

        # merge MF16 (gamma)
        gendf_mf = 16
        ngg = len(self.egg)-1
        matrix = np.zeros((ngn, ngg), dtype=np.float64)
        xs_tot = np.zeros(ngn, dtype=np.float64)
        for gendf_mt in target_reaction:
            mul = self._gendf.reactions[gendf_mt].mf[3].xs
            xs_tot += mul
            if gendf_mf in self._gendf.reactions[gendf_mt].mf:
                seg = self._gendf.reactions[gendf_mt].mf[gendf_mf]._getTransMatrix(ngg)
                mul *= self._gendf.reactions[gendf_mt].mf[gendf_mf]._getMultiplicity()
                mul = np.expand_dims(mul, axis=1)
                matrix += seg * np.broadcast_to(mul, matrix.shape)

        # get net multiplicity matrix
        divider = np.sum(matrix, axis=1)
        multiplicity = np.divide(divider, xs_tot,
                                 out=np.zeros_like(divider),
                                 where=xs_tot!=0)

        # normalize transition matrix
        divider = np.expand_dims(divider, axis=1)
        divider = np.broadcast_to(divider, matrix.shape)
        matrix = np.divide(matrix, divider,
                           out=np.zeros_like(matrix),
                           where=divider!=0)
        if np.sum(multiplicity) > 0:
            self.reactions[mt].mf[gendf_mf] = MF16([], [], len(self.egn)-1, None)
            self.reactions[mt].mf[gendf_mf].setFromMatrix(matrix, multiplicity)

    def _mergeResidual(self, mt):
        nsample = int(ENV["res_merge_nsample"])
        reaction = self.reactions[mt]
        sampling_inst = reaction_multiplicity[mt]

        data = []
        label = []

        for igroup in range(len(self.egn) - 1):
            if reaction.mf[3].xs[igroup] == 0:
                continue

            ene_count = np.zeros(len(self.egn) - 1, dtype=np.int32)
            for _ in range(nsample):
                energy = 0
                for mf in sampling_inst.keys():
                    if mf == 6 or mf == 21:
                        continue
                    for imul in range(sampling_inst[mf]):
                        energy += self._samplingEnergy(mt, mf, igroup)
                ene_count[np.argmax(energy < self.egn) - 1] += 1
            gfirst = np.argmax(ene_count > 0)
            glast = len(self.egn) - 1 - np.argmax(np.flip(ene_count) > 0)
            label += [[len(label), igroup + 1, gfirst + 1]]
            data += [np.expand_dims(np.append(0, ene_count[gfirst:glast].astype(np.float64) / nsample * reaction.mf[3].xs[igroup]), axis=1)]
        
        reaction.mf[27] = MF6Like(data, label, len(self.egn) - 1, 27)

        # remove heavy ion angle-energy distributiom (MF6Like)
        for mf in sampling_inst.keys():
            if mf == 6 or mf == 21:
                continue
            del reaction.mf[mf]
            
        if 26 in reaction.mf:
            del reaction.mf[26]

    def _absorp(self, unresolved_list):
        """
            merging all reactions that not have any secondary particles
        """
        target_reaction = unresolved_list
        for mt in self._gendf.reactions:
            if mt in reaction_absorption:
                target_reaction += [mt]
        if len(target_reaction) == 0:
            return     

        # generate absorption cross section (MT = 27)
        self.reactions[27] = Reaction(27)

        # calculate total cross section (MF = 3)
        xs = np.zeros(len(self.egn) - 1, dtype=np.float64)
        for mt in target_reaction:
            xs += self._gendf.reactions[mt].mf[3].xs
        self.reactions[27].mf[3] = MF3(xs)

        # calculate the energy distribution (MF = 6)
        data = []
        label = []
        for egroup in range(len(self.egn) - 1):
            edist = np.zeros(len(self.egn) - 1, dtype=np.float64)
            for mt in target_reaction:
                xs = self._gendf.reactions[mt].mf[3].xs[egroup]
                if xs == 0:
                    continue
                residual_energy = logMean(self.egn[egroup], self.egn[egroup+1])
                residual_energy += self._endf.reactions[mt].Q_reaction
                if 16 in self._gendf.reactions[mt].mf: # photon is generated
                    residual_energy -= self._gendf.getGammaMeanEnergy(mt, egroup)
                edist[np.argmax(self.egn > residual_energy) - 1] += xs
            if np.sum(edist) == 0: # no XS
                continue
            # write data and label
            lower_e = np.argmax(edist > 0)
            upper_e = len(self.egn) - np.argmax(np.flip(edist) > 0) - 1
            data_seg = np.copy(edist[lower_e:upper_e])
            # flux dummy
            data_seg = np.append(0, data_seg)
            label += [[len(data), egroup + 1, lower_e + 1]]
            data += [np.expand_dims(data_seg, axis=1)]
        self.reactions[27].mf[27] = MF6Like(data, label, len(self.egn)-1, 27)

        # calculate gamma energy spectrum (MF = 16)
        data = []
        label = []
        for egroup in range(len(self.egn) - 1):
            edist = np.zeros(len(self.egg) - 1, dtype=np.float64)
            for mt in target_reaction:
                xs = self._gendf.reactions[mt].mf[3].xs[egroup]
                if xs == 0:
                    continue
                if 16 in self._gendf.reactions[mt].mf.keys(): # photon is generated
                    spectrum = self._gendf.reactions[mt].mf[16]._getSpectrum(egroup)
                    if spectrum is None:
                        continue
                    spectrum = np.pad(spectrum, (0,len(edist) - len(spectrum)))
                    edist += xs * spectrum
            if np.sum(edist) == 0: # no spectrum
                continue
            # write data and label
            lower_e = np.argmax(edist > 0)
            upper_e = len(self.egg) - np.argmax(np.flip(edist) > 0) - 1
            data_seg = np.copy(edist[lower_e:upper_e])
            # flux dummy
            data_seg = np.append(0, data_seg)
            label += [[len(data), egroup + 1, lower_e + 1]]
            data += [np.expand_dims(data_seg, axis=1)]

        if len(data) > 0:
            self.reactions[27].mf[16] = MF16(data, label, len(self.egn)-1, self.reactions[27].mf[3].xs)

    def genEquiProb(self, verbose=False):
        """
            generate equiprob angular distribution
        """
        if verbose:
            print("*** GENERATE EQUIPROB ANGULAR DISTRIBUTION ***")
        for mt in self.reactions:
            if mt == 2: # elastic scattering. It follows different equiprob cosine generating scheme.
                pass
            else:
                for mf in self.reactions[mt].mf:
                    if mf not in (3,16,26,27):
                        self.reactions[mt].mf[mf].genEquiProbMap()
                        if verbose:
                            print("MT={} {}, MF={} is converted to equiprob map".format(mt, self.reactions[mt].__repr__(), mf))

    def write(self, file_name, get_reactions_list=False):
        """
            write binary file of GPUMC compressed neutron data library
        """
        file = NdlBinary(file_name, mode="w")
        reactions_list = np.empty(0, dtype=np.int32) # for debugging
        # generate reaction cumulative probability map
        reaction_cumul_map = np.empty((len(self.egn) - 1, len(self.reactions)), dtype=np.float64)
        for i, mt in enumerate(self.reactions):
            reaction_cumul_map[:,i] = self.reactions[mt].mf[3].xs
            reactions_list = np.append(reactions_list, mt)
        reaction_cumul_map = np.cumsum(reaction_cumul_map, axis=1)
        # save ZA and atomic mass
        file.write(np.array([self.za], dtype=np.int32))
        file.write(np.array([self._endf.target['mass']], dtype=np.float32))
        # save total cross section
        file.write(reaction_cumul_map[:,-1])
        # save reactions MT list
        if get_reactions_list:
            file.write(reactions_list.astype(np.int32))
        # normalize cumulative probability map
        reaction_cumul_map /= np.broadcast_to(np.expand_dims(reaction_cumul_map[:,-1], 1), reaction_cumul_map.shape)
        # save reaction type sampling probability map
        file.write(reaction_cumul_map.astype(np.float32))
        for mt in self.reactions: 
            # for all reactions, build sampling law card
            # always this order: [res_dose, gamma, hadron1, hadron2, ...]
            # sampling law card structure [pid, pointer, pid, pointer, ...]
            n = 0
            sampling_law = np.empty(0, dtype=np.int32)
            # check res dose
            if 27 in self.reactions[mt].mf:
                sampling_law = np.append(sampling_law, [27, n])
                n += 1
            else:
                sampling_law = np.append(sampling_law, [27, -1])
            # check gamma
            if 16 in self.reactions[mt].mf:
                sampling_law = np.append(sampling_law, [16, n])
                n += 1
            else:
                sampling_law = np.append(sampling_law, [16, -1])
            # check secondary hadron
            for mf in (6, 21):
                if mf in self.reactions[mt].mf: # neutron
                    if mf in reaction_multiplicity[mt]:
                        multiplicity = reaction_multiplicity[mt][mf]
                    else:
                        multiplicity = 1
                    for m in range(multiplicity):
                        sampling_law = np.append(sampling_law, [mf, n])
                    n += 1
            # write sampling law array
            file.write(sampling_law.astype(np.int32))
            # for each reaction, write target tape and probability map
            if 27 in self.reactions[mt].mf:
                file.write(self.reactions[mt].mf[27].target_tape)
                file.write(self.reactions[mt].mf[27].prob_map[:,0].astype(np.float32))
            if 16 in self.reactions[mt].mf:
                file.write(self.reactions[mt].mf[16].target_tape)
                file.write(self.reactions[mt].mf[16].prob_map.astype(np.float32))
            if 6 in self.reactions[mt].mf:
                file.write(self.reactions[mt].mf[6].target_tape)
                file.write(self.reactions[mt].mf[6].equiprob_map.astype(np.float32))
            if 21 in self.reactions[mt].mf:
                file.write(self.reactions[mt].mf[21].target_tape)
                file.write(self.reactions[mt].mf[21].equiprob_map.astype(np.float32))                
        file.close()
        

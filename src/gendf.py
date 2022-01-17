from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import eval_legendre

from src.endf_io import *
from src.algorithm import *
from src.setting import *


class MF3:
    def __init__(self, xs):
        self.xs = xs

    def __repr__(self):
        return "reaction cross-sections"


class MF6Like:
    def __init__(self, data, label, ngroup, mf):
        self._mf = mf
        self.equiprob_map = None
        if len(data) == 0:
            self.target_tape = None
            self.prob_map = None
            return
        # generate group transition probability density map

        self.target_tape = np.ones((ngroup, 2), dtype=np.int32) * -1
        # target_tape structure
        """
            target_tape[n,0] = index of target probability array 
                               for n-th incident neutron energy group
            target_tape[n,1] = lowest energy group of exit channel
                                when the n-th group neutron take reaction
        """
        self.prob_map = np.empty((0, data[0].shape[1]), dtype=np.float64)
        # prob_map structure
        """
            prob_map[m,0] = cumulative function of energy of exit channel
            prob_map[n,1:] = Legendre polynomial coefficient (angular distribution)
        """

        for data_index, group, target in label:
            # change group structure from ENDF (lowest=1) to Python (lowest=0)
            group -= 1
            target -= 1
            # set target tape
            if np.sum(data[data_index][1:]) == 0:
                continue
            self.target_tape[group] = [len(self.prob_map), target]
            self.prob_map = np.append(self.prob_map, data[data_index][1:], axis=0)

        # set cumulative density function
        self.prob_map = np.pad(self.prob_map, [(0,0), (1,0)], mode='constant')
        target_temp = np.append(self.target_tape[:,0], len(self.prob_map))
        for i in range(len(target_temp) - 1):
            if target_temp[i] < 0:
                continue
            ti = i+1
            while True:
                if target_temp[ti] > 0:
                    break
                if ti > ngroup:
                    raise ValueError
                ti += 1
            target_cum = self.prob_map[target_temp[i]:target_temp[ti],1]
            target_cum = np.cumsum(target_cum)
            if target_cum[-1] == 0: # no information
                self.target_tape[i,0] = -1
                self.target_tape[i,1] = -1
                target_cum = np.zeros(target_cum.shape)
            else:
                target_cum /= target_cum[-1]
            self.prob_map[target_temp[i]:target_temp[ti],0] = target_cum

        # normalize legendre polynomials
        divider = np.broadcast_to(np.expand_dims(self.prob_map[:,1], axis=1), self.prob_map[:,1:].shape)
        self.prob_map[:,1:] = np.divide(self.prob_map[:,1:], divider, out=np.zeros_like(divider), where=divider!=0)

    def __repr__(self):
        if self._mf == 6:
            return "neutron energy-angle distribution"
        elif self._mf == 21:
            return "proton energy-angle distribution"
        elif self._mf == 22:
            return "deuteron energy-angle distribtion"
        elif self._mf == 23:
            return "triton energy-angle distribution"
        elif self._mf == 24:
            return "helium-3 energy-angle distribution"
        elif self._mf == 25:
            return "alpha energy-angle distribution"
        elif self._mf == 26:
            return "residual nucleus (a>4) energy-angle distribution"
        elif self._mf == 27:
            return "deposited energy distribution"
        else:
            raise ValueError("unexpected MF value")

    def _getTransMatrix(self):
        """
            get group-to-group transition probability matrix
        """
        matrix = np.zeros((self.target_tape.shape[0], self.target_tape.shape[0], self.prob_map.shape[1]), dtype=np.float64)
        for i in range(len(self.target_tape)):
            if self.target_tape[i,0] < 0:
                continue
            target_begin = self.target_tape[i,0]
            if i == len(self.target_tape) - 1: # end of card
                target_end = len(self.prob_map)
            else:
                if np.sum(self.target_tape[i+1:,0] > 1):
                    next_pos = np.argmax(self.target_tape[i+1:,0] > 1)
                    target_end = self.target_tape[i+1:,0][next_pos]
                else:
                    target_end = len(self.prob_map) 
            target_group = self.target_tape[i,1]
            prob_seg = np.copy(self.prob_map[target_begin:target_end])
            prob_seg[1:,0] -= prob_seg[:-1,0]
            if np.sum(prob_seg[:,0]) == 0:
                raise ValueError
            prob_seg[:,0] /= np.sum(prob_seg[:,0])
            
            # set matrix element of i-th incident group
            matrix[i,target_group:target_group+len(prob_seg)] = prob_seg

        return matrix

    def setFromMatrix(self, matrix):
        """
            set prob map and target tape from transition probability matrix
        """
        gmin = np.argmax(np.sum(matrix[:,:,0], axis=1) > 0)
        matrix_start = np.argmax(matrix[:,:,0] > 0, axis=1)
        prob_map_size = np.sum(matrix[:,:,0] > 0, axis=1)

        # set target tape
        self.target_tape = np.zeros((matrix.shape[0], 2), dtype=np.int32)
        self.target_tape[1:,0] = np.cumsum(prob_map_size)[:-1]
        self.target_tape[:,1] = matrix_start
        self.target_tape[:gmin] = [-1, -1]

        # set prob map
        prob_map_region = matrix[:,:,0] > 0
        self.prob_map = matrix[prob_map_region]
        cum = np.cumsum(matrix, axis=1)
        self.prob_map[:,0] = cum[prob_map_region][:,0]

        return

    def genEquiProbMap(self):
        """
            convert Legendre distribution to equiprobable distribution
        """
        nbin = int(ENV["equiprob_nbin"])
        self.equiprob_map = np.empty((len(self.prob_map), nbin + 2), dtype=np.float64)
        self.equiprob_map[:,0] = np.copy(self.prob_map[:,0])
        modifier = (np.arange(0, self.prob_map.shape[1] - 1, 1) * 2 + 1) / 2
        for i in range(len(self.equiprob_map)):
            self.equiprob_map[i,1:] = legendreToEquibin(self.prob_map[i,1:] * modifier, nbin)[0]

class MF16:
    def __init__(self, data, label, ngroup, xs):
        self.equiprob_map = None
        is_const = False # constant spectrum flag
        const_floor = None # lowest energy group of constant spectrum 
        const_target_index = None # constant spectrum target index
        if len(data) == 0:
            self.target_tape = None
            self.prob_map = None
            return
        self.target_tape = np.ones((ngroup, 3), dtype=np.int32) * -1
        self.target_tape[:,1] = 0
        # target_tape structure
        """
            target_tape[n,0] = index of target probability array
                               for n-th incident neutron energy group
            target_tape[n,1] = gamma multiplicity (encoded from float32 to int32)
            target_tape[n,2] = lowest energy group of exit channel
                               when the n-th group neutron take reaction
        """
        self.prob_map = np.empty(0, dtype=np.float64)
        """
            prob_map = cumulative function of group of exit channel
        """
        for data_index, group, floor in label:
            if group == 0: # constant spectrum mod is activated, read constant spectrum
                is_const = True
                const_floor = floor - 1
                const_target_index = len(self.prob_map)
                cumul = np.cumsum(data[data_index][:,0])
                cumul = cumul / cumul[-1]
                self.prob_map = np.append(self.prob_map, cumul)
            elif floor == 0: # constant spectrum mod is activated, read gamma multiplicity
                if is_const == False:
                    raise SyntaxError
                multiplicity = data[data_index][1] / xs[group - 1]
                mul_encoded = multiplicity.astype(np.float32).tobytes()
                mul_int32 = np.frombuffer(mul_encoded, dtype=np.int32)[0]
                self.target_tape[group-1] = [const_target_index, mul_int32, const_floor]
            else: # constant spectrum mode is deactived, read spectrum
                if is_const == True:
                    is_const = False
                # set target tape
                if np.sum(data[data_index][1:,0]) == 0:
                    continue
                multiplicity = np.sum(data[data_index][1:,0]) / xs[group - 1]
                mul_encoded = multiplicity.astype(np.float32).tobytes()
                mul_int32 = np.frombuffer(mul_encoded, dtype=np.int32)[0]
                self.target_tape[group-1] = [len(self.prob_map), mul_int32, floor - 1]
                # set prob map
                cumul = np.cumsum(data[data_index][1:,0])
                if cumul[-1] == 0: # no information
                    self.target_tape[group-1,0] = -1
                    self.target_tape[group-1,2] = -1
                    cumul = np.zeros(cumul.shape)
                else:
                    cumul = cumul / cumul[-1]
                self.prob_map = np.append(self.prob_map, cumul)

    def __repr__(self):
        return "photon multiplication and spectrum"

    def _getMultiplicity(self):
        mul_encoded = self.target_tape[:,1].tobytes()
        return np.frombuffer(mul_encoded, dtype=np.float32)

    def _getSpectrum(self, group):
        target, mul_int32, floor = self.target_tape[group]
        if floor == -1:
            return None
        # get multiplicity
        mul_encoded = np.array([mul_int32], dtype=np.int32).tobytes()
        multiplicity = np.frombuffer(mul_encoded, dtype=np.float32)[0]
        target_to = np.argmax(self.prob_map[target:] == 1.e0) + 1 + target
        prob = np.copy(self.prob_map[target:target_to])
        prob[1:] -= prob[:-1]
        return np.pad(prob, (floor,0)) * multiplicity

    def _getTransMatrix(self, ngg):
        """
            get neutron group to photon group transition probability matrix
        """
        matrix = np.zeros((self.target_tape.shape[0], ngg), dtype=np.float64)
        for group in range(len(self.target_tape)):
            target, mul_int32, floor = self.target_tape[group]
            if floor < 0:
                continue
            target_to = np.argmax(self.prob_map[target:] == 1.e0) + 1 + target
            prob = np.copy(self.prob_map[target:target_to])
            prob[1:] -= prob[:-1]
            matrix[group] = np.pad(prob, (floor,ngg-floor-len(prob)))
        return matrix

    def setFromMatrix(self, matrix, multiplicity):
        """
            set prob map and target tape from transition probability matrix
        """
        gmin = np.argmax(np.sum(matrix, axis=1) > 0)     
        floor = np.argmax(matrix > 0, axis=1)
        ceil = -np.argmax(np.flip(matrix, axis=1) > 0, axis=1)
        self.target_tape = np.zeros((matrix.shape[0],3), dtype=np.int32)
        self.prob_map = np.empty(0, dtype=np.float64)
        # set multiplicity
        mul_bytes = multiplicity.astype(np.float32).tobytes()
        mul_int = np.frombuffer(mul_bytes, dtype=np.int32)
        self.target_tape[:,1] = mul_int
        # set target tape line that under the reaction threshold
        self.target_tape[:gmin,0] = -1
        self.target_tape[:gmin,2] = -1
        for group in range(gmin,matrix.shape[0]):
            # set target tape
            self.target_tape[group,0] = len(self.prob_map)
            self.target_tape[group,2] = floor[group]
            # set prob map
            seg = matrix[group,floor[group]:ceil[group]]
            seg = np.cumsum(seg)
            seg /= seg[-1]
            self.prob_map = np.append(self.prob_map, seg)
        return

class Reaction:
    def __init__(self, mt):
        self.mt = mt
        self.mf = {}

    def __repr__(self):
        return reaction_type[self.mt]

    def _add(self, egn, mf, nz, lrflag, data, label):
        if mf == 3: # cross-section
            xs = np.zeros(len(egn) - 1, dtype=np.float64)
            for i in range(len(label)):
                xs[label[i][1] - 1] = data[i][1,0]
            self.mf[mf] = MF3(xs)
        elif mf == 6: # neutron energy-angle distribution
            if self.mt == 221: # thermal scattering law
                self.mf[mf] = MF6Like(data[:-1], label[:-1], egn.shape[0]-1, mf)
            else:
                self.mf[mf] = MF6Like(data, label, egn.shape[0] - 1, mf)
        elif mf == 16: # photon multiplication & spectrum
            if 3 in self.mf:
                self.mf[mf] = MF16(data, label, egn.shape[0] - 1, self.mf[3].xs)
        else: # other charged particle
            self.mf[mf] = MF6Like(data, label, egn.shape[0] - 1, mf)


class GendfInterface:
    def __init__(self):
        self.egn = None
        self.egg = None

        # plot attributes
        self._legend = []
        self._plot_type = None

        # reaction list
        self.reactions = {}

    def show(self):
        """
            show plotted data
        """
        plt.legend(self._legend)
        self._legend = []
        self._plot_type = None
        plt.show()

    def plotXS(self, mt): # type = 0
        """
            plot reaction cross section
        """
        if self._plot_type is None:
            self._plot_type = 0
        elif self._plot_type != 0:
            raise TypeError

        plt.step(self.egn[:-1] * 1e-6, self.reactions[mt].mf[3].xs, where='post')
        plt.title("material {} reaction cross section".format(self.za))
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Cross-section (barn)")
        self._legend += [self.reactions[mt]]

    def plotTransitionMatrix(self, mt): # type = 1
        """
            plot energy group transition probability matrix
        """
        if self._plot_type is None:
            self._plot_type = 1
        elif self._plot_type != 1:
            raise TypeError
        
        plt.imshow(self.reactions[mt].mf[6]._getTransMatrix()[:,:,0], 
                   cmap="jet", origin="lower", norm=LogNorm())
        plt.title("{} transition matrix".format(self.reactions[mt].__repr__()))
        plt.xlabel("energy group index of exit particle")
        plt.ylabel("energy group index of incident particle")
        plt.colorbar()
        self._legend = []
        self._plot_type = None
        plt.show()

    def plotEnergyDist(self, mt, mf, inc_group): # type = 2
        """
            plot energy distribution of secondary particle
        """
        if self._plot_type is None:
            self._plot_type = 2
        elif self._plot_type != 2:
            raise TypeError

        matrix = np.transpose(self.reactions[mt].mf[mf]._getTransMatrix()[:,:,0])
        data = matrix[:,inc_group]
        plt.step(self.egn[:-1] * 1e-6, data / (self.egn[1:] - self.egn[:-1]) * 1e6 , where='post')
        plt.title("material {}, energy distribution of secondary".format(self.za))
        plt.xscale("log")
        plt.xlabel("Energy (MeV)")
        plt.ylabel("#/MeV/event")
        
        energy = logMean(self.egn[inc_group], self.egn[inc_group+1])
        _legend = "{} {}, E={:.2e} eV".format(self.reactions[mt].__repr__(), particle_type[mf], energy)
        self._legend += [_legend]

    def plotAngularDist(self, mt, mf, inc_group, out_group, equiprob=False):
        """
            plot angular distribution of secondary particle
        """
        if self._plot_type is None:
            self._plot_type = 5
        elif self._plot_type != 5:
            raise TypeError    
        target_tape = self.reactions[mt].mf[mf].target_tape
        prob_map = self.reactions[mt].mf[mf].prob_map
        if target_tape[inc_group][0] < 0: # 0 cross section error
            first_group = np.argmax(target_tape[:,0] >= 0)
            raise ValueError("lowest incident energy group for MT={} and MF={} is {}".format(mt, mf, first_group))

        # get possible outgoing particle energy group range
        group_floor = target_tape[inc_group,1]
        if inc_group == len(target_tape) - 1: #last element
            group_ceil = group_floor + len(prob_map) - target_tape[inc_group,0]
        else:
            group_ceil = group_floor + target_tape[inc_group+1,0] - target_tape[inc_group,0]

        if out_group < group_floor or out_group >= group_ceil: # outgoing particle energy group out of range error
            raise ValueError("outgoing particle energy group must be in ({}, {})".format(group_floor, group_ceil))

        cpoint = target_tape[inc_group,0]
        legendre = prob_map[cpoint + out_group - group_floor,1:]

        if equiprob:
            """
            modifier = (np.arange(0, len(legendre), 1) * 2 + 1) / 2
            nbin = int(ENV["equiprob_nbin"])
            angle_bin, area_tot = legendreToEquibin(legendre * modifier, nbin)
            y = 1 / (angle_bin[1:] - angle_bin[:-1]) / nbin
            y = np.append(y, y[-1]) 
            plt.step(angle_bin, y * area_tot, where="post")
            """
            nbin = int(ENV["equiprob_nbin"])
            angle_bin = self.reactions[mt].mf[mf].equiprob_map[cpoint + out_group - group_floor,1:]
            y = 1 / (angle_bin[1:] - angle_bin[:-1]) / nbin
            y = np.append(y, y[-1]) 
            plt.step(angle_bin, y, where="post")

        else:
            x = np.linspace(-1, 1, 201)
            y = np.zeros(x.shape)
            for i, val in enumerate(legendre):
                y += eval_legendre(i, x) * (2 * i + 1) / 2 * val

            plt.plot(x, y)

        plt.title("material " + str(self.za) + ", angular distribuion")
        plt.xlabel("cosine")
        plt.ylabel("normalized probability")
        energy_in = logMean(self.egn[inc_group], self.egn[inc_group+1])
        energy_out = logMean(self.egn[out_group], self.egn[out_group+1])
        _legend = "{} {}, E_in={:.2e}, E_out={:.2e}".format(self.reactions[mt].__repr__(), particle_type[mf], energy_in, energy_out)
        self._legend += [_legend]
    
    def plotGammaMultiplicity(self, mt): # type = 3
        """
            plot gamma multiplicity
        """
        if self._plot_type is None:
            self._plot_type = 3
        elif self._plot_type != 3:
            raise TypeError     

        plt.step(self.egn[:-1] * 1e-6, self.reactions[mt].mf[16]._getMultiplicity(),
                 where='post')
        plt.title("material {}, gamma multiplicity".format(self.za))
        plt.xscale("log")
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Multiplicity")   
        self._legend += [self.reactions[mt]]

    def plotGammaSpectrum(self, mt, inc_group): # type = 4
        """
            plot gamma spectrum
        """
        if self._plot_type is None:
            self._plot_type = 4
        elif self._plot_type != 4:
            raise TypeError   

        spectrum = self.reactions[mt].mf[16]._getTransMatrix(len(self.egg)-1)[inc_group]
        spectrum *= self.reactions[mt].mf[16]._getMultiplicity()[inc_group]

        plt.step(self.egg[:-1] * 1e-6, spectrum * 100, where='post') 
        plt.title("material {}, gamma spectrum".format(self.za))  
        plt.xscale("log")
        plt.xlabel("Energy (MeV)")
        plt.ylabel("Yield (%)")   

        energy = logMean(self.egn[inc_group], self.egn[inc_group+1])
        _legend = "{}, E={:.2e} eV".format(self.reactions[mt].__repr__(), energy)
        self._legend += [_legend]

    def getGammaMeanEnergy(self, mt, inc_group):
        spectrum = self.reactions[mt].mf[16]._getTransMatrix(len(self.egg)-1)[inc_group]
        spectrum *= self.reactions[mt].mf[16]._getMultiplicity()[inc_group]
        if np.sum(spectrum) == 0:
            return 0 

        vfunc = np.vectorize(logMean)
        egg_energy = vfunc(self.egg[1:], self.egg[:-1])
        return np.sum(spectrum * egg_energy)
    
    def getHadronMeanEnergy(self, mt, mf, inc_group):
        spectrum = self.reactions[mt].mf[mf]._getTransMatrix()[inc_group,:,0]
        vfunc = np.vectorize(logMean)
        neu_energy = vfunc(self.egn[1:], self.egn[:-1])
        return np.sum(spectrum * neu_energy)

    def _samplingEnergy(self, mt, mf, inc_group):
        """
            sampling secondary particle energy
            with log uniform assumption
        """
        target_tape = self.reactions[mt].mf[mf].target_tape
        prob_map = self.reactions[mt].mf[mf].prob_map

        target_pointer, target_group = target_tape[inc_group]
        if target_pointer < 0: # no cross section
            # min_group = np.argmax(target_tape[:,0] >= 0)
            # raise ValueError("incident particle group must be larger than {}".format(min_group))
            return 0

        rand = np.random.random()
        while True:
            if rand < prob_map[target_pointer, 0]:
                break
            target_pointer += 1
            target_group += 1
        rand = np.random.random()
        return np.exp(np.log(self.egn[target_group]) * rand + np.log(self.egn[target_group+1]) * (1-rand))
        

class GENDF(GendfInterface):
    def __init__(self, file_name):
        super().__init__()

        file = open(file_name)
        # header attributes
        self.title = None
        self.za = None
        self.awr = None
        self.nz = None
        self.temp = None
        self.sigz = None

        self._mf3_finished = False

        # read header
        self._tpidio(file)
        self._headio(file)

        # read continue cards iteratively
        while True:
            is_eof = self._contio(file)
            if is_eof:
                break

        file.close()

        # check MF3 and MF6Like integrity
        del_target = []
        for mt in self.reactions:
            if 3 not in self.reactions[mt].mf:
                del_target += [mt]
                continue
            reaction = self.reactions[mt]
            xs = reaction.mf[3].xs
            for igroup in range(len(xs)):
                if xs[igroup] == 0:
                    continue
                for mf in reaction.mf:
                    if mf == 3:
                        continue
                    if mf in (6, 21):
                        if reaction.mf[mf].target_tape[igroup, 0] == -1:
                            xs[igroup] = 0.e0
        for mt in del_target:
            del self.reactions[mt]

        del_target = []
        for mt in self.reactions:
            if np.sum(self.reactions[mt].mf[3].xs) == 0:
                del_target += [mt]
        for mt in del_target:
            del self.reactions[mt]

        # check MT3 and MF16 exist
        # and copy MF16 to each nonelastic reaction if MT3 exist
        if 3 in self.reactions:
            if 16 in self.reactions[3].mf:
                for mt in reaction_nonelastic:
                    if mt in self.reactions:
                        self._getMT3Gamma(mt)
            else:
                return

        # neutron nonelastic
        for mt in self.reactions:
            if mt in reaction_secondary_neutron:
                self._getMT3Gamma(mt)

        # proton nonelastic
        for mt in self.reactions:
            if mt in reaction_secondary_proton:
                self._getMT3Gamma(mt)        

    def _getMT3Gamma(self, mt):
        self.reactions[mt].mf[16] = deepcopy(self.reactions[3].mf[16])
        group_min = np.argmax(self.reactions[mt].mf[3].xs > 0)
        cutoff, _, start_point = self.reactions[mt].mf[16].target_tape[group_min]
        
        """
        # cut off target tape
        self.reactions[mt].mf[16].target_tape[group_min:, 0] -= cutoff
        self.reactions[mt].mf[16].target_tape[:group_min, 0] = -1
        self.reactions[mt].mf[16].target_tape[:group_min, 1] = 0
        self.reactions[mt].mf[16].target_tape[:group_min, 2] = -1
        
        # cut off prob map
        self.reactions[mt].mf[16].prob_map = self.reactions[mt].mf[16].prob_map[start_point:]
        """
            
    def _tpidio(self, buffer):
        seg = parser(buffer, mode="str")
        self.title = seg[0]

    def _headio(self, buffer):
        # first line
        seg = parser(buffer)
        self.za  = int(seg[0][0])
        self.awr = seg[0][1]
        self.nz  = int(seg[0][3])
        ntw = int(seg[0][5])
        
        # second line
        seg = parser(buffer)
        self.temp = seg[0][0]
        ngw = int(seg[0][2])
        ngg = int(seg[0][3])
        nw =  int(seg[0][4])

        # fields
        field = np.empty(nw, dtype=np.float64)
        for i in range(int(np.ceil(nw/6))):
            seg = parser(buffer)
            field[6*i:min(6*i+6, nw)] = seg[0]
        scope0 = ntw
        scope1 = ntw + self.nz
        scope2 = ntw + self.nz + ngw + 1
        scope3 = ntw + self.nz + ngw + ngg + 2
        self.sigz = field[scope0:scope1]
        self.egn  = field[scope1:scope2]
        self.egg  = field[scope2:scope3]

        # check end line of MT=451
        seg = parser(buffer)
        if seg[2] != 0 or seg[3] != 0 or seg[4] != 0:
            raise ValueError("MT451 parsing EOF error!")

    def _contio(self, buffer):
        #first line
        seg = parser(buffer)
        if seg[3] == 0 and seg[4] == 0:
            return True
        nl = int(seg[0][2])            # number of Legendre components
        nz = int(seg[0][3])            # number of sigma-zero values
        lrflag = int(seg[0][4])        # break-up identifier flag
        ngn = int(seg[0][5])           # number of groups

        mf = int(seg[2])
        mt = int(seg[3])

        is_first = True
        list_first = True

        ng2 = None
        ig2lo = None
        nw = None
        ig = None

        while True:
            seg = parser(buffer)
            
            if seg[3] == 0:
                if mf > 3:
                    if self._mf3_finished == False: # generate MT=3 xs
                        self._mf3_finished = True
                        self._genXsMT3()
                if mt not in self.reactions.keys():
                    self.reactions[mt] = Reaction(mt)
                self.reactions[mt]._add(self.egn, mf, nz, lrflag, data, label)
                if seg[4] == 0:
                    return True
                return False
            if list_first:
                list_first = False
                pointer = 0
                ng2 = int(seg[0][2])     # number of secondary positions
                ig2lo = int(seg[0][3])   # index of lowest nonzero group
                nw = int(seg[0][4])      # number of words in LIST
                                         # (nw = nl * nz * ng2)
                ig = int(seg[0][5])      # group index for this record

                data_segment = np.empty(nw, dtype=np.float64)
                if is_first:
                    is_first = False
                    # data = np.zeros((len(self.egn)-ig2lo,nw), dtype=np.float64)
                    data = []
                    label = []
                    # label[:,0] = card position,
                    # label[:,1] = group index for this record,
                    # label[:,2] = lowest energy group)
            else:
                data_segment[6*pointer:6*pointer+min(6,nw)] = seg[0]
                nw -= 6
                if nw <= 0:
                    list_first = True
                    label += [[len(data), ig, ig2lo]]
                    data += [np.reshape(data_segment, (len(data_segment) // nl, nl))]
                pointer += 1
        
    def _genXsMT3(self):
        self.reactions[3] = Reaction(3)
        is_first = True
        for mt in reaction_nonelastic:
            if mt in self.reactions:
                if is_first:
                    self.reactions[3].mf[3] = deepcopy(self.reactions[mt].mf[3])
                    is_first = False
                else:
                    self.reactions[3].mf[3].xs += self.reactions[mt].mf[3].xs
        if is_first:
            del self.reactions[3]
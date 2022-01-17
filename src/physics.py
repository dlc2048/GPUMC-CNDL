import numpy as np
from scipy import integrate

if __name__ == "__main__":
    from constants import *
    from endf_io import particle_type, reaction_multiplicity
    from algorithm import *
else:
    from src.constants import *
    from src.endf_io import particle_type, reaction_multiplicity
    from src.algorithm import *


class Hadron:
    def __init__(self, z, a):
        self.z = z
        self.a = a
        self.type = checkParticleType(z, a)

    def __repr__(self):
        return particle_type[self.type]

    def __add__(self, hadron):
        z = self.z + hadron.z
        a = self.a + hadron.a
        return Hadron(z, a)

    def __sub__(self, hadron):
        z = self.z - hadron.z
        a = self.a - hadron.a
        return Hadron(z, a)

def checkParticleType(z, a):
    if z == 0:
        if a == 1:
            return 6
        else:
            return -1
    elif z == 1:
        if a == 1:
            return 21
        elif a == 2:
            return 22
        elif a == 3:
            return 23
        else:
            return -1
    elif z == 2:
        if a < 3:
            return -1
        elif a == 3:
            return 24
        elif a == 4:
            return 25
        else:
            return -1
    else:
        return 26

predefined_hadron = {6 : Hadron(0, 1),
                     21: Hadron(1, 1),
                     22: Hadron(1, 2),
                     23: Hadron(1, 3),
                     24: Hadron(2, 3),
                     25: Hadron(2, 4)}

def checkResNucType(za, mt):
    """
        check type of residual nuclei
        when material (ZA) takes (MT=mt) reaction
    """
    z = za // 1000
    a = za - z * 1000
    target = Hadron(z, a)
    # neutron incident
    target += predefined_hadron[6]
    for particle in reaction_multiplicity[mt]:
        for _ in range(reaction_multiplicity[mt][particle]):
            target -= predefined_hadron[particle]
    return target.type

def labMu(r, A):
    """
        get scattering angle (lab) when outgoing energy (E') is rE
    """
    return ((A+1)**2*r + 1 - A**2)/(2*(A+1))/np.sqrt(r)

def labMuCumul(mu, A, c):
    """
        get cumulative probability function of mu when M/m = A
        offset is c
    """
    if mu >= 0:
        return -1 / 2 / A * (A + 1 - mu**2 - np.sqrt(mu**4 + mu**2 * (A**2 - 1))) - c
    else:
        return -1 / 2 / A * (A + 1 - mu**2 + np.sqrt(mu**4 + mu**2 * (A**2 - 1))) - c


class ThermalScattering2D:
    def __init__(self, A, T, E, thres):
        self.A = A
        self.T = T
        self.E = E
        self.thres = thres
        self.deno = integrate.dblquad(self._kernel, 
                                      -self.E/BOLTZMANN/self.T, 
                                      (self.thres-self.E)/BOLTZMANN/self.T,
                                      self._aMin,
                                      self._aMax,
                                      epsabs=1e-5,
                                      epsrel=1e-5)[0]

    def _kernel(self, a, b):
        """
            thermal neutron scattering kernel (free gas)
        """
        #return 1/np.sqrt(a) * np.exp(-b/2) * np.exp(-(a+b)**2/4/a)
        return 1/np.sqrt(a) * np.exp(-(a+b)**2/4/a)

    def _aMin(self, b):
        return (np.sqrt(self.E+b*BOLTZMANN*self.T) - np.sqrt(self.E))**2/(self.A*BOLTZMANN*self.T)

    def _aMax(self, b):
        return (np.sqrt(self.E+b*BOLTZMANN*self.T) + np.sqrt(self.E))**2/(self.A*BOLTZMANN*self.T)

    def getProb(self, emin, emax):
        """
            get probability that scattered particle energy is in (emin, emax)
        """

        nume = integrate.dblquad(self._kernel, 
                                 (emin-self.E)/BOLTZMANN/self.T, 
                                 (emax-self.E)/BOLTZMANN/self.T,
                                 self._aMin,
                                 self._aMax,
                                 epsabs=1e-5,
                                 epsrel=1e-5)[0]
        return nume / self.deno
    

class ThermalScattering1D:
    def __init__(self, A, T, E, E_out):
        self.A = A
        self.T = T
        self.E = E
        self.E_out = E_out
        self.b = (E_out - E) / BOLTZMANN / T
        self.deno = integrate.quad(self._kernel, self._a(1), self._a(-1))[0]

    def _kernel(self, a):
        """
            thermal neutron scattering kernel (free gas)
        """
        #return 1/np.sqrt(a) * np.exp(-b/2) * np.exp(-(a+b)**2/4/a)
        return 1/np.sqrt(a) * np.exp(-(a+self.b)**2/4/a)    

    def _a(self, mu):
        return (self.E + self.E_out - 2*np.sqrt(self.E*self.E_out)*mu)/self.A/BOLTZMANN/self.T

    def getProb(self, mu_min, mu_max):
        nume = integrate.quad(self._kernel, self._a(mu_max), self._a(mu_min))[0]
        if self.deno > 0:
            return nume / self.deno
        else: # integrate is 0, assuming isotropic
            return (mu_max - mu_min) / 2


class MF4AngularDistribution:
    def __init__(self, A, angular_distribution):
        self._A = A
        self._a = (A-1)**2/(A+1)**2
        self._ad = angular_distribution # must be pyne.endf.Evaluate.reactions[2].angular_distribution

    def _getAngularDist(self, inc_energy):
        if self._ad.type == 'tabulated': # tabulated dataset
            t = np.argmax(self._ad.energy > inc_energy)
            # get concatenate tabulated point
            angle_point = np.unique(np.concatenate((self._ad.probability[t-1].x, self._ad.probability[t].x)), 0)
            tab_lower = interp1d(self._ad.probability[t-1].x, 
                                 self._ad.probability[t-1].y, 2).get(angle_point)
            tab_upper = interp1d(self._ad.probability[t].x, 
                                 self._ad.probability[t].y, 2).get(angle_point)
            pyfunc = lambda en_low, en_up, ang_low, ang_up, ene: interp1d([en_low, en_up], [ang_low, ang_up], 4).get(ene)
            vfunc = np.vectorize(pyfunc)
            return angle_point, vfunc(self._ad.energy[t-1], self._ad.energy[t],
                                      tab_lower, tab_upper, inc_energy)

    def getArea(self, inc_energy, en_floor, en_ceil):
        mu_floor = max(energyToMuCM(self._a, inc_energy, en_floor), -1)
        mu_floor = min(mu_floor, 1)
        mu_ceil = min(energyToMuCM(self._a, inc_energy, en_ceil), 1)
        mu_ceil = max(mu_ceil, -1)
        if self._ad.type == 'tabulated': # tabulated dataset
            angle_point, prob_point = self._getAngularDist(inc_energy)
            area_min = getInterpFtnCumulArea(angle_point, prob_point, mu_floor)
            area_max = getInterpFtnCumulArea(angle_point, prob_point, mu_ceil)
        return area_max - area_min                

    def getEquiAngularBin(self, inc_energy, en_floor, en_ceil, nbin):
        mu_floor = max(energyToMuCM(self._a, inc_energy, en_floor), -1)
        mu_floor = min(mu_floor, 1)
        mu_ceil = min(energyToMuCM(self._a, inc_energy, en_ceil), 1)
        mu_ceil = max(mu_ceil, -1)
        if self._ad.type == 'tabulated': # tabulated dataset
            angle_point, prob_point = self._getAngularDist(inc_energy)
            area_min = getInterpFtnCumulArea(angle_point, prob_point, mu_floor)
            area_max = getInterpFtnCumulArea(angle_point, prob_point, mu_ceil)
            
            area_target = np.linspace(area_min, area_max, nbin+1, dtype=np.float64)
            mu_cm = np.empty(area_target.shape, dtype=np.float64)
            mu_cm[0] = mu_floor
            mu_cm[-1] = mu_ceil
            for i in range(1, len(mu_cm) - 1):
                mu_cm[i] = getInterpFtnCumulValue(angle_point, prob_point, area_target[i])

        # transform cm to lab
        return (1 + self._A*mu_cm) / np.sqrt(self._A**2 + 1 + 2*self._A*mu_cm)



def energyToMuCM(alpha, inc_energy, out_energy):
    return (2*out_energy/inc_energy - (1+alpha))/(1-alpha)


if __name__ == "__main__":
    t = checkResNucType(3007, 24)
    print(t)


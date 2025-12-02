# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


# pylint: disable=C0103, R0902, R0914, R0913, C0114, C0116

from math import exp
from scipy.special import erf
import numpy as np
from iapws import IAPWS97



class lu_mc_cartney:
    """Lu and McCartney (2024) soil water retention model for bentonite-based buffer materials.
    References:
    https://link.springer.com/article/10.1007/s11440-024-02341-9
    https://www.sciencedirect.com/science/article/pii/S0266352X25006858#b0050
    """
    def __init__(self, material='MX80 Bentonite'):
        self.R = 8.31432 #J/(mol·K)
        self.hp = 6.626068e-34 #J·s
        self.kB = 1.3806505e-23 #J/K
        self.T_0 = 273.15 #K
        
        self.material = material
        # material-specific parameters

        self.params = {'Boom Clay': {      'E1_minus_EL': np.float64(6803.853838682097),
                                           'rho_d': np.float64(1400.0),
                                           'CEC_max': np.float64(0.2534798545419938),
                                           'b_w_T': [22+self.T_0, 80+self.T_0],
                                           'b_w': [0.11, 0.11],
                                           'nu_mr': np.float64(7.994125e-02),
                                           'phi': 0.483,
                                           'SSA': 260e3,
                                           'epsilon_s': 1.3267932723717908,
                                           'n_s': 35.435652398543425,
                                           'eta_alpha': 0.6,
                                           'alpha_0_inv': 36e3,
                                           'M': 0.085,
                                           'N': 1.29},
                       'FEBEX Bentonite': {'E1_minus_EL': np.float64(7904.547103777231),
                                           'rho_d': np.float64(1500.0),
                                           'CEC_max': np.float64(1.0108839312584308),
                                           'b_w_T': [26+self.T_0, 80+self.T_0],
                                           'b_w': [0.18, 0.18],
                                           'nu_mr': np.float64(3.374814e-01),
                                           'phi': 0.443,
                                           'SSA': 860e3,
                                           'epsilon_s': 0.940367803768519,
                                           'n_s': 748.7804881357005,
                                           'eta_alpha': 3.5,
                                           'alpha_0_inv': 330e3,
                                           'M': 0.132,
                                           'N': 1.22},
                       'MX80 Bentonite': {'E1_minus_EL': np.float64(7533.930402019812),
                                          'rho_d': np.float64(1600.0),
                                          'CEC_max': np.float64(0.8004082526701221),
                                          'b_w_T': [20+self.T_0, 60+self.T_0],
                                          'b_w': [0.18, 0.18],
                                          'nu_mr': np.float64(2.752035e-01),
                                          'phi': 0.432,
                                          'SSA': 700e3,
                                          'epsilon_s': 3.208925112257479,
                                          'n_s': 1.3298392371979677,
                                          'eta_alpha': 5.0,
                                          'alpha_0_inv': 3.3e6,
                                          'M': 0.105,
                                          'N': 1.15},
                       'GMZ01 Bentonite': {'E1_minus_EL': np.float64(8369.804359014617),
                                           'rho_d': np.float64(1700.0),
                                           'CEC_max': np.float64(0.8004082526701221),
                                           'b_w_T': [20+self.T_0, 40+self.T_0, 60+self.T_0, 80+self.T_0, ],
                                           'b_w': [0.25, 0.25, 0.25, 0.25],
                                           'nu_mr': np.float64(1.505956e-01),
                                           'phi': 0.359,
                                           'SSA': 700e3,
                                           'epsilon_s': 2.1590265530454893,
                                           'n_s': 2175.885110079861,
                                           'eta_alpha': 8.5,
                                           'alpha_0_inv': 8.0e6,
                                           'M': 0.135,
                                           'N': 1.3}
                       }
        # parameters for adsorptive water content
        self.nu_w = 1.8e-5 #m^3/mol
        self.E1_minus_EL = self.params[material]['E1_minus_EL'] #J/mol
        #self.zeta_s = 2.8 #meq/g
        self.nu_mr = self.params[material]['nu_mr']
        self.CEC_max = self.params[material]['CEC_max'] #meq/g
        self.A = 0.92
        self.B = 0.9
        self.T1 = 373.15 #K
        self.T2 = 1273.15 #K
        self.M = self.params[material]['M']

        # parameters for capillary water content
        self.SSA = self.params[material]['SSA'] #m^2/kg
        #self.AH = 10213000 #Pa
        self.epsilon_s = self.params[material]['epsilon_s']
        self.nu_e = 2.45e9 #Hz
        self.n_s = self.params[material]['n_s']
        self.n_w = 1.33
        self.alpha_0_inv = self.params[material]['alpha_0_inv'] #Pa
        self.Tr = 293.14 # K
        self.eta_alpha = self.params[material]['eta_alpha']
        self.C_1 = -0.00151
        self.N = self.params[material]['N']


        # material properties
        self.phi = self.params[material]['phi'] #m3/m3
        self.rho_d = self.params[material]['rho_d'] #kg/m3
        self.theta_s = self.phi #m3/m3

    def water_content(self, psi, T):
        theta_ad = self.adsorptive_water_content(psi, T)
        theta_cap = self.capillary_water_content(psi, T)
        return theta_ad + theta_cap

    def adsorptive_water_content(self, psi, T):
        arg = (psi - self.psi_max(T))/psi
        term = np.exp(arg)
        return self.theta_a_max(T) * (1-term**self.M)

    def capillary_water_content(self, psi, T):
        theta_mean = (self.theta_s - self.adsorptive_water_content(psi, T))/2
        chi_frac = (self.chi(T) + T) / (self.chi_r + self.Tr)
        erf_term = erf(np.sqrt(2)* (psi/self.psi_c(T)*chi_frac-1))
        power_term = (self.alpha(T) * psi * chi_frac)**self.N
        cap_term = theta_mean * (1 - erf_term) * (1 + power_term)**((1/self.N)-1)
        return cap_term

    def theta_a_max(self, T):
        return (1-self.theta_s) * (self.CEC(T) / self.zeta_s(T) + self.bw(T))


    def psi_max(self, T):
        umax = self.R * T * self.c(T) / (3.0 * self.nu_w)
        return umax

    def bw(self, T):
        return np.interp(T, self.params[self.material]['b_w_T'], self.params[self.material]['b_w'])

    def c(self, T):
        return np.exp((self.E1_minus_EL) / (self.R * T))

    def CEC(self, T):
        cos_arg = self.A * np.pi * (self.B*T - self.T1) / (self.T2 - self.T1)
        return 0.5* self.CEC_max * (np.cos(cos_arg) + 1)

    def psi_c(self, T):
        return self.A_H(T)/ (6 * np.pi) * (self.theta_a_max(T) / ( self.SSA * self.rho_d))**(-3.0)

    def A_H(self, T):
        dielectric_term = ((self.epsilon_s - self.epsilon_w(T))/(self.epsilon_s + self.epsilon_w(T)))**2
        RI_term = (self.n_s**2 - self.n_w**2)**2 / (self.n_s**2 + self.n_w**2)**(3/2)
        prefactor_dielectic = (3 * self.kB * T)/4
        prefactor_RI = (3 * self.hp * self.nu_e)/ (16 * np.sqrt(2))
        return prefactor_dielectic * dielectric_term + prefactor_RI * RI_term

    def chi(self, T):
        return self.delta_h(T)/self.C_1

    @property
    def chi_r(self):
        #sign error in paper?
        return self.delta_h(self.Tr)/self.C_1

    def delta_h(self, T):
        delta_hr = -0.516 # J/m^2 high plasticity clay
        delta_hr = -0.5151936137548038
        return delta_hr * ((1 - self.Tr)/(1 - T))**0.38

    def alpha(self, T):
        psi_aev = self.alpha_0_inv * exp(self.eta_alpha * (self.Tr / T - 1))
        return 1.0 / psi_aev

    def epsilon_w(self, T):
        #Tc = T-273.15
        #return (87.914 - 0.404399*T + 9.5877e-4 * T**2 - 1.3281e-6 * T**3)/(1.0 + 1.949e-3*T - 1.016e-6*T**2)
        return 10**(0.7017+642.0/T-1.167e5/T**2+9.190e6/T**3+(1.667-11.41/T-3.526e4/T**2)*np.log10(self.rho_w(T)/1000))+1

    def zeta_s(self, T):
        return (self.c(T)-1)/self.c(T) * self.CEC(T)/self.nu_mr
    
    def rho_w(self, T):
        water = IAPWS97(T=T, P=10)
        return water.rho
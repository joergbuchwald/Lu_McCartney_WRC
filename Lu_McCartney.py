# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


# pylint: disable=C0103, R0902, R0914, R0913, C0114, C0116

from math import exp
from scipy.special import erf
import numpy as np


class lu_mc_cartney:
    """Lu and McCartney (2024) soil water retention model for bentonite-based buffer materials.
    References:
    https://link.springer.com/article/10.1007/s11440-024-02341-9
    https://www.sciencedirect.com/science/article/pii/S0266352X25006858#b0050
    """
    def __init__(self):
        self.R = 8.314 #J/(mol·K)
        self.hp = 6.626e-34 #J·s
        self.kB = 1.381e-23 #J/K
        # parameters for adsorptive water content
        self.nu_w = 1.8e-5 #m^3/mol
        self.E1_minus_EL = 7534 #J/mol
        self.zeta_s = 2.8 #meq/g
        self.bw = 0.18 #g/g
        self.CEC_max = 0.776 #meq/g
        self.A = 0.92
        self.B = 0.9
        self.T1 = 373.15 #K
        self.T2 = 1273.15 #K
        self.M = 0.12

        # parameters for capillary water content
        self.SSA = 7e5 #m^2/kg
        self.AH = 10213000 #Pa
        self.epsilon_s = 3.5
        self.n_s = 0.55
        self.n_w = 1.33
        self.alpha_0_inv = 3.3e6 #Pa
        self.Tr = 293.14 # K
        self.eta_alpha = 5.0
        self.C_1 = -0.00151
        self.N = 1.15


        # material properties
        self.phi = 0.45 #m3/m3
        self.rho_d = self.phi * 2660
        self.theta_s = self.phi #m3/m3
        self.params = {}

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
        erf_term = erf(np.sqrt(2)* (psi/self.psi_c(T)*chi_frac - 1))
        power_term = (self.alpha(T) * psi * chi_frac)**self.N
        cap_term = theta_mean * (1 - erf_term) * (1 + power_term)**(1/self.N-1)
        return cap_term

    def theta_a_max(self, T):
        return (1-self.theta_s) * (self.CEC(T) / self.zeta_s + self.bw)


    def psi_max(self, T):
        umax = self.R * T * self.c(T) / (3000.0 * self.nu_w)
        return umax

    def c(self, T):
        if "c" in self.params:
            return self.params['c']
        return np.exp((self.E1_minus_EL) / (self.R * T))

    def CEC(self, T):
        cos_arg = self.A * np.pi * (self.B*T - self.T1) / (self.T2 - self.T1)
        return self.CEC_max * (np.cos(cos_arg) + 1)

    def psi_c(self, T):
        return self.A_H(T)/(6*np.pi) * (self.theta_a_max(T) / (self.SSA * self.rho_d))**(-3)

    def A_H(self, T):
        dielectric_term = (self.epsilon_s - self.epsilon_w(T))/(self.epsilon_s + self.epsilon_w(T))**2
        pre_factor1 = (3 * self.kB * T)/4
        pre_factor2 = 3 * (self.hp * self.nu_w)/(16 * np.sqrt(2))
        RI_term = ((self.n_s**2 - self.n_w**2)**2/(self.n_s**2 + self.n_w**2)**(3/2))
        return pre_factor1 * dielectric_term + pre_factor2 * RI_term

    def chi(self, T):
        return -self.delta_h(T)/self.C_1

    @property
    def chi_r(self):
        return -self.delta_h(self.Tr)/self.C_1

    def delta_h(self, T):
        delta_hr = -0.516 # J/m^2 high plasticity clay
        return delta_hr * ((1 - self.Tr)/(1 - T))**0.38

    def alpha(self, T):
        psi_aev = self.alpha_0_inv * exp(self.eta_alpha * (self.Tr / T - 1))
        return 1.0 / psi_aev

    def epsilon_w(self, T):
        Tc = T-273.15
        return (87.914 - 0.404399*T + 9.5877e-4 * T**2 - 1.3281e-6 * T**3)/(1.0 + 1.949e-3*T - 1.016e-6*T**2)

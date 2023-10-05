import numpy as np
from scipy import integrate
from scipy import special
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

def N_cen(M_h, M_min, sigma_logM):
    return 0.5*(1+special.erf((np.log10(M_h)-np.log10(M_min))/(np.sqrt(2)*sigma_logM)))

def N_sat(M_h, M_sat, alpha, M_min, sigma_logM):
    M_cut = np.power(M_min, -0.5)
    return  N_cen(M_h, M_min, sigma_logM)*np.power((M_h-M_cut)/M_sat,alpha)

def N_tot(M_h, M_sat, alpha, M_min, sigma_logM):
    return N_cen(M_h, M_min, sigma_logM) + N_sat(M_h, M_sat, alpha, M_min, sigma_logM)

def HMF(M_h, z): 
    return 1

def n_g_INTEGRAND(M_h, M_min, sigma_logM, M_sat, alpha, z):
    return HMF(M_h, z)*N_tot(M_h, M_sat, alpha, M_min, sigma_logM)

def n_g(M_min, sigma_logM, M_sat, alpha, z):
    return integrate.quad(n_g_INTEGRAND, 1e8, 1e13, args=(M_min, sigma_logM, M_sat, alpha, z))[0]

def u_FT(k, M_h, z):
    r_200 = np.power(M_h/(4/3*np.pi)/(200*cosmo.critical_density(z).value*2e31),1/3)

    si, ci = special.sici(k*r_s)
    si_c, ci_c = special.sici(k*r_s*(1+c))
    return 3*(np.sin(k*r_s)*())

def PS_1h_cs_INTEGRAND(M_h, k, M_min, sigma_logM, M_sat, alpha, z):
    return HMF(M_h, z)*N_cen(M_h, M_min, sigma_logM)*N_sat(M_h, M_sat, alpha, M_min, sigma_logM)*u_FT(k, M_h, z)

def PS_1h_cs(k, M_min, sigma_logM, M_sat, alpha, z):
    intg = integrate.quad(PS_1h_cs_INTEGRAND, 1e6, 1e15, args=(k, M_min, sigma_logM, M_sat, alpha, z))[0]
    return intg*2/np.power(n_g(M_min, sigma_logM, M_sat, alpha, z),2)

def PS_1h_ss_INTEGRAND(M_h, k, M_min, sigma_logM, M_sat, alpha, z):
    return HMF(M_h, z)*np.power(N_sat(M_h, M_sat, alpha, M_min, sigma_logM),2)*np.power(u_FT(k, M_h, z),2)

def PS_1h_ss(k, M_min, sigma_logM, M_sat, alpha, z):
    intg = integrate.quad(PS_1h_ss_INTEGRAND, 1e6, 1e15, args=(k, M_min, sigma_logM, M_sat, alpha, z))[0]
    return intg*1/np.power(n_g(M_min, sigma_logM, M_sat, alpha, z),2)
    
def PS_1h(k, M_min, sigma_logM, M_sat, alpha, z):
    return PS_1h_cs(k, M_min, sigma_logM, M_sat, alpha, z)+PS_1h_ss(k, M_min, sigma_logM, M_sat, alpha, z)

def omega_INTEGRAND_z_k(k, z, theta, M_min, sigma_logM, M_sat, alpha):
    I_z = np.power(n_g(M_min, sigma_logM, M_sat, alpha, z),2)*(cosmo.H(z).value/299792.458)
    I_k = PS_1h(k, M_min, sigma_logM, M_sat, alpha, z)*k/(2*np.pi)*special.j0(k*theta*cosmo.comoving_distance(z).value)
    return I_z*I_k

def omega(theta, M_min, sigma_logM, M_sat, alpha, z_c, z_bin):
    return integrate.dblquad(omega_INTEGRAND_z_k, z_c-z_bin/2, z_c+z_bin/2, lambda x: 0, lambda x: 1e4, args=(theta, M_min, sigma_logM, M_sat, alpha))[0]


'''
### FIXED ##################
sigma_logM, alpha = 0.2, 1.0
############################
### Harikane Goldrush IV ###
z_c, z_bin = 1.7, 0.5
log_M_min  = 12.46 
log_M_sat  = 14.18
############################
rad_to_arcsec = 1/206265
theta = 1*rad_to_arcsec # 1 arcsec in rad
HOD.omega(theta, 10**(log_M_min), sigma_logM, 10**(log_M_sat), alpha, z_c, z_bin) 
'''
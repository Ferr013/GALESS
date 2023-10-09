import numpy as np
from scipy import integrate
from scipy import special
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

def N_cen(M_h, M_min, sigma_logM):
    return 0.5*(1+special.erf((np.log10(M_h)-np.log10(M_min))/(np.sqrt(2)*sigma_logM)))

def N_sat(M_h, M_sat, alpha, M_min, sigma_logM):
    M_cut = np.power(M_min, -0.5)
    return  N_cen(M_h, M_min, sigma_logM)*np.power((M_h-M_cut)/M_sat,alpha)

def N_tot(M_h, M_sat, alpha, M_min, sigma_logM):
    return N_cen(M_h, M_min, sigma_logM) + N_sat(M_h, M_sat, alpha, M_min, sigma_logM)
    
########################################################################################################################
def k_space(kr):
    return np.where(kr > 1.4e-6, (3 / kr**3) * (np.sin(kr) - kr * np.cos(kr)), 1)

def dw_dlnkr(kr):
    return np.where(kr > 1e-3,(9 * kr * np.cos(kr) + 3 * (kr**2 - 3) * np.sin(kr)) / kr**3,0,)

def power_spectrum(k):
    n_eff = 0.96
    norm  = 0.008298892589156328 #for 0.96
    return np.power(k, n_eff)/norm

def _sigma(M_h, z):
    rho_m = cosmo.Om(z)*cosmo.critical_density(z).value*200
    r = 2.15e10*(3.0 * M_h / (4.0 * np.pi * rho_m)) ** (1.0 / 3.0)
    k = np.logspace(0, 1/r, 100)
    dlnk = np.log(k[1]/k[0])
    rk = np.outer(r, k)
    return integrate.simps(power_spectrum(k) * np.power(k,2) * k_space(rk), dx=dlnk, axis=-1)

def _dlnsdlnm(M_h, z):
    """ math:: frac{d\ln\sigma}{d\ln m} = \frac{3}{2\sigma^2\pi^2R^4}\int_0^\infty \frac{dW^2(kR)}{dM}\frac{P(k)}{k^2}dk """
    rho_m = cosmo.Om(z)*cosmo.critical_density(z).value*200 
    r = 2.15e10*(3.0 * M_h / (4.0 * np.pi * rho_m)) ** (1.0 / 3.0)
    k = np.logspace(0, 1/r, 100)
    dlnk = np.log(k[1]/k[0])
    sigma = _sigma(M_h, z)
    rk = np.outer(r, k)
    rest = power_spectrum(k)*np.power(k,3)
    w = k_space(rk)
    dw = dw_dlnkr(rk)
    integ = w * dw * rest
    dlnss_dlnr = integrate.simps(integ, dx=dlnk, axis=-1) / (np.pi**2 * sigma**2)
    return 0.5 * dlnss_dlnr * (1.0 / 3.0) #for the `m\propto r^3` mass assignment, this is just 1/3.

########################################################################################################################
def f_TINKER(sigma, z): 
	Dlt = 200
	A0, a0, b0, c0 = 0.186, 1.47, 2.57, 1.19
	alpha = 0.0106756286522959 #10**(-(0.75 / np.log10(200 / 75.0))**1.2)
	A = A0 * (1.0 + z)**-0.140
	a = a0 * (1.0 + z)**-0.060
	b = b0 * (1.0 + z)**-alpha
	return A * ((sigma / b)**-a + 1.0) * np.exp(-c0 / sigma**2)

def HMF(M_h, z):
    rho_m = 2.15e10*cosmo.Om(z)*cosmo.critical_density(z).value*200 #M_sun/kpc^3
    sigma = _sigma(M_h, z)
    return f_TINKER(sigma, z) * rho_m * np.abs(_dlnsdlnm(M_h, rho_m))/M_h**2

def n_g_INTEGRAND(M_h, M_min, sigma_logM, M_sat, alpha, z):
    return HMF(M_h, z)*N_tot(M_h, M_sat, alpha, M_min, sigma_logM)

def n_g(M_min, sigma_logM, M_sat, alpha, z):
    return integrate.quad(n_g_INTEGRAND, 1e8, 1e15, args=(M_min, sigma_logM, M_sat, alpha, z))[0]

def get_c_from_M_h(M_h, z):
    # Eq. 4. https://iopscience.iop.org/article/10.1086/367955/pdf
    c_norm = 8
    return (c_norm)/(1+z)*np.power(M_h/(1.4e14), -0.13)

def u_FT(k, M_h, z):
    r_v = np.power(M_h/(4/3*np.pi*cosmo.critical_density(z).value*200*2e31), 1/3) #rho = M_sun/kpc^3
    c   = get_c_from_M_h(M_h, z)
    f_c = np.log(1+c)-c/(1+c)
    r_s = r_v/c
    si, ci = special.sici(k*r_s)
    si_c, ci_c = special.sici(k*r_s*(1+c))
    return (np.sin(k*r_s)*(si_c-si)+np.cos(k*r_s)*(ci_c-ci)-(np.sin(c*k*r_s)/((1+c)*k*r_s)))/f_c

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
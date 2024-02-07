# pylint: disable-msg=C0103,W0611,E0611,W0613,R0914,R1705,R0913
"""Module providing a set of functions to evaluate the distributions of 
lenses and lensed sources in a given survey."""

import os.path
import numpy as np
from scipy import integrate as integral
from scipy import signal, stats
from scipy.special import gamma as gammafunc
from scipy.special import ellipe as E_ell
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM
import galess.Utils.ls_utils as utils

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

def schechter_LF(M_int,zs):
    '''
    Returns the Schechter UV galaxy Luminosity Function.
    Parameters are taken from Bouwens et al. 2022 
    link: (https://ui.adsabs.harvard.edu/abs/2022ApJ...940...55B/abstract)

            Parameters:
                    M_int: ndarray(dtype=float, ndim=1)
                        Absolute magnitude (mag)
                    zs: (float)
                        Redshift of the galaxy population
            Returns:
                    LF: ndarray(dtype=float, ndim=1) 
                        Schechter UV LF
    '''
    zt = 2.42
    phi_0 = 0.38*10**(-3.)*np.power(10,(-0.35*(zs-6)+(-0.027)*(zs-6)**2))
    alpha = -1.95 + (-0.11) * (zs - 6)
    if zs < zt:
        Mstar = -20.87+(-1.1*(zs-zt))
    else:
        Mstar = -21.04+(-0.05*(zs-6))
    nrmlzt = phi_0 * (np.log(10) / 2.5)
    pwrlaw = np.power(10, 0.4 * (Mstar - M_int) * (alpha + 1))
    expctf = np.exp(-1 * np.power(10, 0.4 * (Mstar - M_int)))
    return nrmlzt * pwrlaw * expctf

def Theta_E(sigma, zl, zs):
    '''
    Returns the Einstein radius of a Singular Isothermal lens.

            Parameters:
                    sigma: (float)
                        Velocity dispersion (km/s)
                    zl: (float)
                        Redshift of the lens
                    zs: (float)
                        Redshift of the source
            Returns:
                    Theta_E: (float)
                        Einstein radius (arcsec)
    '''
    v_ref = 161 #km/s per SIE = 0.9 arcsec
    Ds = cosmo.angular_diameter_distance(zs).value
    Dds = cosmo.angular_diameter_distance_z1z2(zl,zs).value
    return 0.9 * Dds/Ds * np.power (sigma/v_ref,2) #arcsec

def sigma_from_R_Ein(zs, zl, R_E):
    '''
    Returns the velocity dispersion given the Einstein radius and
    lens and source redshifts.
    The model assumes a Singular Isothermal lens.

            Parameters:
                    zl: (float)
                        Redshift of the lens
                    zs: (float)
                        Redshift of the source
                    Theta_E: (float)
                        Einstein radius (arcsec)
            Returns:
                    sigma: (float)
                        Velocity dispersion (km/s)
    '''
    v_ref = 161 #km/s per SIE = 0.9 arcsec
    Ds = cosmo.angular_diameter_distance(zs).value
    Dds = cosmo.angular_diameter_distance_z1z2(zl,zs).value
    return v_ref*np.power(R_E / 0.9 * Ds / Dds, 0.5)

def Phi_vel_disp_SDSS(sigma, zl = 0):
    '''
    Returns the velocity dispersion function (VDF) evolution with z.
    Parameters are taken from Choi et al. 2007 
    link: (https://ui.adsabs.harvard.edu/abs/2007ApJ...658..884C/abstract)

            Parameters:
                    sigma: (float)
                        Velocity dispersion (km/s)
                    zl: (float)
                        Redshift of the lens
            Returns:
                    VDF: (float)
                        Velocity Dispersion Function
    '''
    Phi_star = 8e-3 #Mpc^-3 * h^3
    Phi_star = 8e-3*(cosmo.H0.value/100)**3 #Mpc^-3
    alpha = 2.32
    beta = 2.67
    sigma_star = 161 #km/s
    pwrlaw = np.power(sigma/sigma_star, alpha)
    expctf = np.exp(-np.power(sigma/sigma_star,beta))
    gammav = gammafunc(alpha/beta)
    return Phi_star * pwrlaw * expctf / gammav * (beta/sigma)

def Phi_vel_disp_Mason(sigma, zl):
    '''
    Returns the velocity dispersion function (VDF) evolution with z.
    Parameters are taken from Mason et al. 2015 
    link: (https://ui.adsabs.harvard.edu/abs/2015ApJ...805...79M/abstract)

            Parameters:
                    sigma: (float)
                        Velocity dispersion (km/s)
                    zl: (float)
                        Redshift of the lens
            Returns:
                    VDF: (float)
                        Velocity Dispersion Function
    '''
    p = 0.24
    beta = 0.2
    alpha_s = -0.54
    Phi_star = 3.75*1e-3 #Mpc^-3
    sigma_star = 216 #km/s
    Phi_star_z = Phi_star*np.power(1+zl,-2.46)
    sigma_z = sigma*np.power(1+zl,beta)
    pwrlaw = np.power(sigma/sigma_star, (1+alpha_s)/p)
    expctf = np.exp(-np.power(sigma/sigma_star,1/p))
    return np.log(10) / p * (Phi_star_z/sigma_z) * pwrlaw * expctf

def Phi_vel_disp_Geng(sigma, zl):
    '''
    Returns the velocity dispersion function (VDF) evolution with z.
    Parameters are taken from Geng et al. 2021
    link: (https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.1319G/abstract)

            Parameters:
                    sigma: (float)
                        Velocity dispersion (km/s)
                    zl: (float)
                        Redshift of the lens
            Returns:
                    VDF: (float)
                        Velocity Dispersion Function
    '''
    Phi_star = 8e-3 #Mpc^-3 * h^3
    Phi_star = 8e-3*(cosmo.H0.value/100)**3 #Mpc^-3
    alpha = 2.32
    beta = 2.67
    sigma_star = 161 #km/s
    nu_n = -1.2
    nu_v = 0.2
    # P = -0.88
    # Q = 0.09
    #Phi_star_z = Phi_star*np.power(10,P*zl)
    #sigma_star_z = sigma_star*np.power(10,Q*zl)
    Phi_star_z = Phi_star*np.power(1+zl,nu_n)
    sigma_star_z = sigma_star*np.power(1+zl,nu_v)
    pwrlaw = np.power(sigma/sigma_star_z, alpha)
    expctf = np.exp(-np.power(sigma/sigma_star_z,beta))
    return Phi_star_z*pwrlaw*(expctf/gammafunc(alpha/beta))*(beta/sigma)

def dTau_dz_dsigma(sigma, zl, zs, Phi_vel_disp = Phi_vel_disp_Mason):
    '''
    Returns the differential multiple image optical depth (tau)
    over lens redshift and velcity dispersion

            Parameters:
                    sigma: (float)
                        Velocity dispersion (km/s)
                    zl: (float)
                        Redshift of the lens
                    zs: (float)
                        Redshift of the source
                    Phi_vel_disp: (function)
                        Velocity Dispersion Function
            Returns:
                    dTau_dz_dsigma: (float)
                        Differential multiple image optical depth
                        over dz and dsigma
    '''
    rad_to_arcsec = 1/206265
    c_sp = 299792.458 #km/s
    Hz = cosmo.H(zl).value #km s^-1 Mpc^-1
    Dd = cosmo.angular_diameter_distance(zl).value
    eps = 1e-8
    area = np.pi*np.power(Dd+eps,2)*np.power(Theta_E(sigma, zl+eps, zs+eps+eps),2)
    return Phi_vel_disp(sigma, zl)*np.power(1+zl,2)*(c_sp/Hz)*area*(rad_to_arcsec**2)

def dTau_dz(zl, zs, Phi_vel_disp = Phi_vel_disp_Mason):
    '''
    Returns the differential multiple image optical depth (tau)
    integrated over the lens velocity dispersion

            Parameters:
                    zl: (float)
                        Redshift of the lens
                    zs: (float)
                        Redshift of the source
                    Phi_vel_disp: (function)
                        Velocity Dispersion Function
            Returns:
                    dTau_dz: (float)
                        Differential multiple image optical depth 
                        over dz
    '''
    #1-500km/s to integrate over
    return integral.quad(dTau_dz_dsigma, 1, 500, args=(zl, zs, Phi_vel_disp))[0]

def Tau(zs, Phi_vel_disp = Phi_vel_disp_Mason):
    '''
    Returns the multiple image optical depth (tau).

            Parameters:
                    zs: (float)
                        Redshift of the source
                    Phi_vel_disp: (function)
                        Velocity Dispersion Function
            Returns:
                    Tau: (float)
                        Multiple image optical depth
    '''
    #0-zs redshift to integrate over
    return integral.quad(dTau_dz, 0, zs, args=(zs, Phi_vel_disp))[0]

def integrand_Lens_cone_volume_diff(z):
    '''
    Returns the differential shell of comoving volume around redshift z.
    This approximation assumes a flat universe.

            Parameters:
                    z: (float)
                        Redshift
            Returns:
                    dVdz: (float)
                        Comoving volume shell
    '''
    c_sp = 299792.458                                  #km/s
    Hz = cosmo.H(z).value                              #km s^-1 Mpc^-1
    return np.power(cosmo.angular_diameter_distance(z).value*(1+z),2)*(c_sp/Hz)

def Lens_cone_volume_diff(z, area_sq_degree, dz=0.5):
    '''
    Returns the comoving volume of a cone underlying an area of A square degrees,
    up to redshift z. This approximation assumes a flat universe.

            Parameters:
                    z: (float)
                        Redshift
            Returns:
                    V: (float)
                        Comoving volume cone
    '''
    area_sterad = area_sq_degree*1/(57.2958**2) #sterad
    if z-dz/2 > 0:
        vol_l = integrand_Lens_cone_volume_diff(z-dz/2)
        vol_r = integrand_Lens_cone_volume_diff(z+dz/2)
        return dz/2 * (vol_l + vol_r) * area_sterad
    vol_l = integrand_Lens_cone_volume_diff(z)
    vol_r = integrand_Lens_cone_volume_diff(z+dz/2)
    return dz/4 * (vol_l + vol_r) * area_sterad

def N_distribution(x, mean, sigma):
    '''
    Returns a normal distribution diven mean and sigma.

            Parameters:
                    x: ndarray(dtype=float, ndim=1)
                        x-cooordinate
                    mean: (float)
                        Mean of the gaussian
                    sigma: (float)
                        Sigma of the gaussian
            Returns:
                    N: ndarray(dtype=float, ndim=1)
                        Gaussian distribution
    '''
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-np.power(x - mean, 2) / (2 * sigma ** 2))

def Exp_cutoff(x, x_cut):
    '''
    Returns an exponential cutoff of an input array past a threshold value.

            Parameters:
                    x: ndarray(dtype=float, ndim=1)
                        Input array
                    x_cut: (float)
                        Threshold
            Returns:
                    EC: ndarray(dtype=float, ndim=1)
                        Exponential cutoff
    '''
    mask = x > x_cut
    r = np.ones(len(x))
    r[mask] = np.exp(-((x[mask] - x_cut) * 10) ** 2)
    return r

def fraction_ell(f):
    '''
    Returns the isophotes axis ratio (f) distribution of the galaxies in SDSS.
    dP/dq from van der Wel et al. 14 (Fig. 1, 1.5<z<2, log M = 10-10.5).
    link: https://ui.adsabs.harvard.edu/abs/2014ApJ...792L...6V/abstract

            Parameters:
                    f: ndarray(dtype=float, ndim=1)
                        Axis ratio array
            Returns:
                    dP/df: ndarray(dtype=float, ndim=1)
                        Distribution of axis ratio
    '''
    T1 = N_distribution(f,0.375,0.11)
    T2 = 2 * N_distribution(f,0.7,0.2) * Exp_cutoff(f,0.82)
    T3 = 0.1 * N_distribution(f,0.5,0.1)
    t = T1+T2+T3
    normt = np.sum(t*(np.ones(len(t))*(f[1]-f[0])))
    return t/normt

def dPdMu_Point_SIS(mu):
    '''
    Returns the magnification probabability distribution of a Singular Isothermal 
    Sphere (SIS) lens and a point source.

            Parameters:
                    mu: ndarray(dtype=float, ndim=1)
                        Magnification array
            Returns:
                    dP/dmu: ndarray(dtype=float, ndim=1)
                        Distribution of magnification
    '''
    if hasattr(mu, '__len__'):
        return np.append(np.zeros(len(mu[mu<2])), 2/np.power(mu[mu>=2]-1,3))
    return 2/np.power(mu-1,3) if mu>=2 else 0

def Lensed_Point_LF(M_int, mu_min, dPdMu, LF_func, zs):
    '''
    Returns the Lensed Luminosity Function.

            Parameters:
                    M_int: ndarray(dtype=float, ndim=1)
                        Absolute magnitude array
                    mu_min: (float)
                        Lower bound of integral over magnification
                    dPdMu: (function)
                        Magnification distribution function
                    LF_func: (function)
                        Luminosity Function function
                    zs: (float)
                        Redshift of the source population
            Returns:
                    LensdLF: ndarray(dtype=float, ndim=1)
                        Lensed Luminosity Function
    '''
    mu, _dPdMu_ = np.logspace(-2, 3., 300), dPdMu(np.logspace(-2, 3., 300))
    mu_index = np.argmin((mu-mu_min)**2)
    displ_LF = LF_func(M_int + 5 / 2 * np.log10(mu[mu_index:]), zs)
    return np.trapz(_dPdMu_[mu_index:] * displ_LF, mu[mu_index:])

def Magnification_Bias(M_int, dPdMu, LF_func, zs):
    '''
    Returns the magnification bias for a given combination of Luminosity Function
    and Magnification probability distribution.
    This assumes a Singular Isothermal Sphere (SIS) model for the lens mass profile.

            Parameters:
                    M_int: ndarray(dtype=float, ndim=1)
                        Absolute magnitude array
                    dPdMu: (function)
                        Magnification distribution function
                    LF_func: (function)
                        Luminosity Function function
                    zs: (float)
                        Redshift of the source population
            Returns:
                    Bias: ndarray(dtype=float, ndim=1)
                        Magnification Bias
    '''
    if hasattr(M_int, '__len__'):
        res = [Lensed_Point_LF(M, 0, dPdMu, LF_func, zs)/LF_func(M, zs) for M in M_int]
    else:
        res = Lensed_Point_LF(M_int, 0, dPdMu, LF_func, zs) / LF_func(M_int, zs)
    return np.array(res)

def Fraction_lensed(M_int, dPdMu, LF_func, Phi_vel_disp, zs):
    '''
    Returns the fraction of lensed galaxies comparing the LF to the magnification bias
    and the multiple image optical depth (Tau).
    This assumes a Singular Isothermal Sphere (SIS) model for the lens mass profile.

            Parameters:
                    M_int: ndarray(dtype=float, ndim=1)
                        Absolute magnitude array
                    dPdMu: (function)
                        Magnification distribution function
                    LF_func: (function)
                        Luminosity Function function
                    Phi_vel_disp: (function)
                        Velocity Dispersion Function function
                    zs: (float)
                        Redshift of the source population
            Returns:
                    F: ndarray(dtype=float, ndim=1)
                        Fraction lensed galaxies
    '''
    tau = Tau(zs, Phi_vel_disp)
    if hasattr(M_int, '__len__'):
        res = [tau * Magnification_Bias(M, dPdMu, LF_func, zs)/\
            (tau * Magnification_Bias(M, dPdMu, LF_func, zs)+(1-tau)) for M in M_int]
    else:
        res = tau * Magnification_Bias(M_int, dPdMu, LF_func, zs)/\
            (tau * Magnification_Bias(M_int, dPdMu, LF_func, zs) + (1 - tau))
    return np.array(res)

def mu_lim_second_image(M, M_lim):
    '''
    Returns the minimum magnification of the brightest image in a Singular Isothermal 
    Sphere (SIS) lens that allows a secondary image of a source of magnitude M to be 
    brigther than a limit magnitude M_lim.

            Parameters:
                    M: (float)
                        Absolute magnitude of the unlensed source
                    M_lim: (float)
                        Limiting magnitude of the survey
            Returns:
                    mu_min: (float)
                        Minumum magnification
    '''
    if M >= M_lim:
        return 1e3
    return -2 / (np.power(10, -0.4 * (M_lim - M)) - 1)

def Fraction_1st_image_arc(mu_arc, M_int, LF_func, zs, dPdMu = dPdMu_Point_SIS):
    '''
    Returns the correction to the Fraction lensed galaxies to considering only 
    galaxies with an arc stretched more than a given value of mu_arc.
    This assumes a Singular Isothermal Sphere (SIS) model for the lens mass profile.

            Parameters:
                    mu_arc: (float)
                        Minimum magnification of brightest arc
                    M_int: ndarray(dtype=float, ndim=1)
                        Absolute magnitude array
                    LF_func: (function)
                        Luminosity Function function
                    zs: (float)
                        Redshift of the source population
                    dPdMu: (function)
                        Magnification distribution function
            Returns:
                    FArc: ndarray(dtype=float, ndim=1)
                        Correction to F accounting for bright arcs
    '''
    if hasattr(M_int, '__len__'):
        res = [Lensed_Point_LF(M, mu_arc, dPdMu, LF_func, zs)/\
            Lensed_Point_LF(M, 0, dPdMu, LF_func, zs) for M in M_int]
    else:
        res = Lensed_Point_LF(M_int, mu_arc, dPdMu, LF_func, zs)/\
            Lensed_Point_LF(M_int, 0, dPdMu, LF_func, zs)
    return np.nan_to_num(np.array(res))

def Fraction_2nd_image_above_Mlim(M_int, M_lim, LF_func, zs, dPdMu = dPdMu_Point_SIS):
    '''
    Returns the correction to the Fraction lensed galaxies to considering only 
    galaxies with a secondary image above the magnitude limit M_lim.
    This assumes a Singular Isothermal Sphere (SIS) model for the lens mass profile.

            Parameters:
                    img_N: (int)
                        Image multiplicity
                    M_int: ndarray(dtype=float, ndim=1)
                        Absolute magnitude array
                    M_lim: (float)
                        Magnitude limit
                    LF_func: (function)
                        Luminosity Function function
                    zs: (float)
                        Redshift of the source population
            Returns:
                    F2nd: ndarray(dtype=float, ndim=1)
                        Correction to F accounting for visible 2nd images
    '''
    if hasattr(M_int, '__len__'):
        res = [Lensed_Point_LF(M, mu_lim_second_image(M, M_lim), dPdMu, LF_func, zs)/\
            Lensed_Point_LF(M, 0, dPdMu, LF_func, zs) for M in M_int]
    else:
        res = Lensed_Point_LF(M_int, mu_lim_second_image(M_int, M_lim), dPdMu, LF_func, zs)/\
            Lensed_Point_LF(M_int, 0, dPdMu, LF_func, zs)
    return np.nan_to_num(np.array(res))

######### SIE LENS FUNCTIONS ###################################################################
def SIE_corr(f):
    '''
    Returns the correction to the area in the source plane producing multiple images,
    accounting for the ellitpicity of the lens.
    This assumes a Singular Isothermal Ellipsoid (SIE) model for the lens mass profile.


            Parameters:
                    f: (float)
                        Axis ratio
            Returns:
                    C: (float)
                        Correction to area of source plane inside outer caustic
    '''
    return (np.pi*f**0.5)/(2*E_ell(1-f**2))

def SIE_avg_corr():
    '''
    Returns the average correction to the area in the source plane producing multiple images,
    accounting for the ellitpicity of the lens and weighted over the ellipticity distribution 
    from dP/dq from van der Wel et al. 14 (Fig. 1, 1.5<z<2, log M = 10-10.5).
    link: https://ui.adsabs.harvard.edu/abs/2014ApJ...792L...6V/abstract
    This assumes a Singular Isothermal Ellipsoid (SIE) model for the lens mass profile.

            Parameters:
                    None
            Returns:
                    C: (float)
                        Average correction to area of source plane inside outer caustic
    '''
    f = np.linspace(0, 1, 100)
    return integral.trapz(SIE_corr(f)*fraction_ell(f), f)

def load_weights_dP_dmu_SIE():
    '''
    Returns the magnification probabability distributions of a point source and Singular 
    Isothermal Ellipsoid (SIE) lens, in each region of the source plane characterised by 
    different image multiplicities.

            Parameters:
                    None
            Returns:
                    dPdmu1 ... w4: (float)
                        Magnification distributions and weights for a SIE lens
    '''
    # BASEPATH = os.path.dirname(os.path.abspath(''))+'/GALESS/galess/'
    BASEPATH = '/Users/giofer/Documents/GitHub/GALESS/galess/'
    if os.path.isfile(BASEPATH+'data/SIE_dPdmu/weights_dP_dmu_SIE.txt'):
        weights = np.loadtxt(BASEPATH+'data/SIE_dPdmu/weights_dP_dmu_SIE.txt')
        __dP_dmu_SIE1, __w1 = np.loadtxt(BASEPATH+'data/SIE_dPdmu/dP_dmu_SIE_1.txt'), weights[0]
        __dP_dmu_SIE2, __w2 = np.loadtxt(BASEPATH+'data/SIE_dPdmu/dP_dmu_SIE_2.txt'), weights[1]
        __dP_dmu_SIE3, __w3 = np.loadtxt(BASEPATH+'data/SIE_dPdmu/dP_dmu_SIE_3.txt'), weights[2]
        __dP_dmu_SIE4, __w4 = np.loadtxt(BASEPATH+'data/SIE_dPdmu/dP_dmu_SIE_4.txt'), weights[3]
        __dP_dmu_SIE1_3 = np.loadtxt(BASEPATH+'data/SIE_dPdmu/dP_dmu_SIE_1_3reg.txt')
        __dP_dmu_SIE3_3 = np.loadtxt(BASEPATH+'data/SIE_dPdmu/dP_dmu_SIE_3_3reg.txt')
        __dP_dmu_SIE1_4= np.loadtxt(BASEPATH+'data/SIE_dPdmu/dP_dmu_SIE_1_4reg.txt')
        return __dP_dmu_SIE1,__dP_dmu_SIE2,__dP_dmu_SIE3,__dP_dmu_SIE4,\
            __dP_dmu_SIE1_3,__dP_dmu_SIE3_3,__dP_dmu_SIE1_4,__w1,__w2,__w3,__w4
    print('Did not load files')
    return 0

PM1, PM2, PM3, PM4, PM13, PM33, PM14, __w1,__w2,__w3,__w4 = load_weights_dP_dmu_SIE()
__dP_dmu_SIE1,__dP_dmu_SIE2,__dP_dmu_SIE3,__dP_dmu_SIE4 = PM1, PM2, PM3, PM4
__dP_dmu_SIE1_3,__dP_dmu_SIE3_3,__dP_dmu_SIE1_4 = PM13, PM33, PM14

def Lensed_Point_LF_SIE(img_N, mu_min, M_int, LF_func, zs):
    '''
    Returns the Lensed Luminosity Function.
    This assumes a Singular Isothermal Ellipsoid (SIE) model for the lens mass profile.
            Parameters:
                    img_N: (int)
                        Image multiplicity
                    mu_min: (float)
                        Lower bound of integral over magnification
                    M_int: ndarray(dtype=float, ndim=1)
                        Absolute magnitude array
                    LF_func: (function)
                        Luminosity Function function
                    zs: (float)
                        Redshift of the source population
            Returns:
                    LensdLF: ndarray(dtype=float, ndim=1)
                        Lensed Luminosity Function
    '''
    dP_dmu_SIE, w = 0, 0
    if   img_N==1:
        dP_dmu_SIE, w = __dP_dmu_SIE1, __w1
    elif img_N==2:
        dP_dmu_SIE, w = __dP_dmu_SIE2, __w2
    elif img_N==3:
        dP_dmu_SIE, w = __dP_dmu_SIE3, __w3
    elif img_N==4:
        dP_dmu_SIE, w = __dP_dmu_SIE4, __w4
    mu = np.logspace(-2,3.,300)
    if hasattr(mu_min, '__len__') is False:
        mu_index = np.argmin((mu-mu_min)**2)
        displ_LF = LF_func(M_int + 5 / 2 * np.log10(mu[mu_index:]), zs)
        return np.trapz(w * dP_dmu_SIE[mu_index:] * displ_LF, mu[mu_index:])
    elif len(mu_min)==2:
        mu_indexL = np.argmin((mu-mu_min[0])**2)
        mu_indexU = np.argmin((mu-mu_min[1])**2)
        displ_LF = LF_func(M_int + 5 / 2 * np.log10(mu[mu_indexL:mu_indexU]), zs)
        return np.trapz(w * dP_dmu_SIE[mu_indexL:mu_indexU] * displ_LF, mu[mu_indexL:mu_indexU])
    else:
        return 0

def Magnification_Bias_SIE(img_N, M_int, LF_func, zs):
    '''
    Returns the magnification bias for a given combination of Luminosity Function
    and Magnification probability distribution.
    This assumes a Singular Isothermal Ellipsoid (SIE) model for the lens mass profile.

            Parameters:
                    img_N: (int)
                        Image multiplicity
                    M_int: ndarray(dtype=float, ndim=1)
                        Absolute magnitude array
                    dPdMu: (function)
                        Magnification distribution function
                    LF_func: (function)
                        Luminosity Function function
                    zs: (float)
                        Redshift of the source population
            Returns:
                    Bias: ndarray(dtype=float, ndim=1)
                        Magnification Bias
    '''
    if hasattr(M_int, '__len__'):
        res = [Lensed_Point_LF_SIE(img_N, 0, M, LF_func, zs) / LF_func(M, zs) for M in M_int]
    else:
        res = Lensed_Point_LF_SIE(img_N, 0, M_int, LF_func, zs) / LF_func(M_int, zs)
    return np.nan_to_num(np.array(res))

def Fraction_lensed_SIE(img_N, M_int, LF_func, Phi_vel_disp, zs):
    '''
    Returns the fraction of lensed galaxies comparing the LF to the magnification bias
    and the multiple image optical depth (Tau).
    This assumes a Singular Isothermal Ellipsoid (SIE) model for the lens mass profile.

            Parameters:
                    img_N: (int)
                        Image multiplicity
                    M_int: ndarray(dtype=float, ndim=1)
                        Absolute magnitude array
                    LF_func: (function)
                        Luminosity Function function
                    Phi_vel_disp: (function)
                        Velocity Dispersion Function function
                    zs: (float)
                        Redshift of the source population
            Returns:
                    F: ndarray(dtype=float, ndim=1)
                        Fraction lensed galaxies
    '''
    #Elliptical caustic area averaged over axis ratio distribution obtained from SIE_avg_corr
    tau = Tau(zs, Phi_vel_disp) * 0.8979070411803386
    if hasattr(M_int, '__len__'):
        res = [tau * Magnification_Bias_SIE(img_N, M, LF_func, zs)/\
            (tau * Magnification_Bias_SIE(img_N, M, LF_func, zs) + (1 - tau)) for M in M_int]
    else:
        res = tau * Magnification_Bias_SIE(img_N, M_int, LF_func, zs)/\
            (tau * Magnification_Bias_SIE(img_N, M_int, LF_func, zs) + (1 - tau))
    return np.nan_to_num(np.array(res))

def get_mu_rel_from_cumulative(P1, P2, mu_array, SMOOTH_FLAG=True):
    '''
    Helper function that returns matched cumulative distributions.

            Parameters:
                    P1: ndarray(dtype=float, ndim=1)
                        Normalised cumulative distribution 1
                    P2: ndarray(dtype=float, ndim=1)
                        Normalised cumulative distribution 2
                    mu_array: ndarray(dtype=float, ndim=1)
                        Magnification array
                    SMOOTH_FLAG: (boolean)
                        Flag for smoothing the function
            Returns:
                    mu_diff: (float)
                        Relation between mu1 and mu2
    '''
    diff = np.zeros(0)
    for id_mu2, mu2 in enumerate(mu_array):
        id_mu1 = np.argmin((P1-P2[id_mu2])**2)
        diff   = np.append(diff, mu_array[id_mu1]-mu2)
    if SMOOTH_FLAG:
        gauss_sigma, id_gf = 5, 10
        diff = np.append(diff[:id_gf], gaussian_filter1d(diff[id_gf:], gauss_sigma))
    return diff

def mu_lim_Nth_image_SIE(img_N, M, M_lim, zs):
    '''
    Returns the minimum magnification of the brightest image in a Singular Isothermal Ellipsoid
    (SIE) lens that allows the n-th image of a source of magnitude M to be brigther than a limit
    magnitude M_lim.
    If mu_1 is greater than this the second image in a SIS is above M_lim.
    log_10(mu_N/mu_1) = 0.4*(M_1-M_lim) solve the equation for mu_1.
    The following function gets the lower bound of the integral on mu_1 by mapping the dP/dmuN
    distribution to dP/dmu1.

            Parameters:
                    img_N: (float)
                        N-th image.
                    M: ndarray(dtype=float, ndim=1)
                        Absolute magnitude array
                    M_lim: (float)
                        Magnitude limit 
                    zs: (float)
                        redshift
            Returns:
                    mu_lim: (float)
                        Minimum magnification to satisfy M_nth < M_lim
    '''
    mu_array = np.logspace(-2,3.,300)
    mu_array_ext_steps = np.zeros(0)
    for i in range(len(mu_array) - 1):
        dist = mu_array[i+1] - mu_array[i]
        mu_array_ext_steps = np.append(mu_array_ext_steps,dist)
    mu_array_ext_steps = np.append(mu_array_ext_steps,0)
    if img_N == 2:
        dP_dmu_A, dP_dmu_B = __dP_dmu_SIE1  , __dP_dmu_SIE2
    elif img_N == 3:
        dP_dmu_A, dP_dmu_B = __dP_dmu_SIE1_3, __dP_dmu_SIE3_3
    elif img_N == 4:
        dP_dmu_A, dP_dmu_B = __dP_dmu_SIE1_4, __dP_dmu_SIE4
    else:
        return 0
    PA = np.cumsum(mu_array_ext_steps*dP_dmu_A)/np.sum(mu_array_ext_steps*dP_dmu_A)
    PB = np.cumsum(mu_array_ext_steps*dP_dmu_B)/np.sum(mu_array_ext_steps*dP_dmu_B)
    diff = get_mu_rel_from_cumulative(PA, PB, mu_array)
    ratiomuBmuA = mu_array/(mu_array+diff)
    ratiomuBmuA[0] = 0
    if img_N==3:
        ratiomuBmuA[mu_array+diff<1.45] = 0
    targetval  = np.power(10,-0.4*(M_lim-M))
    idx = np.argwhere(np.diff(np.sign(ratiomuBmuA-targetval))).flatten()
    mu1_best = np.zeros(0)
    for _idx in idx:
        mu1 = mu_array[_idx]+diff[_idx]
        if mu1<175:
            mu1_best = np.append(mu1_best,mu1)
    if len(mu1_best) == 0:
        return (9.9e2,1e3)
    elif len(mu1_best) == 1:
        return (mu1_best,1e3)
    elif len(mu1_best) == 2:
        return  mu1_best
    else:
        return  mu1_best[-2:]

def Fraction_1st_image_arc_SIE(mu_arc, M_int, LF_func, zs):
    '''
    Returns the correction to the Fraction lensed galaxies to considering only galaxies with an arc
    stretched more than a given value of mu_arc.
    This assumes a Singular Isothermal Ellipsoid (SIE) model for the lens mass profile.

            Parameters:
                    mu_arc: (float)
                        Minimum magnification of brightest arc
                    M_int: ndarray(dtype=float, ndim=1)
                        Absolute magnitude array
                    LF_func: (function)
                        Luminosity Function function
                    zs: (float)
                        Redshift of the source population
            Returns:
                    FArc: ndarray(dtype=float, ndim=1)
                        Correction to F accounting for bright arcs
    '''
    if hasattr(M_int, '__len__'):
        res = [Lensed_Point_LF_SIE(1, mu_arc, M, LF_func, zs)/\
            Lensed_Point_LF_SIE(1, 0, M, LF_func, zs) for M in M_int]
    else:
        res = Lensed_Point_LF_SIE(1, mu_arc, M_int, LF_func, zs)/\
            Lensed_Point_LF_SIE(1, 0, M_int, LF_func, zs)
    return np.nan_to_num(np.array(res))

def Fraction_Nth_image_above_Mlim_SIE(img_N, M_int, M_lim, LF_func, zs):
    '''
    Returns the correction to the Fraction lensed galaxies to considering only galaxies with a n-th
    image above the magnitude limit M_lim.
    This assumes a Singular Isothermal Ellipsoid (SIE) model for the lens mass profile.

            Parameters:
                    img_N: (int)
                        Image multiplicity
                    M_int: ndarray(dtype=float, ndim=1)
                        Absolute magnitude array
                    M_lim: (float)
                        Magnitude limit
                    LF_func: (function)
                        Luminosity Function function
                    zs: (float)
                        Redshift of the source population
            Returns:
                    F2nd: ndarray(dtype=float, ndim=1)
                        Correction to F accounting for visible 2nd images
    '''
    if   img_N==1:
        w = __w1
    elif img_N==2:
        w = __w2
    elif img_N==3:
        w = __w3
    elif img_N==4:
        w = __w4
    else:
        return 0
    if hasattr(M_int, '__len__'):
        res = [Lensed_Point_LF_SIE(1, mu_lim_Nth_image_SIE(img_N, M, M_lim, zs), M, LF_func, zs)/\
            Lensed_Point_LF_SIE(1, 0, M, LF_func, zs) for M in M_int]
    else:
        mu_LIM = mu_lim_Nth_image_SIE(img_N, M_int, M_lim, zs)
        res = Lensed_Point_LF_SIE(1, mu_LIM, M_int, LF_func, zs)/\
            Lensed_Point_LF_SIE(1, 0, M_int, LF_func, zs)
    return np.nan_to_num(np.array(res)) * w

#### Signal-to-Noise Ratio ########################################################################
def magnitude2cps(m, m_zp):
    '''
    Converts an apparent magnitude to counts per second.

            Parameters:
                    m: (float)
                        Magnitude
                    m_zp: (float)
                        Magnitude zero point
            Returns:
                    CpS: (float)
                        Counts per second of astronomical object
    '''
    delta_m = m - m_zp
    return 10 ** (- delta_m / 2.5)

def Signal_to_noise_ratio(app_mag_src, src_size_arcsec,  sky_bckgnd_m_per_arcsec_sq,
                         zero_point_m, exp_time_sec, num_exposures = 1):
    '''
    Returns the Signal-to-Noise ratio (SNR).

            Parameters:
                    app_mag_src: (float)
                        Apparent Magnitude
                    src_size_arcsec: (float)
                        Source size [arcsec]
                    sky_bckgnd_m_per_arcsec_sq: (float)
                        Sky background
                    zero_point_m: (float)
                        Magnitude zero point
                    exp_time_sec: (float)
                        Exposure time [sec]
                    num_exposures: (int)
                        Number of exposures
            Returns:
                    SNR: (float)
                        Signal-to-Noise ratio
    '''
    exposure_time_tot = num_exposures * exp_time_sec
    phc_src = magnitude2cps(app_mag_src, zero_point_m) * exposure_time_tot
    phc_sky = magnitude2cps(sky_bckgnd_m_per_arcsec_sq, zero_point_m) * exposure_time_tot
    sig_sq_bkg = (phc_sky * np.pi * src_size_arcsec ** 2) / (exposure_time_tot ** 2)
    return phc_src / np.sqrt(phc_src + sig_sq_bkg)

############## Calculating K-correction ###########################################################
def get_UV_cont_slope(zs, intrinsic_Mag_UV):
    '''
    Returns the interpolation for the evolution with redshift of the UV Continuum Slope and its 
    derivative with magnitude.
    For 0 < z < 4 Use linear interpolation from Fig.6 of Mondal et al. 2023
    link: https://iopscience.iop.org/article/10.3847/1538-4357/acc110/pdf
    For 4 < z < 6 Table 4 from Bouwens et al. 2014 
    link: https://iopscience.iop.org/article/10.1088/0004-637X/793/2/115/pdf
    For 6 < z < 8 Table 3 from Bouwens et al. 2014 
    link: https://iopscience.iop.org/article/10.1088/0004-637X/793/2/115/pdf
    For 8 < z < 10 from Eq.16 in Liu et al. 2016 (Dragons paper IV)
    link: https://ui.adsabs.harvard.edu/abs/2016MNRAS.462..235L/abstract

            Parameters:
                    zs: (float)
                        Redshift of the source
                    intrinsic_Mag_UV: (float)
                        Intrinsic UV Magnitude of the source
            Returns:
                    beta: (float)
                        UV Continuum Slope
    '''
    B, dBdM, M_beta = -2.16, -0.16, -18.8
    if np.around(zs,0) < 4: 
        B, dBdM, M_beta = -1.5+(zs-1)*((1.5-2.21)/6), 0, -18.8
    elif np.around(zs,0)==4:
        B, dBdM, M_beta = -1.95, -0.13, -18.8
        dBdM = dBdM if intrinsic_Mag_UV <= M_beta else -0.08
    elif np.around(zs,0)==5:
        B, dBdM, M_beta = -2.05, -0.17, -18.8
        dBdM = dBdM if intrinsic_Mag_UV <= M_beta else -0.08
    elif np.around(zs,0)==6:
        B, dBdM, M_beta = -2.22, -0.24, -18.8
        dBdM = dBdM if intrinsic_Mag_UV <= M_beta else -0.08
    elif np.around(zs,0)==7:
        B, dBdM, M_beta = -2.05, -0.20, -19.5
    elif np.around(zs,0)==8:
        B, dBdM, M_beta = -2.13, -0.15, -19.5
    elif np.around(zs,0)==9:
        B, dBdM, M_beta = -2.19, -0.16, -19.5
    elif np.around(zs,0)==10:
        B, dBdM, M_beta = -2.16, -0.16, -19.5
    # if(zs>=4 and zs<=6):
    #     _dBdM    = dBdM if (intrinsic_Mag_UV <= -18.8) else -0.08
    #     UV_slope   = _dBdM*(intrinsic_Mag_UV+18.8)+B
    # elif(zs>=7 and zs<=8): 
    #     UV_slope  = dBdM*(intrinsic_Mag_UV+19.5)+B
    # else: 
    #     UV_slope = -1.7
    return dBdM * (intrinsic_Mag_UV - M_beta) + B    

def Flux_integral_div_C(UV_slope, avg_wave, FWHM_wave):
    '''
    Returns the Integrated Flux over a rectangular window with mean avg_wave and width FWHM_wave.

            Parameters:
                    beta: (float)
                        UV Continuum Slope
                    avg_wave: (float)
                        Window mean
                    FWHM_wave: (float)
                        Window width
            Returns:
                    FluxI: (float)
                        Flux integrated over a rectangular window
    '''
    return ((avg_wave+FWHM_wave/2)**(UV_slope+1)-(avg_wave-FWHM_wave/2)**(UV_slope+1))/(UV_slope+1)

def K_corr_singleMag(zs, observing_band, restframe_band, M):
    '''
    Returns the K correction for a source with magnitude M at redshift z,
    with the intrinsic magnitude calculated in the restframe_band and observed in the 
    observing_band.

            Parameters:
                    zs: (float)
                        Redshift of the source
                    observing_band: (string)
                        Observing band (see supported photo bands)
                    restframe_band: (string)
                        Restframe band (see supported photo bands)
                    M: (float)
                        Intrinsic Magnitude
            Returns:
                    Kcorr: (float)
                        K correction
    '''
    supported_K_correctn_photo_bands = [
        'galex_FUV', 'galex_NUV', 
        'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0', 
        'ukirt_wfcam_Y', 'ukirt_wfcam_J', 'ukirt_wfcam_H', 'ukirt_wfcam_K']
    if all(x in supported_K_correctn_photo_bands for x in [observing_band, restframe_band]):
        rest_frame_wave, rest_frame_FWHM = get_wavelength_nm_from_photo_band(restframe_band)
        obsr_frame_wave, obsr_frame_FWHM = get_wavelength_nm_from_photo_band(observing_band)
        UV_slope = get_UV_cont_slope(zs, M)+1
        rest_frame_flux = Flux_integral_div_C(UV_slope, rest_frame_wave, rest_frame_FWHM)
        obsr_frame_flux = Flux_integral_div_C(UV_slope, obsr_frame_wave, obsr_frame_FWHM)/(1 + zs)
        return -2.5 * np.log10(obsr_frame_flux / rest_frame_flux)
    return 0

def K_correction(zs, observing_band, restframe_band, intrinsic_Mag):
    '''
    Returns the K correction for a population of sources with magnitude M at redshift z,
    with the intrinsic magnitude calculated in the restframe_band and observed in the 
    observing_band.

            Parameters:
                    zs: (float)
                        Redshift of the source
                    observing_band: (string)
                        Observing band (see supported photo bands)
                    restframe_band: (string)
                        Restframe band (see supported photo bands)
                    M: ndarray(dtype=float, ndim=1)
                        Intrinsic Magnitudes
            Returns:
                    Kcorr: (float)
                        K correction
    '''
    if hasattr(intrinsic_Mag, '__len__'):
        res = [K_corr_singleMag(zs, observing_band, restframe_band, M) for M in intrinsic_Mag]
    else:
        res = K_corr_singleMag(zs, observing_band, restframe_band, intrinsic_Mag)
    return np.asarray(res)

def K_correction_from_UV(zs, observing_band, intrinsic_Mag): 
    '''
    Returns the K correction for a population of sources with magnitude M at redshift z,
    with the intrinsic magnitude calculated in the UV and observed in the observing_band.

            Parameters:
                    zs: (float)
                        Redshift of the source
                    observing_band: (string)
                        Window mean
                    M: ndarray(dtype=float, ndim=1)
                        Intrinsic Magnitudes
            Returns:
                    Kcorr: (float)
                        K correction
    '''
    return K_correction(zs, observing_band, 'galex_NUV', intrinsic_Mag)

############## Fundamental Plane ###############################################################
def get_wavelength_nm_from_photo_band(photo_band):
    wav_nm, FWHM = 0, 0
    if  (photo_band == 'galex_FUV'):wav_nm, FWHM = 150, 30 
    elif(photo_band == 'galex_NUV'):wav_nm, FWHM = 230, 50  
    elif(photo_band == 'sdss_g0'):  wav_nm, FWHM = 475, 128  
    elif(photo_band == 'sdss_r0'):  wav_nm, FWHM = 622, 138  
    elif(photo_band == 'sdss_i0'):  wav_nm, FWHM = 763, 149  
    elif(photo_band == 'sdss_z0'):  wav_nm, FWHM = 905, 152
    elif(photo_band == 'ukirt_wfcam_Y'): wav_nm, FWHM = 1031, 120  
    elif(photo_band == 'ukirt_wfcam_J'): wav_nm, FWHM = 1241, 213  
    elif(photo_band == 'ukirt_wfcam_H'): wav_nm, FWHM = 1631, 307  
    elif(photo_band == 'ukirt_wfcam_K'): wav_nm, FWHM = 2201, 390  
    return wav_nm, FWHM

def get_photo_band_from_wavelength_nm(wav_nm):
    supported_K_correctn_photo_bands = [
        'galex_FUV', 'galex_NUV', 
        'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0', 
        'ukirt_wfcam_Y', 'ukirt_wfcam_J', 'ukirt_wfcam_H', 'ukirt_wfcam_K']
    return supported_K_correctn_photo_bands[np.argmin(np.power(np.asarray([150, 230, 475, 622, 763, 905, 1031, 1241, 1631, 2201])-wav_nm,2))]

def get_highest_LYA_rest_fram_observable(photo_band): 
    # from (SDSS+UKIDSS) to match FP calibration https://ui.adsabs.harvard.edu/abs/2010MNRAS.408.1335L/abstract
    return get_wavelength_nm_from_photo_band(photo_band)[0]/ 121.567-1 #nm

def get_obs_frame_band_from_rest_frame_band(z, restframe_band):
    return get_photo_band_from_wavelength_nm(get_wavelength_nm_from_photo_band(restframe_band)[0]*(z+1))

def get_rest_frame_band_from_obs_frame_band(z, obsframe_band):
    return get_photo_band_from_wavelength_nm(get_wavelength_nm_from_photo_band(obsframe_band)[0]/(z+1))
    
def get_obs_frame_band_from_rest_frame_FUV(z):
    return  get_obs_frame_band_from_rest_frame_band(z, 'galex_FUV')

def get_FP_parameters_for_band_and_z_LaBarbera(photo_band, zl): #La Barbera+2010 https://ui.adsabs.harvard.edu/abs/2010MNRAS.408.1335L/abstract
    _alpha_0B, _beta_0B, _gamma_0B = 1.30, -0.82, -0.443 #FP parameters redshift evolution calibrated from the B-band in
    _alpha_2B, _beta_2B, _gamma_2B = 0.46, -0.46, +1.275 #Table2: https://iopscience.iop.org/article/10.3847/1538-4357/abce66/pdf 
    # Assuming the slope in parameter evolution is constant for each rest frame wavelength
    alpha_slope = (_alpha_2B - _alpha_0B)/2
    beta_slope  = (-0.4)*(_beta_2B - _beta_0B)/2
    gamma_0B = _gamma_0B + 10.8*_beta_0B
    gamma_2B = _gamma_2B + _beta_2B*0.4*(27+10*np.log10(1+2)) - np.log10(cosmo.angular_diameter_distance(2).value/206.265)
    gamma_slope = (gamma_2B - gamma_0B)/2
    if  (photo_band == 'sdss_g0'      ): 
        alpha_0, beta_0, gamma_0 = 1.384, 0.315, -9.164 
        alpha_s, beta_s, gamma_s = 0.024, 0.001,  0.079
    elif(photo_band == 'sdss_r0'      ): 
        alpha_0, beta_0, gamma_0 = 1.390, 0.314, -8.867 
        alpha_s, beta_s, gamma_s = 0.018, 0.001,  0.058
    elif(photo_band == 'sdss_i0'      ): 
        alpha_0, beta_0, gamma_0 = 1.426, 0.312, -8.789 
        alpha_s, beta_s, gamma_s = 0.016, 0.001,  0.053
    elif(photo_band == 'sdss_z0'      ): 
        alpha_0, beta_0, gamma_0 = 1.418, 0.317, -8.771 
        alpha_s, beta_s, gamma_s = 0.021, 0.001,  0.072
    elif(photo_band == 'ukirt_wfcam_Y'): 
        alpha_0, beta_0, gamma_0 = 1.467, 0.314, -8.557 
        alpha_s, beta_s, gamma_s = 0.019, 0.001,  0.058
    elif(photo_band == 'ukirt_wfcam_J'): 
        alpha_0, beta_0, gamma_0 = 1.530, 0.318, -8.600
        alpha_s, beta_s, gamma_s = 0.017, 0.001,  0.060
    elif(photo_band == 'ukirt_wfcam_H'): 
        alpha_0, beta_0, gamma_0 = 1.560, 0.318, -8.447 
        alpha_s, beta_s, gamma_s = 0.021, 0.002,  0.077
    elif(photo_band == 'ukirt_wfcam_K'): 
        alpha_0, beta_0, gamma_0 = 1.552, 0.316, -8.270
        alpha_s, beta_s, gamma_s = 0.021, 0.002,  0.076
    elif(photo_band == 'galex_NUV' or photo_band == 'galex_FUV'): 
        alpha_0, beta_0, gamma_0 = 1.384, 0.315, -9.164  #use blu-est filter -> 'sdss_g0'
        alpha_s, beta_s, gamma_s = 0.024, 0.001,  0.079
    else: alpha_0, beta_0, gamma_0, alpha_s, beta_s, gamma_s, zl = 0, 0, 0, 0, 0, 0, 0
    alpha = alpha_0 + zl*alpha_slope
    beta  = beta_0  + zl*beta_slope
    gamma = gamma_0 + zl*gamma_slope
    return alpha, beta, gamma, alpha_s, beta_s, gamma_s

def Source_size_arcsec(M_array_UV, zs):
    # Get Source SB from L-size relation with NO lensing in kpc -> arcsec
    Ls_M0    = -21
    # Parameters evolution from Shibuya et al. (2015) 
    # https://ui.adsabs.harvard.edu/abs/2015ApJS..219...15S/abstract
    Ls_gamma = 0.27
    Ls_R0    = 6.9*((1+zs)**-1.20) 
    return (np.power(10,(M_array_UV-Ls_M0)*(-0.4*Ls_gamma))*Ls_R0)/(cosmo.angular_diameter_distance(zs).value*1e3)*206265 

def get_vel_disp_from_M_star(M_star_10_10_Msun, z): #Cannarozzo Sonnenfeld Nipoti (2020)
    return np.power(10, 2.21 + 0.18*np.log10(M_star_10_10_Msun/1e11) + 0.48*np.log10(1+z))

def get_M_star_from_vel_disp(sigma, z):   #invert Cannarozzo Sonnenfeld Nipoti (2020)
    return np.power(10,(np.log10(sigma) - 0.48*np.log10(1+z) - 2.21)/0.18)*1e11 

def get_log_R_eff_kpc(photo_band, zl, SAMPLE_INTERVAL=False): #gaussian spaced R_eff intervals
    #La Barbera+(2010) Fig. 11 - https://academic.oup.com/mnras/article/408/3/1313/1072129
    if  (photo_band == 'sdss_g0'      ): 
        mean, sigma  = 0.53, 0.41 
    elif(photo_band == 'sdss_r0'      ): 
        mean, sigma  = 0.50, 0.38 
    elif(photo_band == 'sdss_i0'      ): 
        mean, sigma  = 0.51, 0.39 
    elif(photo_band == 'sdss_z0'      ): 
        mean, sigma  = 0.49, 0.43 
    elif(photo_band == 'ukirt_wfcam_Y'): 
        mean, sigma  = 0.41, 0.36 
    elif(photo_band == 'ukirt_wfcam_J'): 
        mean, sigma  = 0.41, 0.34 
    elif(photo_band == 'ukirt_wfcam_H'): 
        mean, sigma  = 0.39, 0.37
    elif(photo_band == 'ukirt_wfcam_K'): 
        mean, sigma  = 0.38, 0.39 
    elif(photo_band == 'galex_NUV' or photo_band == 'galex_FUV'): 
        mean, sigma  = 0.53, 0.41   #use blu-est filter -> 'sdss_g0'
    else:
        mean, sigma  = 0, 0 
    # Parameters evolution from Shibuya et al. (2015) 
    # https://ui.adsabs.harvard.edu/abs/2015ApJS..219...15S/abstract
    mean = mean-1.20*np.log10(1+zl)
    if SAMPLE_INTERVAL:
        distribution = stats.norm(loc=mean, scale=sigma)
        bounds_for_range = distribution.cdf([mean-1.5*sigma, mean+1.5*sigma])
        return distribution.ppf(np.linspace(*bounds_for_range, num=11))
    else: return np.array([mean])

def get_FP_parameters(zl_rest_frame_photo_band, zl, n_sigma = 3, SAMPLE_INTERVAL = False): #gaussian spaced FP params
    alpha, beta, gamma, alpha_s, beta_s, gamma_s = get_FP_parameters_for_band_and_z_LaBarbera(zl_rest_frame_photo_band, zl)
    if SAMPLE_INTERVAL:
        alpha_distribution     = stats.norm(loc=alpha, scale=alpha_s)
        alpha_bounds_for_range = alpha_distribution.cdf([alpha-n_sigma*alpha_s, alpha+n_sigma*alpha_s])
        alpha_array            = alpha_distribution.ppf(np.linspace(*alpha_bounds_for_range, num=11))
        gamma_distribution     = stats.norm(loc=gamma, scale=gamma_s)
        gamma_bounds_for_range = gamma_distribution.cdf([gamma-n_sigma*gamma_s, gamma+n_sigma*gamma_s])
        gamma_array            = gamma_distribution.ppf(np.linspace(*gamma_bounds_for_range, num=11))
        return alpha_array, np.array([beta]), gamma_array #beta is considered fixed
    else: return np.array([alpha]), np.array([beta]), np.array([gamma])

def Check_R_from_sigma_FP(sigma, zl, zs, m_array, M_array_UV, obs_photo_band, n_sigma = 3, SAMPLE_INTERVAL=False): 
    #If LENS_LIGHT_FLAG we sample the Fundamental Plane (given \sigma), and get the weights for the prob of seeing the bright image or 
    #the second through the lens light, averaging over the image position (i.e., [1,2]\theta_E for bright image and [0,1]\theta_E for 2nd img.
    zl_rest_frame_photo_band = get_rest_frame_band_from_obs_frame_band(zl, obs_photo_band)
    alpha, beta, gamma = get_FP_parameters(zl_rest_frame_photo_band, zl, n_sigma = n_sigma, SAMPLE_INTERVAL=SAMPLE_INTERVAL)
    if(alpha.mean() == 0): return (0, 0)  #FIXME: it does not throw a warning!
    arr_logRe = get_log_R_eff_kpc(zl_rest_frame_photo_band, zl, True) 
    cosm_dimm_zl = 10*np.log10(1+zl) #Account for cosmological dimming \propto(1+z)^4
    Reinst = Theta_E(sigma, zl, zs)  #Einstein radius of the lens
    SB_iLF = m_array+2.5*np.log10(np.pi*np.power(Source_size_arcsec(M_array_UV, zs),2)) #mag arcsec^-2
    frac1st, frac2nd = 0, 0
    for Re_logkpc in arr_logRe:
        Re_arc  = np.power(10, Re_logkpc)/(cosmo.angular_diameter_distance(zl).value*1e3)*206265
        FP_frac1st, FP_frac2nd = 0, 0
        for _alpha, _gamma in zip(alpha, gamma[::-1]):
            SBe_avg = (np.log10(Re_arc)-_gamma-_alpha*np.log10(sigma))/beta #SB in [mag x arcsec^-2]
            SBe = SBe_avg + 1.393 #for a deVac profile <SBe> = SBe - 1.393
            img_ps = np.linspace(1,2,10)*Reinst
            _frac1st, _frac2nd = 0, 0
            for R_1st in img_ps:
                SBr_1st = SBe + 8.32678*((R_1st/Re_arc)**0.25 - 1)          # SB of the lens light profile at the position of the first image
                SBr_2nd = SBe + 8.32678*(((R_1st-Reinst)/Re_arc)**0.25 - 1) # SB of the lens light profile at the position of the second image
                _frac1st, _frac2nd = _frac1st + (SB_iLF<(SBr_1st+cosm_dimm_zl)), _frac2nd + (SB_iLF<(SBr_2nd+cosm_dimm_zl))
            FP_frac1st, FP_frac2nd = FP_frac1st + _frac1st/len(img_ps), FP_frac2nd + _frac2nd/len(img_ps)
        frac1st, frac2nd = frac1st + FP_frac1st/len(alpha), frac2nd + FP_frac2nd/len(alpha)
    return frac1st/len(arr_logRe), frac2nd/len(arr_logRe)

########################################################################################################################################################                   
def get_prob_lensed_bckgnd(sigma, zl, zs, M_array, dzs=0.5, LF_func = schechter_LF, SIE_FLAG=False):   
    E_sq_deg = np.pi*(Theta_E(sigma, zl, zs)/(3600))**2 #deg^2
    cVol     = Lens_cone_volume_diff(zs, E_sq_deg, dzs) #Mpc^3
    C = 0.8979070411803386 #Correction for elliptical caustic area in SIE lens averaged over axis ratio distribution
    if(SIE_FLAG): LF = [Lensed_Point_LF_SIE(1, 0, M, LF_func, zs) * C for M in M_array]
    else:         LF = [Lensed_Point_LF(M, 0, dPdMu_Point_SIS, LF_func, zs) for M in M_array]
    return np.nan_to_num(cVol * np.array(LF) * np.abs(np.diff(M_array)[0]))

#########################################################################################################################################################

def calculate_num_lenses_and_prob(sigma_array, zl_array, zs_array, M_array_UV, app_magn_limit, survey_area_sq_degrees,
                                  seeing_arcsec, SNR, exp_time_sec, sky_bckgnd_m_per_arcsec_sq, zero_point_m,  
                                  photo_band, mag_cut = None, arc_mu_threshold = 3, seeing_trsh = 1.5, num_exposures = 1, 
                                  Phi_vel_disp = Phi_vel_disp_Mason, LF_func = schechter_LF, restframe_band = 'galex_NUV', LENS_LIGHT_FLAG = False, SIE_FLAG = True):
    supported_Lens_Light_photo_bands = ['sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0', 'ukirt_wfcam_Y', 'ukirt_wfcam_J', 'ukirt_wfcam_H', 'ukirt_wfcam_K']
    if ((photo_band not in supported_Lens_Light_photo_bands) and (LENS_LIGHT_FLAG==True)): 
        print('Photo band not supported for lens light fitting')
        return 0, 0, 0
    if(mag_cut==None): mag_cut = app_magn_limit
    M_array_UV   = M_array_UV[::-1] if (M_array_UV[0]>M_array_UV[-1]) else M_array_UV
    # Ngal_tensor will store the number of lenses for each (zs, zl, sigma, Mag_UV) combination
    Ngal_tensor  = np.zeros((len(zs_array), len(sigma_array), len(zl_array), len(M_array_UV)))
    # idxM_matrix will store the magnitude at which we should evaluate the cumulative probability
    idxM_matrix  = np.zeros((len(zs_array), len(sigma_array), len(zl_array))).astype('int')
    # N_gal_matrix is prob_matrix \times the number of galaxies in the sampled volume with a given sigma, evaluated at the magnitude described in idxM_matrix 
    Ngal_matrix  = np.zeros((len(zs_array), len(sigma_array), len(zl_array)))
    # The Einstein radius distribution matrix
    Theta_E_mat  = np.zeros((len(zs_array), len(sigma_array), len(zl_array)))
    # Reduce the evaluation to the redshift range for which the rest frame Ly\alpha can be seen by the photometric filter in use
    zs_array = zs_array[zs_array<=get_highest_LYA_rest_fram_observable(photo_band)]
    # Loop over zs, sigma and zl
    for izs, zs in enumerate(tqdm(zs_array)):
        _dzs = zs_array[1]-zs_array[0] if (izs==0) else (zs-zs_array[izs-1])
        if(zs==0): continue #avoid division by 0
        #correcting for distance modulus and K-correction
        obs_band_to_intr_UV_corr = 5 * np.log10(cosmo.luminosity_distance(zs).value * 1e5) + K_correction(zs, photo_band, restframe_band, M_array_UV)
        m_array = M_array_UV + obs_band_to_intr_UV_corr
        M_lim_b = app_magn_limit - 5 * np.log10(cosmo.luminosity_distance(zs).value * 1e5)
        M_lim   = M_lim_b - K_correction(zs, photo_band, restframe_band, M_lim_b)
        idxM_matrix[izs][:][:] = int(np.argmin(np.power(m_array-mag_cut,2)))
        #Calculate the probability (at each mag bin) that the first image arc is stretched at least arc_mu_threshold
        frac_arc     = Fraction_1st_image_arc_SIE(arc_mu_threshold, M_array_UV, LF_func, zs) if SIE_FLAG else Fraction_1st_image_arc(arc_mu_threshold, M_array_UV, LF_func, zs) 
        #Calculate the probability (at each mag bin) that the second image is brighter than M_lim
        frac_2nd_img = Fraction_Nth_image_above_Mlim_SIE(2, M_array_UV, M_lim, LF_func, zs)  if SIE_FLAG else Fraction_2nd_image_above_Mlim(M_array_UV, M_lim, LF_func, zs)
        #TODO: if(SIE_FLAG): frac_3rd_img, frac_4th_img = Fraction_Nth_image_above_Mlim_SIE(3, ...)
        for isg, sigma in enumerate(sigma_array):
            _dsg = sigma_array[1]-sigma_array[0] if (isg==0) else (sigma-sigma_array[isg-1])
            for izl, zl in enumerate(zl_array):
                _dzl = zl_array[1]-zl_array[0] if (izl==0) else (zl-zl_array[izl-1])
                if(zl==0): continue #avoid division by 0
                #The (\Theta_e > c*seeing) condition is a first order approximation that works well in the JWST/EUCLID cases (small seeing).
                #TODO: A complete treatment would involve finding which lensed sources can be seen after the deconvolution of the seeing
                if((zs > zl) and (Theta_E(sigma, zl, zs) > seeing_trsh * seeing_arcsec)):
                    prob_lens      = get_prob_lensed_bckgnd(sigma, zl, zs, M_array_UV, dzs = _dzs, LF_func = LF_func, SIE_FLAG = SIE_FLAG)
                    number_of_ETGs = Lens_cone_volume_diff(zl, survey_area_sq_degrees, dz = _dzl) * (Phi_vel_disp(sigma - _dsg/2, zl) + Phi_vel_disp(sigma + _dsg/2, zl)) * _dsg/2
                    #We approximate the selection of the arc strect OR the second image above M_lim with the max (eval at each mag bin)
                    SNR_1img = Signal_to_noise_ratio(m_array - 2.5 * np.log10(3), Source_size_arcsec(M_array_UV, zs), 
                                                     sky_bckgnd_m_per_arcsec_sq, zero_point_m, exp_time_sec, num_exposures = num_exposures)>=SNR
                    SNR_2img = Signal_to_noise_ratio(m_array, Source_size_arcsec(M_array_UV, zs),  
                                                     sky_bckgnd_m_per_arcsec_sq, zero_point_m, exp_time_sec, num_exposures = num_exposures)>=SNR
                    weight_1img, weight_2img      = Check_R_from_sigma_FP(sigma, zl, zs, m_array, M_array_UV, photo_band) if LENS_LIGHT_FLAG else (1,1)
                    weighted_prob_lens            = prob_lens * np.max(np.vstack((frac_arc * weight_1img * SNR_1img, frac_2nd_img * weight_2img * SNR_2img)), axis=0)
                    Ngal_tensor[izs][isg][izl][:] = weighted_prob_lens * number_of_ETGs
                    Ngal_matrix[izs][isg][izl]    = np.cumsum(Ngal_tensor[izs][isg][izl][:])[idxM_matrix[izs][isg][izl]]
                    Theta_E_mat[izs][isg][izl]    = Theta_E(sigma, zl, zs)
    return Ngal_matrix, Theta_E_mat, Ngal_tensor
    
def get_N_and_P_projections(N_gal_matrix, sigma_array, zl_array, zs_array, SMOOTH=True):
    Ngal_zl_sigma = np.sum(N_gal_matrix, axis=0)
    Ngal_zl_zs    = np.sum(N_gal_matrix, axis=1)
    Ngal_sigma_zs = np.sum(N_gal_matrix, axis=2)
    P_zs          = np.sum(N_gal_matrix, axis=(1,2))/np.sum(N_gal_matrix)*(1/(zs_array[1]-zs_array[0]))
    P_zl          = np.sum(N_gal_matrix, axis=(0,1))/np.sum(N_gal_matrix)*(1/(zl_array[1]-zl_array[0]))
    P_sg          = np.sum(N_gal_matrix, axis=(0,2))/np.sum(N_gal_matrix)*(1/(sigma_array[1]-sigma_array[0]))
    if SMOOTH:### ROLLING AVERAGE ###
        Ngal_zl_sigma = signal.convolve2d(Ngal_zl_sigma, np.ones((3,3))/9, mode='same')
        Ngal_zl_zs    = signal.convolve2d(Ngal_zl_zs   , np.ones((3,3))/9, mode='same')
        Ngal_sigma_zs = signal.convolve2d(Ngal_sigma_zs, np.ones((3,3))/9, mode='same')
        P_zs          = np.convolve(P_zs, np.ones(3)/3, mode='same')
        P_zl          = np.convolve(P_zl, np.ones(3)/3, mode='same')
        P_sg          = np.convolve(P_sg, np.ones(3)/3, mode='same')
    return Ngal_zl_sigma, Ngal_sigma_zs, Ngal_zl_zs, P_zs, P_zl, P_sg

#########################################################################################################################################################
def calculate_num_quad_cusps_lenses(sigma_array, zl_array, zs_array, M_array_UV, app_magn_limit, survey_area_sq_degrees,
                                  seeing_arcsec, SNR, exp_time_sec, sky_bckgnd_m_per_arcsec_sq, zero_point_m,  
                                  photo_band, mag_cut = None, arc_mu_threshold = 3, seeing_trsh = 1.5, num_exposures = 1, 
                                  Phi_vel_disp = Phi_vel_disp_Mason, LF_func = schechter_LF, restframe_band = 'galex_NUV', LENS_LIGHT_FLAG = False, SIE_FLAG = True):
    SIE_FLAG = True
    LENS_LIGHT_FLAG = False 
    supported_Lens_Light_photo_bands = ['sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0', 'ukirt_wfcam_Y', 'ukirt_wfcam_J', 'ukirt_wfcam_H', 'ukirt_wfcam_K']
    if ((photo_band not in supported_Lens_Light_photo_bands) and (LENS_LIGHT_FLAG==True)): 
        print('Photo band not supported for lens light fitting')
        return 0, 0, 0
    if(mag_cut==None): mag_cut = app_magn_limit
    M_array_UV   = M_array_UV[::-1] if (M_array_UV[0]>M_array_UV[-1]) else M_array_UV
    # idxM_matrix will store the magnitude at which we should evaluate the cumulative probability
    idxM_matrix  = np.zeros((len(zs_array), len(sigma_array), len(zl_array))).astype('int')
    # N_gal_matrix is prob_matrix \times the number of galaxies in the sampled volume with a given sigma, evaluated at the magnitude described in idxM_matrix 
    Ngal_matrix_cusp  = np.zeros((len(zs_array), len(sigma_array), len(zl_array)))
    Ngal_matrix_quad  = np.zeros((len(zs_array), len(sigma_array), len(zl_array)))
    # The Einstein radius distribution matrix
    Theta_E_mat  = np.zeros((len(zs_array), len(sigma_array), len(zl_array)))
    # Reduce the evaluation to the redshift range for which the rest frame Ly\alpha can be seen by the photometric filter in use
    zs_array = zs_array[zs_array<=get_highest_LYA_rest_fram_observable(photo_band)]
    # Loop over zs, sigma and zl
    for izs, zs in enumerate(tqdm(zs_array)):
        _dzs = zs_array[1]-zs_array[0] if (izs==0) else (zs-zs_array[izs-1])
        if(zs==0): continue #avoid division by 0
        #correcting for distance modulus and K-correction
        obs_band_to_intr_UV_corr = 5 * np.log10(cosmo.luminosity_distance(zs).value * 1e5) + K_correction(zs, photo_band, restframe_band, M_array_UV)
        m_array = M_array_UV + obs_band_to_intr_UV_corr
        M_lim_b = app_magn_limit - 5 * np.log10(cosmo.luminosity_distance(zs).value * 1e5)
        M_lim   = M_lim_b - K_correction(zs, photo_band, restframe_band, M_lim_b)
        idxM_matrix[izs][:][:] = int(np.argmin(np.power(m_array-mag_cut,2)))
        #Calculate the probability (at each mag bin) that the cusp and quad image is brighter than M_lim
        frac_cusp_img = Fraction_Nth_image_above_Mlim_SIE(3, M_array_UV, M_lim, LF_func, zs) 
        frac_quad_img = Fraction_Nth_image_above_Mlim_SIE(4, M_array_UV, M_lim, LF_func, zs)
        #TODO: if(SIE_FLAG): frac_3rd_img, frac_4th_img = Fraction_Nth_image_above_Mlim_SIE(3, ...)
        for isg, sigma in enumerate(sigma_array):
            _dsg = sigma_array[1]-sigma_array[0] if (isg==0) else (sigma-sigma_array[isg-1])
            for izl, zl in enumerate(zl_array):
                _dzl = zl_array[1]-zl_array[0] if (izl==0) else (zl-zl_array[izl-1])
                if(zl==0): continue #avoid division by 0
                #The (\Theta_e > c*seeing) condition is a first order approximation that works well in the JWST/EUCLID cases (small seeing).
                #TODO: A complete treatment would involve finding which lensed sources can be seen after the deconvolution of the seeing
                if((zs > zl) and (Theta_E(sigma, zl, zs) > seeing_trsh * seeing_arcsec)):
                    prob_lens      = get_prob_lensed_bckgnd(sigma, zl, zs, M_array_UV, dzs = _dzs, LF_func = LF_func, SIE_FLAG = SIE_FLAG)
                    number_of_ETGs = Lens_cone_volume_diff(zl, survey_area_sq_degrees, dz = _dzl) * (Phi_vel_disp(sigma - _dsg/2, zl) + Phi_vel_disp(sigma + _dsg/2, zl)) * _dsg/2
                    weighted_prob_lens_cusp, weighted_prob_lens_quad = prob_lens * frac_cusp_img, prob_lens * frac_quad_img
                    Ngal_matrix_cusp[izs][isg][izl]    = np.cumsum(weighted_prob_lens_cusp)[idxM_matrix[izs][isg][izl]] * number_of_ETGs
                    Ngal_matrix_quad[izs][isg][izl]    = np.cumsum(weighted_prob_lens_quad)[idxM_matrix[izs][isg][izl]] * number_of_ETGs
                    Theta_E_mat[izs][isg][izl]         = Theta_E(sigma, zl, zs)
    return Ngal_matrix_cusp, Ngal_matrix_quad, Theta_E_mat

#########################################################################################################################################################
def get_param_space_idx_from_obs_constraints(CW_ER_zl, CW_ER_zs, E_ring_rad, zs_array, zl_array, sigma_array):
    m_sg = np.zeros((len(zs_array), len(zl_array)))
    for izs, _zs in enumerate(zs_array):
        for izl, _zl in enumerate(zl_array):
            if((_zs>_zl) and (_zs>CW_ER_zs[0]-CW_ER_zs[2]) and (_zs<CW_ER_zs[0]+CW_ER_zs[1])):
                if((_zl>CW_ER_zl[0]-CW_ER_zl[2]) and (_zl<CW_ER_zl[0]+CW_ER_zl[1])):
                    m_sg[izs][izl] = sigma_from_R_Ein(_zs, _zl, E_ring_rad)
    sig_nozero_idx = np.zeros(0).astype(int)
    for sg_from_RE in m_sg[np.where(m_sg > 0)]:
            sig_nozero_idx = np.append(sig_nozero_idx, int(np.argmin(np.abs(sigma_array-sg_from_RE))))
    zs_nozero_idx, zl_nozero_idx = np.where(m_sg > 0)[0], np.where(m_sg > 0)[1]
    return zl_nozero_idx, zs_nozero_idx, sig_nozero_idx

def prob_for_obs_conf_in_param_space_per_sq_degree(survey_title, 
                                                    CW_ER_zl, CW_ER_zs, E_ring_rad,
                                                    zs_array, zl_array, sigma_array, CHECK_LL = True):
    survey_params = utils.read_survey_params(survey_title, VERBOSE = 0)
    area     = survey_params['area']
    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(survey_title)
    Ngal_zl_sigma_LL, Ngal_zs_sigma_LL, Ngal_zs_zl_LL, _ , __ , ___ = get_N_and_P_projections(matrix_LL, sigma_array, zl_array, zs_array, SMOOTH=1)
    mat = matrix_LL if CHECK_LL else matrix_noLL
    res = 0
    zl_nozero_idx, zs_nozero_idx, sig_nozero_idx = get_param_space_idx_from_obs_constraints(CW_ER_zl, CW_ER_zs, E_ring_rad, zs_array, zl_array, sigma_array)
    for src, sig, lns in zip(zs_nozero_idx, sig_nozero_idx, zl_nozero_idx):
        res = res + mat[src][sig][lns]
    return res/area

def get_src_magnitude_distr(m_obs, m_cut, zs_array, prob, M_array_UV, obs_band = 'sdss_i0'):
    m_num = np.zeros(len(m_obs))
    M_array_UV   = M_array_UV[::-1] if (M_array_UV[0]>M_array_UV[-1]) else M_array_UV
    for izs, zs in enumerate(zs_array[zs_array>0]):
        obs_band_to_intr_UV_corr = 5 * np.log10(cosmo.luminosity_distance(zs).value * 1e5) + K_correction_from_UV(zs, obs_band, M_array_UV) 
        m_array_i = M_array_UV + obs_band_to_intr_UV_corr - 2.5 * np.log10(3)
        idcut = int(np.argmin(np.power(m_array_i-m_cut,2)))
        N_per_M = np.sum(prob,axis=(1,2))[izs][:]
        # N_per_M = prob[izs][:][:][:]
        # N_per_M[:][:][idcut-1:] = 0
        # N_per_M = np.sum(N_per_M, axis=(0,1))
        for imu, mu in enumerate(m_array_i):
            m_idx = int(np.argmin(np.abs(m_obs - mu)))
            #if(imu < idcut):
            m_num[m_idx] = m_num[m_idx]+np.sum(N_per_M[imu])
    return m_num

def get_len_magnitude_distr(m_obs, zl_array, sigma_array, matrix, obs_band = 'sdss_i0'):
    m_num = np.zeros(len(m_obs))
    for izl, zl in enumerate(zl_array[zl_array>0]):
        #https://ui.adsabs.harvard.edu/abs/2019ApJ...887...10S/abstract
        M_array_V = -2.5 * (4.86 * np.log10(sigma_array / 200) + 8.52)
        obs_band_to_intr_UV_corr = 5 * np.log10(cosmo.luminosity_distance(zl).value * 1e5) + K_correction(zl, obs_band, 'sdss_g0', M_array_V) 
        m_array_i = M_array_V + obs_band_to_intr_UV_corr
        N_per_sg = np.sum(matrix, axis=0)[:, izl]
        for imu, mu in enumerate(m_array_i):
            m_idx = np.argmin(np.abs(m_obs - mu))
            m_num[m_idx] = m_num[m_idx]+N_per_sg[imu]
    return m_num

#### Prob double lenses ################################################################################################################################
def calculate_num_double_lenses_and_prob(sigma_array, zl_array, zs_array, M_array_UV, app_magn_limit, survey_area_sq_degrees,
                                  seeing_arcsec, SNR, exp_time_sec, sky_bckgnd_m_per_arcsec_sq, zero_point_m,  
                                  photo_band, mag_cut = None, arc_mu_threshold = 3, seeing_trsh = 1.5, num_exposures = 1, 
                                  LENS_LIGHT_FLAG = False, SIE_FLAG = True, FLAG_KCORRECTION = True, DEBUG = False, dbg_izs = 3, dbg_izl = 2):
    supported_Lens_Light_photo_bands = ['sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0', 'ukirt_wfcam_Y', 'ukirt_wfcam_J', 'ukirt_wfcam_H', 'ukirt_wfcam_K']
    if ((photo_band not in supported_Lens_Light_photo_bands) and (LENS_LIGHT_FLAG==True)): 
        print('Photo band not supported for lens light fitting')
        return 0, 0, 0
    if(mag_cut==None): mag_cut = app_magn_limit
    M_array_UV   = M_array_UV[::-1] if (M_array_UV[0]>M_array_UV[-1]) else M_array_UV
    # Ngal_tensor will store the number of lenses for each (zs, zl, sigma, Mag_UV) combination
    Ngal_tensor  = np.zeros((len(zs_array), len(zs_array), len(sigma_array), len(zl_array), len(M_array_UV)))
    # idxM_matrix will store the magnitude at which we should evaluate the cumulative probability
    idxM_z1  = np.zeros(len(zs_array)).astype('int')
    idxM_z2  = np.zeros(len(zs_array)).astype('int')
    # N_gal_matrix is prob_matrix \times the number of galaxies in the sampled volume with a given sigma, evaluated at the magnitude described in idxM_matrix 
    Ngal_matrix  = np.zeros((len(zs_array), len(zs_array), len(sigma_array), len(zl_array)))
    # The Einstein radius distribution matrix
    Theta_E_mat_z1 = np.zeros((len(zs_array), len(sigma_array), len(zl_array)))
    Theta_E_mat_z2 = np.zeros((len(zs_array), len(sigma_array), len(zl_array)))
    #Reduce the evaluation to the redshift range for which the rest frame Ly\alpha can be seen by the photometric filter in use
    zs_array = zs_array[zs_array<=get_highest_LYA_rest_fram_observable(photo_band)]
    ##################################################################################################################
    for izs1, zs1 in enumerate(tqdm(zs_array)):
        _dzs1 = zs_array[1]-zs_array[0] if (izs1==0) else (zs1-zs_array[izs1-1])
        if(zs1==0): continue #avoid division by 0
        #correcting for distance modulus and K-correction
        obs_band_to_intr_UV_corr = 5 * np.log10(cosmo.luminosity_distance(zs1).value * 1e5) + K_correction_from_UV(zs1, photo_band, M_array_UV) 
        m_array = M_array_UV + obs_band_to_intr_UV_corr if FLAG_KCORRECTION else M_array_UV + 5 * np.log10(cosmo.luminosity_distance(zs1).value * 1e5) 
        M_lim_b = app_magn_limit - 5 * np.log10(cosmo.luminosity_distance(zs1).value * 1e5)
        M_lim   = M_lim_b - K_correction_from_UV(zs1, photo_band, M_lim_b) if FLAG_KCORRECTION else M_lim_b
        idxM_z1[izs1] = int(np.argmin(np.power(m_array-mag_cut,2)))
        #Calculate the probability (at each mag bin) that the first image arc is stretched at least arc_mu_threshold
        frac_arc     = Fraction_1st_image_arc_SIE(arc_mu_threshold, M_array_UV, schechter_LF, zs1) if SIE_FLAG else Fraction_1st_image_arc(arc_mu_threshold, M_array_UV, schechter_LF, zs1) 
        #Calculate the probability (at each mag bin) that the second image is brighter than M_lim
        frac_2nd_img = Fraction_Nth_image_above_Mlim_SIE(2, M_array_UV, M_lim, schechter_LF, zs1)  if SIE_FLAG else Fraction_2nd_image_above_Mlim(M_array_UV, M_lim, schechter_LF, zs1)
        for izs2, zs2 in enumerate(zs_array):
            if zs2>zs1:
                _dzs2 = zs_array[1]-zs_array[0] if (izs2==0) else (zs2-zs_array[izs2-1])
                if(zs2==0): continue #avoid division by 0
                #correcting for distance modulus and K-correction
                obs_band_to_intr_UV_corr = 5 * np.log10(cosmo.luminosity_distance(zs2).value * 1e5) + K_correction_from_UV(zs2, photo_band, M_array_UV) 
                m_array = M_array_UV + obs_band_to_intr_UV_corr if FLAG_KCORRECTION else M_array_UV + 5 * np.log10(cosmo.luminosity_distance(zs2).value * 1e5) 
                M_lim_b = app_magn_limit - 5 * np.log10(cosmo.luminosity_distance(zs2).value * 1e5)
                M_lim   = M_lim_b - K_correction_from_UV(zs2, photo_band, M_lim_b) if FLAG_KCORRECTION else M_lim_b
                idxM_z2[izs2] = int(np.argmin(np.power(m_array-mag_cut,2)))
                #Calculate the probability (at each mag bin) that the first image arc is stretched at least arc_mu_threshold
                frac_arc     = Fraction_1st_image_arc_SIE(arc_mu_threshold, M_array_UV, schechter_LF, zs2) if SIE_FLAG else Fraction_1st_image_arc(arc_mu_threshold, M_array_UV, schechter_LF, zs2) 
                #Calculate the probability (at each mag bin) that the second image is brighter than M_lim
                frac_2nd_img = Fraction_Nth_image_above_Mlim_SIE(2, M_array_UV, M_lim, schechter_LF, zs2)  if SIE_FLAG else Fraction_2nd_image_above_Mlim(M_array_UV, M_lim, schechter_LF, zs2)
                #TODO: if(SIE_FLAG): frac_3rd_img, frac_4th_img = Fraction_Nth_image_above_Mlim_SIE(3, M_array, M_lim, schechter_LF, zs), Fraction_Nth_image_above_Mlim_SIE(4, M_array, M_lim, schechter_LF, zs)
                for isg, sigma in enumerate(sigma_array):
                    _dsg = sigma_array[1]-sigma_array[0] if (isg==0) else (sigma-sigma_array[isg-1])
                    for izl, zl in enumerate(zl_array):
                        _dzl = zl_array[1]-zl_array[0] if (izl==0) else (zl-zl_array[izl-1])
                        if(zl==0): continue #avoid division by 0
                        #The (\Theta_e > c*seeing) condition is a first order approximation that works well in the JWST/EUCLID cases (small seeing).
                        #TODO: A complete treatment would involve finding which lensed sources can be seen after the deconvolution of the seeing
                        #The condition zs2>zs1 imposed above means the if the next conditions are satisfied for zs1 then they are for zs2 too.
                        if((zs1>zl) and (Theta_E(sigma, zl, zs1)>seeing_trsh*seeing_arcsec)):
                            prob_lens_z1   = get_prob_lensed_bckgnd(sigma, zl, zs1, M_array_UV, dzs = _dzs1, SIE_FLAG = SIE_FLAG)
                            prob_lens_z2   = get_prob_lensed_bckgnd(sigma, zl, zs2, M_array_UV, dzs = _dzs2, SIE_FLAG = SIE_FLAG)
                            number_of_ETGs = Lens_cone_volume_diff(zl, survey_area_sq_degrees, dz=_dzl)*(Phi_vel_disp_Mason(sigma-_dsg/2, zl)+Phi_vel_disp_Mason(sigma+_dsg/2, zl))*_dsg/2
                            #We approximate the selection of the arc strect OR the second image above M_lim with the max (eval at each mag bin)
                            SNR_1img_z1 =Signal_to_noise_ratio(m_array-2.5*np.log10(3), Source_size_arcsec(M_array_UV, zs1), sky_bckgnd_m_per_arcsec_sq, zero_point_m, exp_time_sec, num_exposures = num_exposures)>=SNR
                            SNR_2img_z1 =Signal_to_noise_ratio(m_array, Source_size_arcsec(M_array_UV, zs1),  sky_bckgnd_m_per_arcsec_sq, zero_point_m, exp_time_sec, num_exposures = num_exposures)>=SNR
                            SNR_1img_z2 =Signal_to_noise_ratio(m_array-2.5*np.log10(3), Source_size_arcsec(M_array_UV, zs2), sky_bckgnd_m_per_arcsec_sq, zero_point_m, exp_time_sec, num_exposures = num_exposures)>=SNR
                            SNR_2img_z2 =Signal_to_noise_ratio(m_array, Source_size_arcsec(M_array_UV, zs2),  sky_bckgnd_m_per_arcsec_sq, zero_point_m, exp_time_sec, num_exposures = num_exposures)>=SNR
                            weight_1img_z1, weight_2img_z1 = Check_R_from_sigma_FP(sigma, zl, zs1, m_array, M_array_UV, photo_band) if LENS_LIGHT_FLAG else (1,1)
                            weight_1img_z2, weight_2img_z2 = Check_R_from_sigma_FP(sigma, zl, zs2, m_array, M_array_UV, photo_band) if LENS_LIGHT_FLAG else (1,1)
                            weighted_prob_lens_z1          = prob_lens_z1*np.max(np.vstack((frac_arc*weight_1img_z1*SNR_1img_z1, frac_2nd_img*weight_2img_z1*SNR_2img_z1)), axis=0)
                            weighted_prob_lens_z2          = prob_lens_z2*np.max(np.vstack((frac_arc*weight_1img_z2*SNR_1img_z2, frac_2nd_img*weight_2img_z2*SNR_2img_z2)), axis=0)
                            integrated_over_M_z1           = np.cumsum(weighted_prob_lens_z1)[idxM_z1[izs1]]
                            integrated_over_M_z2           = np.cumsum(weighted_prob_lens_z2)[idxM_z2[izs2]]
                            Ngal_tensor[izs1][izs2][isg][izl][:] = weighted_prob_lens_z1 * weighted_prob_lens_z2 * number_of_ETGs
                            #TO be a double lens yuou need to see both, so its ok to check the brightest magnitude cut between the two
                            Ngal_matrix[izs1][izs2][isg][izl] = integrated_over_M_z1 * integrated_over_M_z2 * number_of_ETGs
                            Theta_E_mat_z1[izs1][isg][izl] = Theta_E(sigma, zl, zs1)
                            Theta_E_mat_z2[izs2][isg][izl] = Theta_E(sigma, zl, zs2)
    return Ngal_matrix, Theta_E_mat_z1, Theta_E_mat_z2, Ngal_tensor

def get_N_and_P_projections_double_lens(N_gal_matrix, sigma_array, zl_array, zs_array, SMOOTH=True):
    Ngal_zl_sigma  = np.sum(N_gal_matrix, axis=(0,1))
    Ngal_zl_zs1    = np.sum(N_gal_matrix, axis=(1,2))
    Ngal_zl_zs2    = np.sum(N_gal_matrix, axis=(0,2))
    Ngal_sigma_zs1 = np.sum(N_gal_matrix, axis=(1,3))
    Ngal_sigma_zs2 = np.sum(N_gal_matrix, axis=(0,3))
    Ngal_zs1_zs2   = np.sum(N_gal_matrix, axis=(2,3))
    P_zs1          = np.sum(N_gal_matrix, axis=(1,2,3))/np.sum(N_gal_matrix)*(1/(zs_array[1]-zs_array[0]))
    P_zs2          = np.sum(N_gal_matrix, axis=(0,2,3))/np.sum(N_gal_matrix)*(1/(zs_array[1]-zs_array[0]))
    P_sg           = np.sum(N_gal_matrix, axis=(0,1,3))/np.sum(N_gal_matrix)*(1/(sigma_array[1]-sigma_array[0]))
    P_zl           = np.sum(N_gal_matrix, axis=(0,1,2))/np.sum(N_gal_matrix)*(1/(zl_array[1]-zl_array[0]))
    if SMOOTH:### ROLLING AVERAGE ###
        Ngal_zl_sigma  = signal.convolve2d(Ngal_zl_sigma, np.ones((3,3))/9, mode='same')
        Ngal_zl_zs1    = signal.convolve2d(Ngal_zl_zs1,   np.ones((3,3))/9, mode='same')
        Ngal_zl_zs2    = signal.convolve2d(Ngal_zl_zs2,   np.ones((3,3))/9, mode='same')
        Ngal_sigma_zs1 = signal.convolve2d(Ngal_sigma_zs1,np.ones((3,3))/9, mode='same')
        Ngal_sigma_zs2 = signal.convolve2d(Ngal_sigma_zs2,np.ones((3,3))/9, mode='same')
        Ngal_zs1_zs2   = signal.convolve2d(Ngal_zs1_zs2,  np.ones((3,3))/9, mode='same')
        P_zs1          = np.convolve(P_zs1, np.ones(3)/3, mode='same')
        P_zs2          = np.convolve(P_zs2, np.ones(3)/3, mode='same')
        P_zl           = np.convolve(P_zl, np.ones(3)/3, mode='same')
        P_sg           = np.convolve(P_sg, np.ones(3)/3, mode='same')
    return Ngal_zl_sigma, Ngal_zl_sigma, Ngal_zl_zs1, Ngal_zl_zs2, Ngal_sigma_zs1, Ngal_sigma_zs2, Ngal_zs1_zs2, P_zs1, P_zs2, P_zl, P_sg

"""Halo occupation, galaxy power spectra and angular clustering.

Masses are in solar masses, distances in Mpc and angular separations in radians.
The HOD model uses its own cosmology, matching the supplied halo mass tables.
"""

from functools import lru_cache
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy import integrate, special

cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089, Tcmb0=2.725)
OmegaM = cosmo.Om(0)
OmegaL = cosmo.Ode(0)
OmegaK = cosmo.Ok(0)
OmegaB = 0.049
OmegaC = OmegaM - OmegaB
H0 = cosmo.H(0).value
h = H0 / 100
s8 = 0.9
c_light = 299792.458  # km/s
G = 4.3009e-9  # Mpc/Msolar * (km/s)^2
Theta = 2.728/2.7
zeq = 2.5e4 * OmegaM * h**2 * Theta**(-4)


def rho_m(z=0):
    """Return the matter density in solar masses per cubic Mpc."""
    return cosmo.Om(z) * cosmo.critical_density(z).value * 1.477e40


@lru_cache(maxsize=128)
def D_growth_factor(z):
    """Linear growth factor of Eisenstein & Hu, normalized at z=0."""
    def expansion(x):
        return OmegaM*(1+x)**3 + (1-OmegaM-OmegaL)*(1+x)**2 + OmegaL

    def integrand(x):
        return (1+x) / expansion(x)**1.5

    norm = integrate.quad(integrand, 0, zeq)[0]
    growth = integrate.quad(integrand, z, zeq)[0] if z else norm
    return expansion(z)**0.5 * growth / norm


def T(k):
    """Return the Eisenstein & Hu transfer function at wavenumber k."""
    s = 44.5*np.log(9.83/(OmegaM*h**2)) / (1+10*(OmegaB*h**2)**0.75)**0.5
    fc, fb = OmegaC/OmegaM, OmegaB/OmegaM
    alpha = fc / (fc+fb)
    Gamma = OmegaM*h**2 * (alpha**0.5 + (1-alpha**0.5)/(1+(0.43*k*s)**4))
    q = k*Theta**2/Gamma
    beta = 1./(1-0.949*fb)
    L = np.log(np.exp(1) + 1.84*beta*alpha**0.5*q)
    C = 14.4 + 325./(1+60.5*q**1.11)
    return L/(L+C*q**2)


def W_F_transf(k, r):
    """Fourier transform of the spherical top-hat window."""
    kr = np.outer(k, r)
    return (3 / kr**3) * (np.sin(kr) - kr*np.cos(kr))


def dW_dlnR(k, r):
    kr = np.outer(k, r)
    return (9*kr*np.cos(kr) + 3*(kr**2-3)*np.sin(kr)) / kr**3


def power_spectrum(k, z, D_ratio, _PS_NORM_):
    """Linear matter power at z for the supplied growth and normalization."""
    n = 0.96
    N = n-1
    deltah = (1.94e-5 * OmegaM**(-0.785-0.05*np.log(OmegaM))
        * np.exp(-0.95*N-0.169*N**2))
    return (2*(np.pi*deltah*T(k))**2*(c_light*k/H0)**(3+n)/(k**3)
        * D_ratio * _PS_NORM_)


@lru_cache(maxsize=1)
def norm_power_spectrum():
    """Normalize the power spectrum to the variance at 8/h Mpc."""
    def integrand(k):
        window = W_F_transf(k, 8/h).item()
        return power_spectrum(k, 0, 1, 1)/(2*np.pi**2) * np.power(k, 2)*window**2

    return s8**2 / integrate.quad(integrand, 0, 1000)[0]


@lru_cache(maxsize=32)
def _mass_integrand(z, D_ratio, norm):
    # Every halo radius uses the same logarithmic wavenumbers and linear power.
    k = np.logspace(-4, 4, 1000)
    power = np.array([power_spectrum(x, z, D_ratio, norm) for x in k])
    return k, power, np.log(k[1]/k[0])


def _sigma(r, z, D_ratio, _PS_NORM_):
    k, power, dlogk = _mass_integrand(z, D_ratio, _PS_NORM_)
    integrand = power/(2*np.pi**2) * np.power(k, 2) * W_F_transf(r, k)**2
    return np.sqrt(np.sum(k*integrand, axis=-1)*dlogk).reshape(np.shape(r))


def _dlnsdlnm(r, sigma, z, D_ratio, _PS_NORM_):
    k, power, dlogk = _mass_integrand(z, D_ratio, _PS_NORM_)
    integrand = power*np.power(k, 2) * W_F_transf(r, k) * dW_dlnR(r, k)
    I_R = (np.sum(k*integrand, axis=-1)*dlogk).reshape(np.shape(r))
    return 1/(6*np.pi**2) * np.abs(I_R) / sigma**2


def f_TINKER(sigma, z):
    """Tinker mass function multiplicity for overdensity 200."""
    A = 0.186*(1+z)**-0.140
    a = 1.47*(1+z)**-0.060
    b = 2.57*(1+z)**-0.0106756286522959
    return A*(np.power(sigma/b, -a) + 1.0) * np.exp(-1.19/sigma**2)


def nuc(sigma, z):
    return (1.69/D_growth_factor(z)) / sigma


def HMF(M_h, z):
    """Differential halo mass function dn/dM at mass M_h and redshift z."""
    norm = norm_power_spectrum()
    growth = (D_growth_factor(z)/D_growth_factor(0))**2 if z else 1
    density = rho_m(z)
    r = np.power(3.0*M_h/(4.0*np.pi*density), 1/3)
    sigma = _sigma(r, z, growth, norm)
    slope = _dlnsdlnm(r, sigma, z, growth, norm)
    return f_TINKER(sigma, z) * density * slope / M_h**2


def N_cen(M_h, M_min, sigma_logM):
    """Mean central occupation with log-normal scatter in the mass threshold."""
    return 0.5*(1+special.erf((np.log10(M_h)-np.log10(M_min))/(np.sqrt(2)*sigma_logM)))


def N_sat(M_h, M_sat, alpha, M_min, sigma_logM):
    """Mean satellite occupation, including the probability of a central."""
    M_cut = np.power(M_min, -0.5)
    return N_cen(M_h, M_min, sigma_logM) * np.power((M_h-M_cut)/M_sat, alpha)


def N_tot(M_h, M_sat, alpha, M_min, sigma_logM):
    NCEN = N_cen(M_h, M_min, sigma_logM)
    M_cut = np.power(M_min, -0.5)
    return NCEN + NCEN*np.power((M_h-M_cut)/M_sat, alpha)


def n_g(M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array):
    """Comoving galaxy number density in the supplied halo mass distribution."""
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM)
    return np.trapz(HMF_array*NTOT, M_h_array)


def get_c_from_M_h(M_h, z):
    # Duffy et al. (2008), Eq. 4 for relaxed halos.
    return 6.71 * np.power(M_h/(2e12/h), -0.091) * (1+z)**-0.44


def _profile_parameters(M_h, z, crit_dens_rescaled):
    c = get_c_from_M_h(M_h, z)
    return c, np.power(M_h/crit_dens_rescaled, 1/3)/c, np.log(1+c)-c/(1+c)


def _profile(k, c, r_s, f_c):
    x = k*r_s
    si, ci = special.sici(x)
    si_c, ci_c = special.sici(x*(1+c))
    return (np.sin(x)*(si_c-si) + np.cos(x)*(ci_c-ci)
        - np.sin(c*k*r_s)/((1+c)*k*r_s)) / f_c


def u_FT(k, M_h, z, crit_dens_rescaled):
    """Normalized Fourier transform of a truncated NFW halo profile."""
    return _profile(k, *_profile_parameters(M_h, z, crit_dens_rescaled))


def PS_1h_cs(k, M_min, sigma_logM, M_sat, alpha, z, crit_dens_rescaled,
    M_h_array, HMF_array, N_G, NCEN, NSAT):
    U = u_FT(k, M_h_array, z, crit_dens_rescaled)
    return np.trapz(HMF_array*NCEN*NSAT*U, M_h_array) * 2 / np.power(N_G, 2)


def PS_1h_ss(k, M_min, sigma_logM, M_sat, alpha, z, crit_dens_rescaled,
    M_h_array, HMF_array, N_G, NSAT):
    U = u_FT(k, M_h_array, z, crit_dens_rescaled)
    return np.trapz(HMF_array*NSAT*NSAT*U*U, M_h_array) / np.power(N_G, 2)


def _one_halo(U, mass, cs, ss, density_sq):
    return (np.trapz(cs*U, mass) * 2 / density_sq
        + np.trapz(ss*U*U, mass) / density_sq)


def PS_1h(k, M_min, sigma_logM, M_sat, alpha, z, crit_dens_rescaled,
    M_h_array, HMF_array, N_G, NCEN, NSAT):
    U = u_FT(k, M_h_array, z, crit_dens_rescaled)
    return _one_halo(U, M_h_array, HMF_array*NCEN*NSAT, HMF_array*NSAT*NSAT,
        np.power(N_G, 2))


def halo_bias_TINKER(nu, rc=1.686):
    """Tinker et al. halo bias at overdensity 200, using peak height nu."""
    y = np.log10(200)
    A = 1.0 + 0.24*y*np.exp(-(4/y)**4)
    a = 0.44*y - 0.88
    C = 0.019 + 0.107*y + 0.19*np.exp(-(4/y)**4)
    return (1 - A*np.power(nu, a)/(np.power(nu, a) + rc**a)
        + 0.183*np.power(nu, 1.5) + C*np.power(nu, 2.4))


def PS_2h(k, M_min, sigma_logM, M_sat, alpha, z, crit_dens_rescaled,
    M_h_array, HMF_array, N_G, NCEN, NSAT, bias, hmf_k, hmf_PS, D_ratio,
    _PS_NORM_, USE_MY_PS):
    """Two-halo term, or its bias factor for a tabulated matter spectrum."""
    U = u_FT(k, M_h_array, z, crit_dens_rescaled)
    weight = (NCEN+NSAT)*HMF_array*bias
    result = np.power(np.trapz(weight*U, M_h_array)/N_G, 2)
    if USE_MY_PS:
        result = power_spectrum(k, z, D_ratio, _PS_NORM_) * result
    return result


def PS_g(k, M_min, sigma_logM, M_sat, alpha, z, crit_dens_rescaled,
    M_h_array, HMF_array, N_G, bias, hmf_k, hmf_PS, D_ratio, NCEN, NSAT,
    _PS_NORM_, USE_MY_PS):
    NCEN = N_cen(M_h_array, M_min, sigma_logM)
    NSAT = N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    U = u_FT(k, M_h_array, z, crit_dens_rescaled)
    one = _one_halo(U, M_h_array, HMF_array*NCEN*NSAT, HMF_array*NSAT*NSAT,
        np.power(N_G, 2))
    two = np.power(np.trapz((NCEN+NSAT)*HMF_array*bias*U, M_h_array)/N_G, 2)
    if USE_MY_PS:
        two = power_spectrum(k, z, D_ratio, _PS_NORM_) * two
    return one, two


def factor_k(k, theta, comoving_distance_z):
    return k/(2*np.pi) * special.j0(k*theta*comoving_distance_z)


def omega_inner_integral(theta, M_min, sigma_logM, M_sat, alpha, z,
    comoving_distance_z, crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN,
    NSAT, bias, hmf_k, hmf_PS, _PS_NORM_, D_ratio, USE_MY_PS):
    """Project both power terms on a logarithmically spaced wavenumber grid."""
    k = np.logspace(-3, 3, 2000) if USE_MY_PS else hmf_k
    dlogk = np.log(k[1]/k[0])
    profile = _profile_parameters(M_h_array, z, crit_dens_rescaled)
    cs, ss = HMF_array*NCEN*NSAT, HMF_array*NSAT*NSAT
    weight = (NCEN+NSAT)*HMF_array*bias
    density_sq = np.power(N_G, 2)
    # A row per wavenumber keeps the halo mass reduction in its original order.
    P1, P2 = np.empty(len(k)), np.empty(len(k))
    for start in range(0, len(k), 256):
        chunk = slice(start, start+256)
        U = _profile(k[chunk, None], *profile)
        P1[chunk] = _one_halo(U, M_h_array, cs, ss, density_sq)
        P2[chunk] = np.power(np.trapz(weight*U, M_h_array)/N_G, 2)
    if USE_MY_PS:
        hmf_PS = np.array([power_spectrum(x, z, D_ratio, _PS_NORM_) for x in k])
    factor = factor_k(k, theta, comoving_distance_z)
    return np.sum(k*P1*factor)*dlogk, np.sum(k*(hmf_PS*P2)*factor)*dlogk


def F_k_hmf_func_1(k, theta, M_min, sigma_logM, M_sat, alpha, z,
    crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT, comoving_distance_z):
    power = PS_1h(k, M_min, sigma_logM, M_sat, alpha, z, crit_dens_rescaled,
        M_h_array, HMF_array, N_G, NCEN, NSAT)
    return float(power * factor_k(k, theta, comoving_distance_z))


def F_k_hmf_func_2(k, theta, M_min, sigma_logM, M_sat, alpha, z,
    crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN, NSAT, bias, hmf_k,
    hmf_PS, D_ratio, _PS_NORM_, comoving_distance_z, USE_MY_PS):
    power = PS_2h(k, M_min, sigma_logM, M_sat, alpha, z, crit_dens_rescaled,
        M_h_array, HMF_array, N_G, NCEN, NSAT, bias, hmf_k, hmf_PS, D_ratio,
        _PS_NORM_, USE_MY_PS)
    if not USE_MY_PS:
        power *= np.interp(k, hmf_k, hmf_PS, left=0, right=0)
    return float(power * factor_k(k, theta, comoving_distance_z))


def _angular_integral(power, theta, distance, bounds):
    def integrand(k):
        return power(k)*factor_k(k, theta, distance)

    return integrate.quad(integrand, *bounds, limit=10000, epsabs=1e-3)[0]


def _angular_terms(z, crit_dens, mass, hmf, N_G, NCEN, NSAT, bias, k_table,
    P_table, norm, D_ratio, use_linear):
    profile = _profile_parameters(mass, z, crit_dens)
    cs, ss = hmf*NCEN*NSAT, hmf*NSAT*NSAT
    density_sq = np.power(N_G, 2)
    weight = (NCEN+NSAT)*hmf*bias

    # The adaptive integrator revisits nodes across angular separations.
    # Cache the scalar powers only for the lifetime of this redshift slice.
    @lru_cache(maxsize=8192)
    def one(k):
        return _one_halo(_profile(k, *profile), mass, cs, ss, density_sq)

    @lru_cache(maxsize=8192)
    def two(k):
        U = _profile(k, *profile)
        if use_linear:
            matter = power_spectrum(k, z, D_ratio, norm)
        else:
            matter = np.interp(k, k_table, P_table, left=0, right=0)
        return matter * np.power(np.trapz(weight*U, mass)/N_G, 2)

    bounds = (1e-3, 1e5)
    if not use_linear:
        bounds = (max(bounds[0], k_table[0]), min(bounds[1], k_table[-1]))
    return one, two, bounds


def omega_inner_integral_1(theta, M_min, sigma_logM, M_sat, alpha, z,
    comoving_distance_z, crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN,
    NSAT, bias, hmf_k, hmf_PS, _PS_NORM_, D_ratio, USE_MY_PS):
    one, _, _ = _angular_terms(z, crit_dens_rescaled, M_h_array, HMF_array,
        N_G, NCEN, NSAT, bias, hmf_k, hmf_PS, _PS_NORM_, D_ratio, USE_MY_PS)
    return _angular_integral(one, theta, comoving_distance_z, (1e-3, 1e5))


def omega_inner_integral_2(theta, M_min, sigma_logM, M_sat, alpha, z,
    comoving_distance_z, crit_dens_rescaled, M_h_array, HMF_array, N_G, NCEN,
    NSAT, bias, hmf_k, hmf_PS, _PS_NORM_, D_ratio, USE_MY_PS):
    _, two, bounds = _angular_terms(z, crit_dens_rescaled, M_h_array, HMF_array,
        N_G, NCEN, NSAT, bias, hmf_k, hmf_PS, _PS_NORM_, D_ratio, USE_MY_PS)
    return _angular_integral(two, theta, comoving_distance_z, bounds)


def _z_terms(z, M_min, sigma_logM, M_sat, alpha, NCEN, NSAT, norm, use_linear,
    rewrite, need_bias):
    mass, hmf, nu, k, power = init_lookup_table(z, rewrite)
    if NCEN is None:
        NCEN = N_cen(mass, M_min, sigma_logM)
        NSAT = N_sat(mass, M_sat, alpha, M_min, sigma_logM)
    N_G = n_g(M_min, sigma_logM, M_sat, alpha, z, mass, hmf)
    crit_dens = 4/3*np.pi*cosmo.critical_density(z).value*200*2e40
    growth = (D_growth_factor(z)/D_growth_factor(0))**2 if use_linear and z else 1
    # The HMF tables contain squared peak height, as expected by halomod Tinker10.
    bias = halo_bias_TINKER(np.sqrt(nu)) if need_bias else 1
    return _angular_terms(z, crit_dens, mass, hmf, N_G, NCEN, NSAT, bias, k,
        power, norm, growth, use_linear)


def omega_z_component_1(z, theta, M_min, sigma_logM, M_sat, alpha, NCEN, NSAT,
    _PS_NORM_, USE_MY_PS=True, REWRITE_TBLS=False):
    one, _, _ = _z_terms(z, M_min, sigma_logM, M_sat, alpha, NCEN, NSAT,
        _PS_NORM_, USE_MY_PS, REWRITE_TBLS, False)
    return _angular_integral(one, theta, cosmo.comoving_distance(z).value, (1e-3, 1e5))


def omega_z_component_2(z, theta, M_min, sigma_logM, M_sat, alpha, NCEN, NSAT,
    _PS_NORM_, USE_MY_PS=True, REWRITE_TBLS=False):
    _, two, bounds = _z_terms(z, M_min, sigma_logM, M_sat, alpha, NCEN, NSAT,
        _PS_NORM_, USE_MY_PS, REWRITE_TBLS, True)
    return _angular_integral(two, theta, cosmo.comoving_distance(z).value, bounds)


def _omega(theta, M_min, sigma_logM, M_sat, alpha, N_z_nrm, z_array, use_linear,
    rewrite, cut_one_halo=False):
    norm = norm_power_spectrum() if use_linear else None
    factor = np.power(np.asarray(N_z_nrm), 2)/(c_light/cosmo.H(z_array).value)
    components = np.zeros((2, len(theta), len(z_array)))
    for i, z in enumerate(z_array):
        if factor[i] == 0 and not rewrite:
            continue
        distance = cosmo.comoving_distance(z).value
        one, two, bounds = _z_terms(z, M_min, sigma_logM, M_sat, alpha,
            None, None, norm, use_linear, rewrite, True)
        for j, angle in enumerate(theta):
            if not cut_one_halo or angle/(1/206265) <= 50:
                components[0,j,i] = _angular_integral(one, angle, distance, (1e-3, 1e5))
            components[1,j,i] = _angular_integral(two, angle, distance, bounds)
    return np.trapz(components*factor, z_array, axis=-1)


def omega(theta, M_min, sigma_logM, M_sat, alpha, N_z_nrm, z_array,
    USE_MY_PS=True, REWRITE_TBLS=False):
    """One- and two-halo angular correlations at theta (radians).

    N_z_nrm gives the normalized redshift distribution on z_array. USE_MY_PS
    selects analytic linear matter power. Set it to False for the nonlinear
    tabulated spectrum, interpolated linearly over its sampled wavenumbers.
    """
    result = _omega([theta], M_min, sigma_logM, M_sat, alpha, N_z_nrm,
        z_array, USE_MY_PS, REWRITE_TBLS)
    return result[0,0], result[1,0]


def omega_array(theta, M_min, sigma_logM, M_sat, alpha, N_z_nrm, z_array,
    USE_MY_PS=True, REWRITE_TBLS=False):
    """Project all angles together, with the dev model's 50 arcsec one-halo cut."""
    theta = np.asarray(theta)
    one, two = _omega(theta, M_min, sigma_logM, M_sat, alpha, N_z_nrm,
        z_array, USE_MY_PS, REWRITE_TBLS, cut_one_halo=True)
    one[one < 0] = 0
    return one, two


def get_N_dens_avg(z_array, M_min, sigma_logM, M_sat, alpha, z_cen, N_z_nrm):
    """Volume-weighted number density for a supplied redshift distribution."""
    densities = np.empty(len(z_array))
    volumes = np.empty(len(z_array))
    for i, z in enumerate(z_array):
        mass, hmf, _, _, _ = init_lookup_table(z)
        densities[i] = n_g(M_min, sigma_logM, M_sat, alpha, z, mass, hmf)
        volumes[i] = cosmo.comoving_distance(z).value**2 * c_light / cosmo.H(z).value
    return np.trapz(densities*volumes*N_z_nrm, z_array)/np.trapz(volumes*N_z_nrm, z_array)


def get_AVG_N_tot(M_min, sigma_logM, M_sat, alpha, z, LOG_INTGR=False):
    mass, hmf, _, _, _ = init_lookup_table(z)
    NTOT = N_tot(mass, M_sat, alpha, M_min, sigma_logM)
    if LOG_INTGR:
        return np.sum(mass*hmf*NTOT)/np.sum(mass*hmf)
    return np.trapz(hmf*NTOT, mass)/np.trapz(hmf, mass)


def get_AVG_Host_Halo_Mass(M_min, sigma_logM, M_sat, alpha, z, LOG_INTGR=False):
    mass, hmf, _, _, _ = init_lookup_table(z)
    NTOT = N_tot(mass, M_sat, alpha, M_min, sigma_logM)
    if LOG_INTGR:
        return np.sum(mass*mass*hmf*NTOT)/np.sum(mass*hmf*NTOT)
    return np.trapz(mass*hmf*NTOT, mass)/np.trapz(hmf*NTOT, mass)


def get_EFF_gal_bias(M_min, sigma_logM, M_sat, alpha, z, LOG_INTGR=False,
    USE_MY_BIAS=False):
    """Occupation-weighted Tinker bias, using the square root of tabulated nu.

    USE_MY_BIAS=True reproduces the dev calculation using squared peak height.
    """
    mass, hmf, nu, _, _ = init_lookup_table(z)
    bias = halo_bias_TINKER(nu if USE_MY_BIAS else np.sqrt(nu))
    NTOT = N_tot(mass, M_sat, alpha, M_min, sigma_logM)
    if LOG_INTGR:
        return np.sum(bias*mass*hmf*NTOT)/np.sum(mass*hmf*NTOT)
    return np.trapz(bias*hmf*NTOT, mass)/np.trapz(hmf*NTOT, mass)


def get_AVG_f_sat(M_min, sigma_logM, M_sat, alpha, z, LOG_INTGR=False):
    mass, hmf, _, _, _ = init_lookup_table(z)
    NTOT = N_tot(mass, M_sat, alpha, M_min, sigma_logM)
    NSAT = N_sat(mass, M_sat, alpha, M_min, sigma_logM)
    if LOG_INTGR:
        return np.sum(mass*hmf*NSAT)/np.sum(mass*hmf*NTOT)
    return np.trapz(hmf*NSAT, mass)/np.trapz(hmf*NTOT, mass)


@lru_cache(maxsize=32)
def _load_table(z, cache_file):
    if cache_file.is_file():
        with np.load(cache_file) as table:
            return tuple(table[key] for key in ('mass', 'dndm', 'nu', 'k', 'power'))
    tenths = round(z*10)
    if abs(z - tenths/10) < 1e-14 and (tenths == 0 or 12 <= tenths <= 22):
        folder = Path(__file__).resolve().parents[1] / 'data' / 'HMF_tables'
        stem = 'redshift_{}_{}'.format(tenths//10, (tenths%10)*10)
        mass, dndm, nu = np.loadtxt(folder/(stem+'.txt'), delimiter=',')
        k, power = np.loadtxt(folder/(stem+'_PS.txt'), delimiter=',')
        return mass, dndm, nu, k, power


def init_lookup_table(z, REWRITE_TBLS=False):
    """Return (mass, dn/dM, peak height squared, k, nonlinear power) at z.

    Tables are bundled for z=0 and z=1.2 through 2.2 in steps of 0.1. With
    the optional hmf package, other redshifts and forced recalculations are
    cached in GALESS_HMF_CACHE or ~/.cache/galess/HMF_tables.
    """
    folder = Path(os.environ.get('GALESS_HMF_CACHE', '~/.cache/galess/HMF_tables')).expanduser()
    cache_file = folder / (float(z).hex() + '.npz')
    table = None if REWRITE_TBLS else _load_table(z, cache_file)
    if table is None:
        from hmf import MassFunction
        hmf = MassFunction(Mmin=9, Mmax=18, dlog10m=0.1, lnk_min=-11.1,
            lnk_max=9.6, dlnk=0.002, z=z, hmf_model='Behroozi', sigma_8=0.8159,
            cosmo_params={'Om0': OmegaM, 'H0': 100*h})
        table = hmf.m/h, hmf.dndm*h**4, hmf.nu, hmf.k*h, hmf.nonlinear_power/h**3
        # Publish both mass and power arrays together, including concurrent runs.
        folder.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=folder, suffix='.npz', delete=False) as tmp:
            try:
                np.savez(tmp, **dict(zip(('mass', 'dndm', 'nu', 'k', 'power'), table)))
                tmp.flush()
                os.replace(tmp.name, cache_file)
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        _load_table.cache_clear()
    # A caller may edit its arrays without altering the next prediction.
    return tuple(a.copy() for a in table)

import os.path
import numpy as np
from scipy import integrate
from scipy import special
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
from astropy.cosmology import FlatLambdaCDM
cosmo  = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
OmegaM = cosmo.Om(0)
OmegaL = cosmo.Ode(0)
OmegaK = cosmo.Ok(0)
OmegaB = 0.045
OmegaC = OmegaM-OmegaB
H0 = cosmo.H(0).value
h  = H0/100
s8 = 0.9
c  = 299792.458 #speed of light km/s
G  = 4.3009e-9  #Mpc/Msolar*(km/s)^2
Theta=2.728/2.7 #Eisenstein_Hu_98
zeq=2.5*10**4*OmegaM*h**2*Theta**(-4) #Eisenstein_Hu_98

def Integrate_log_space(func, VIRIAL_MASS = False): 
    C = 12 if VIRIAL_MASS else 0
    k = np.logspace(-4+C, 4+C, 1000)
    y = func(k)
    ky = k*y
    dlogk = np.log(k[1]/k[0])
    return np.sum(ky) * dlogk
        
#############################################################################################
### LINEAR POWER SPECTRUM ###################################################################
def rho_m(z = 0):
    return cosmo.Om(z)*cosmo.critical_density(z).value*1.477e40# M_sun/Mpc^3

def D_growth_factor(z):
    integrand= lambda x: (1+x)/(OmegaM*(1+x)**3+(1-OmegaM-OmegaL)*(1+x)**2+OmegaL)**1.5 #Eisenstein & Hu (1999)
    Growth=integrate.quad(integrand,z,zeq)
    norm=integrate.quad(integrand,0,zeq) 
    N=norm[0]
    GrowthFactor=(OmegaM*(1+z)**3+(1-OmegaM-OmegaL)*(1+z)**2+OmegaL)**0.5*Growth[0]/N
    return GrowthFactor

def T(k): #EH99
    s=44.5*np.log(9.83/(OmegaM*h**2))/(1+10*(OmegaB*h**2)**0.75)**0.5 
    fc=OmegaC/OmegaM
    fb=OmegaB/OmegaM
    fcb=fc+fb
    alpha=fc/fcb 
    Gamma=OmegaM*h**2*(alpha**0.5+(1-alpha**0.5)/(1+(0.43*k*s)**4)) 
    q=k*Theta**2/Gamma
    Beta=1./(1-0.949*fb) 
    L=np.log(np.exp(1)+1.84*Beta*alpha**0.5*q) 
    C=14.4+(325./(1+60.5*q**1.11)) 
    Tsup=L/(L+C*q**2) 
    Tmaster=Tsup 
    return Tmaster

def norm_power_spectrum():
    n = 0.96
    N = n-1
    deltah = 1.94*10**(-5)*OmegaM**(-0.785-0.05*np.log(OmegaM))*np.exp(-0.95*N-0.169*N**2)
    intgr_n = lambda x: (2*(np.pi*deltah*T(x))**2*(c*x/H0)**(3+n)/(x**3))/(2*np.pi**2)*np.power(x,2)*W_F_transf(x, 8/h)**2
    norm = integrate.quad(intgr_n, 0, 1000)[0]
    return (s8**2)/norm
    
def power_spectrum(k, z, D_ratio, _PS_NORM_):
    n = 0.96
    N = n-1
    deltah = 1.94*10**(-5)*OmegaM**(-0.785-0.05*np.log(OmegaM))*np.exp(-0.95*N-0.169*N**2)
    return 2*(np.pi*deltah*T(k))**2*(c*k/H0)**(3+n)/(k**3)*D_ratio*_PS_NORM_

#############################################################################################
### HALO MASS FUNCTION ######################################################################

def W_F_transf(k, r):
    kr = np.outer(k, r)
    return (3 / kr**3) * (np.sin(kr) - kr * np.cos(kr))

def dW_dlnR(k, r):
    kr = np.outer(k, r)
    return (9 * kr * np.cos(kr) + 3 * (kr**2 - 3) * np.sin(kr)) / kr**3

def _sigma(r, z, D_ratio, _PS_NORM_):
    intgr = lambda x: power_spectrum(x, z, D_ratio, _PS_NORM_)/(2*np.pi**2)*np.power(x,2)*W_F_transf(x, r)**2
    sigma_sq = integrate.quad(intgr, 0, 1000)[0]
    return np.sqrt(sigma_sq)

def _dlnsdlnm(r, sigma, z, D_ratio, _PS_NORM_):
    intgr = lambda x: power_spectrum(x, z, D_ratio, _PS_NORM_)*np.power(x,2)*W_F_transf(x, r)*dW_dlnR(x, r)
    I_R = integrate.quad(intgr, 0, 1000, limit=2000)[0]
    return 1 / (6*np.pi**2) * np.abs(I_R) / sigma**2

def f_TINKER(sigma, z): #for \Delta = 200
	A0, a0, b0, c0 = 0.186, 1.47, 2.57, 1.19
	alpha = 0.0106756286522959 #10**(-(0.75 / np.log10(200 / 75.0))**1.2)
	A = A0 * (1.0 + z)**-0.140
	a = a0 * (1.0 + z)**-0.060
	b = b0 * (1.0 + z)**-alpha
	return A * (np.power(sigma / b,-a) + 1.0) * np.exp(-c0 / sigma**2)

def HMF(M_h, z):
    _PS_NORM_ = norm_power_spectrum()
    D_ratio   = (D_growth_factor(z)/D_growth_factor(0))**2 if z != 0 else 1
    r = np.power(3.0 * M_h / (4.0 * np.pi * rho_m(z)), 1/3)
    sigma = _sigma(r, z, D_ratio, _PS_NORM_)
    f_sigma = f_TINKER(sigma, z)
    return  f_sigma * rho_m(z) * _dlnsdlnm(r, sigma, z, D_ratio, _PS_NORM_) / M_h**2

##########################################################################################
### HALO OCCUPATION DISTRIBUTION #########################################################

def N_cen(M_h, M_min, sigma_logM):
    return 0.5*(1+special.erf((np.log10(M_h)-np.log10(M_min))/(np.sqrt(2)*sigma_logM)))

def N_sat(M_h, M_sat, alpha, M_min, sigma_logM):
    M_cut = np.power(M_min, -0.5)
    return  N_cen(M_h, M_min, sigma_logM)*np.power((M_h-M_cut)/M_sat,alpha)

def N_tot(M_h, M_sat, alpha, M_min, sigma_logM):
    return N_cen(M_h, M_min, sigma_logM) + N_sat(M_h, M_sat, alpha, M_min, sigma_logM)

def n_g(M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array):
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM)
    return integrate.simps(HMF_array*NTOT, M_h_array)
    #dlogk = np.log(M_h_array[1]/M_h_array[0])
    #return np.sum(M_h_array*HMF_array*NTOT) * dlogk

def get_c_from_M_h(M_h, z): #Eq.4 https://iopscience.iop.org/article/10.1086/367955/pdf
    c_norm = 8
    return (c_norm)/(1+z)*np.power(M_h/(1.4e14), -0.13)

def u_FT(k, M_h, z):
    r_v = np.power(M_h/(4/3*np.pi*cosmo.critical_density(z).value*200*2e40), 1/3) #rho = M_sun/Mpc^3
    c   = get_c_from_M_h(M_h, z)
    f_c = np.log(1+c)-c/(1+c)
    r_s = r_v/c
    si, ci = special.sici(k*r_s)
    si_c, ci_c = special.sici(k*r_s*(1+c))
    return (np.sin(k*r_s)*(si_c-si)+np.cos(k*r_s)*(ci_c-ci)-(np.sin(c*k*r_s)/((1+c)*k*r_s)))/f_c

def PS_1h_cs(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G):
    NCEN = N_cen(M_h_array, M_min, sigma_logM)
    NSAT = N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    U_FT = u_FT(k, M_h_array, z)
    intg = integrate.simps(HMF_array*NCEN*NSAT*U_FT, M_h_array)
    return intg*2/np.power(N_G,2)
    #dlogk = np.log(M_h_array[1]/M_h_array[0])
    #return np.sum(M_h_array*HMF_array*NCEN*NSAT*U_FT) * dlogk *1/np.power(N_G,2)

def PS_1h_ss(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G):
    NSAT = N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
    U_FT = u_FT(k, M_h_array, z)
    intg = integrate.simps(HMF_array*NSAT*NSAT*U_FT*U_FT, M_h_array)
    return intg*1/np.power(N_G,2)
    #dlogk = np.log(M_h_array[1]/M_h_array[0])
    #return np.sum(M_h_array*HMF_array*NSAT*NSAT*U_FT*U_FT) * dlogk *1/np.power(N_G,2)
    
def PS_1h(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G):
    return PS_1h_cs(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G)+PS_1h_ss(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G)

def halo_bias_TINKER(sigma):
    rc = 1.686
    nu = rc/sigma
    A = 1.0 + 0.24 * np.log10(200) * np.exp(-(4/np.log10(200))**4)
    a = 0.44 * np.log10(200) - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107*np.log10(200) + 0.19 * np.exp(-(4/np.log10(200))**4)
    c = 2.4
    return 1 - A * np.power(nu,a) / (np.power(nu,a) + rc**a) + B * np.power(nu,b) + C * np.power(nu,c)

def PS_2h(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G, sigma_array, D_ratio, _PS_NORM_):
    PS_m = power_spectrum(k, z, D_ratio, _PS_NORM_)
    NTOT = N_tot(M_h_array, M_sat, alpha, M_min, sigma_logM)
    U_FT = u_FT(k, M_h_array, z)
    intg = integrate.simps(NTOT*HMF_array*halo_bias_TINKER(sigma_array)*U_FT, M_h_array)
    return PS_m * np.power(intg/N_G,2)   
    #dlogk = np.log(M_h_array[1]/M_h_array[0])
    #return PS_m * np.sum(M_h_array*NTOT*HMF_array*halo_bias_TINKER(sigma_array)*U_FT) * dlogk * np.power(1/N_G,2)   
        
def PS_g(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G, sigma_array, D_ratio, _PS_NORM_):
    PS1 = PS_1h(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G)
    PS2 = PS_2h(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G, sigma_array, D_ratio, _PS_NORM_)
    return PS1 + PS2

def get_norm_z_distr_gal(z_array, M_min, sigma_logM, M_sat, alpha):
    N_Garray = np.zeros(0)
    for z in z_array:
        M_h_array, HMF_array = init_lookup_table(z)
        nng  = n_g(M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array)
        dVdz = cosmo.comoving_distance(z).value**2 * c / cosmo.H(z).value
        N_Garray = np.append(N_Garray, nng * dVdz * np.diff(z_array)[0])
    return N_Garray / np.sum(N_Garray)
        
def get_N_dens_avg(z_array, N_z_NORM, M_min, sigma_logM, M_sat, alpha):
    _N_G, _dVdz = np.zeros(0),  np.zeros(0)
    for z in z_array:
        M_h_array, HMF_array = init_lookup_table(z)
        _N_G  = np.append(_N_G, n_g(M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array))
        _dVdz = np.append(_dVdz, cosmo.comoving_distance(z).value**2 * c / cosmo.H(z).value)
    return integrate.simps(_N_G * _dVdz * N_z_NORM, z_array)/integrate.simps(_dVdz*N_z_NORM, z_array)

def DEBUG_omega_inner_integral(theta, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G, 
                                sigma_array, D_ratio, _PS_NORM_, ONLY_1H_TERM, VERBOSE):
    for k in np.array([0.0001, 0.001, 0.01, 0.1, 1, 10]):
        print(f' ------ ------ k = {k}  ')
        print(f' ------ ------ PS_1h    : {PS_1h(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G)}')
        print(f' ------ ------ ------ PS_cs    : {PS_1h_cs(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G)}')
        print(f' ------ ------ ------ PS_ss    : {PS_1h_ss(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G)}')
        print(f' ------ ------ PS_2h    : {PS_2h(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G, sigma_array, D_ratio, _PS_NORM_)}')
        NCEN = N_cen(M_h_array, M_min, sigma_logM)
        NSAT = N_sat(M_h_array, M_sat, alpha, M_min, sigma_logM)
        U_FT = u_FT(k, M_h_array, z)
        intg = integrate.simps(HMF_array*NCEN*NSAT*U_FT, M_h_array)
        print(f' ------ ------ ------ ------ NCEN    : {np.mean(NCEN)}')
        print(f' ------ ------ ------ ------ NSAT    : {np.mean(NSAT)}')
        print(f' ------ ------ ------ ------ U_FT    : {np.mean(U_FT)}') 
        print(f' ------ ------ ------ ------ intg    : {intg}') 
        print(f' ------ ------ J Bessel : {special.j0(k * theta * cosmo.comoving_distance(z).value)}')
    pass

def omega_inner_integral(theta, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G, 
                                sigma_array, D_ratio, _PS_NORM_, ONLY_1H_TERM, VERBOSE):
    k_array = np.logspace(-4, 4, 1000)
    dlogk = np.log(k_array[1]/k_array[0])
    intgrnd = np.zeros(0)
    for k in k_array:
        if(not ONLY_1H_TERM):
            PS = PS_g(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G, sigma_array, D_ratio, _PS_NORM_) 
        else:
            PS = PS_1h(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G)
        intgrnd = np.append(intgrnd, PS * k / (2*np.pi) * special.j0(k * theta * cosmo.comoving_distance(z).value))
    return np.sum(k_array*intgrnd) * dlogk

def omega(theta, M_min, sigma_logM, M_sat, alpha, z_array, ONLY_1H_TERM = True, VERBOSE = False):
    N_z_NORM = get_norm_z_distr_gal(z_array, M_min, sigma_logM, M_sat, alpha)
    intg = np.zeros(0)
    for iz, z in enumerate(z_array):
        M_h_array, HMF_array = init_lookup_table(z)
        sigma_array, D_ratio, _PS_NORM_ = np.zeros(0), 1, 1
        if(not ONLY_1H_TERM):
            _PS_NORM_ = norm_power_spectrum()
            D_ratio   = (D_growth_factor(z)/D_growth_factor(0))**2 if z != 0 else 1
            for M_h in M_h_array:
                r = np.power(3.0 * M_h / (4.0 * np.pi * rho_m(z)), 1/3)
                sigma_array = np.append(sigma_array, _sigma(r, z, D_ratio, _PS_NORM_))
        N_G  = n_g(M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array)
        if(VERBOSE):
            print(f' -- At z : {z}')
            print(f' ------ N(z)^2 : {np.power(N_z_NORM[iz], 2)}')
            print(f' ------ c/H(z) : {c / cosmo.H(z).value}')
        in_K = omega_inner_integral(theta, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G, 
                                    sigma_array, D_ratio, _PS_NORM_, ONLY_1H_TERM, VERBOSE)
        if(VERBOSE): print(f' ------ innr I : {in_K}')
        intg = np.append(intg, np.power(N_z_NORM[iz], 2) * (c / cosmo.H(z).value) * in_K)
    return integrate.simps(intg, z_array)

def init_lookup_table(z):
    FOLDERPATH = os.path.split(os.path.dirname(os.path.abspath('')))[0]+'/GALESS/galess/data/HMF_tables/'
    if os.path.exists(FOLDERPATH):
        FPATH = FOLDERPATH+'redshift_'+str(int(z))+'_'+str(int(np.around(z%1, 2)*100))+'.txt'
        if os.path.isfile(FPATH):
            M_h_l, hmf_p = np.loadtxt(FPATH, delimiter=',')
        else:
            print(f'Calculating HMF table at redshift {z:.2f}')
            M_h_l = np.logspace(10 , 16, 25)
            hmf_p = np.zeros(0)
            for M_h in M_h_l:
                hmf_p = np.append(hmf_p, HMF(M_h, z))
            np.savetxt(FPATH, (M_h_l, hmf_p),  delimiter=',')
        return M_h_l, hmf_p
    else: 
        print(FOLDERPATH)
        raise ValueError('Folder does not exist.')

if __name__ == "__main__":
    print('################# DEGUGGING CODE #################')
    import matplotlib.pyplot as plt

    M_sat, alpha, M_min, sigma_logM = 10**14.18, 1.0, 10**12.46, 0.2
    z = 0
    log_M = np.linspace(10, 16, 21)
    M_h_array = np.power(10, log_M)
    x_axis  = np.zeros(0)
    y_axis  = np.zeros(0)
    '''
    for M_h in M_h_array:
        ###
        r = np.power(3.0 * M_h / (4.0 * np.pi * rho_m(z)), 1/3)
        s = _sigma(r, z, 1, norm_power_spectrum())
        ###
        #y_axis = np.append(y_axis, np.log10(1/s))
        #x_axis = np.append(x_axis, np.log10(1/s))
        #y_axis = np.append(y_axis, f_TINKER(s, z))
        y_axis = np.append(y_axis, HMF(M_h, z)*M_h)
    plt.plot(np.log10(M_h_array), y_axis)
    '''
    log_k = np.linspace(-4, 4, 21)
    k_arr = np.power(10, log_k)
    _PS_NORM_ = norm_power_spectrum()

    N_z_NORM = get_norm_z_distr_gal(np.array([0,0.5]), M_min, sigma_logM, M_sat, alpha)
    N_G_AVG = get_N_avg(np.array([0,0.5]), N_z_NORM, M_min, sigma_logM, M_sat, alpha)
    log_M_h_array, HMF_array = init_lookup_table(z)
    M_h_array = np.power(10, log_M_h_array)
    for k in k_arr:
        PS = PS_1h(k, M_min, sigma_logM, M_sat, alpha, z, M_h_array, HMF_array, N_G_AVG)
        y_axis = np.append(y_axis, PS)
    plt.plot(k_arr, y_axis)
    plt.xscale('log')
    plt.yscale('log')
    #plt.xlim((-0.65, 0.5))
    #plt.ylim((1e-7,1e0))
    #plt.ylim((-3.4, -0.8))
    plt.show()
    print('#################                #################')
        
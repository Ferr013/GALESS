from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

def integrand_Lens_cone_volume_diff(z):                
    c_sp = 299792.458                                  #km/s
    Hz = cosmo.H(z).value                              #km s^-1 Mpc^-1
    return np.power(cosmo.angular_diameter_distance(z).value*(1+z),2)*(c_sp/Hz)

def Lens_cone_volume_diff(z, area_sq_degree, dz=0.5):  
    area_sterad = area_sq_degree*1/(57.2958**2)        #sterad 
    if(z-dz/2>0):
        return dz/2*(integrand_Lens_cone_volume_diff(z-dz/2)+integrand_Lens_cone_volume_diff(z+dz/2))*area_sterad
    else:
        return dz/4*(integrand_Lens_cone_volume_diff(z)+integrand_Lens_cone_volume_diff(z+dz/2))*area_sterad

area_side_arcs = 15
area_sq_degree = (area_side_arcs/3600)**2

z_gal, z_range = 0.05, np.linspace(0, 0.5, 100)
vol = 0
for z in z_range:
    vol = vol + ls.Lens_cone_volume_diff(0.1, area_sq_degree, dz=np.diff(z_range)[0])

M_array = np.linspace(-21, -15, 100)
dM = np.abs(np.diff(M_array)[0])
dens_background_galaxies, dens_background_galaxies_lensed = 0, 0

for M in M_array:
    K_correction_from_UV = ls.K_correction_from_UV(z_gal, 'ukirt_wfcam_Y', M)
    M_NIR = M - K_correction_from_UV
    LF = ls.schechter_LF(M_NIR, z_gal)
    LF_lensed = ls.Lensed_Point_LF(M_NIR, ls.dPdMu_Point_SIS, ls.schechter_LF, z_gal)
    dens_background_galaxies = dens_background_galaxies + LF
    dens_background_galaxies_lensed = dens_background_galaxies_lensed + LF_lensed

N_background_galaxies = vol*dens_background_galaxies*dM
N_background_galaxies_lensed = vol*dens_background_galaxies_lensed*dM

print(f'Number of background galaxies behind an area of {area_side_arcs}x{area_side_arcs} arcsec : {N_background_galaxies:.4f} ({N_background_galaxies_lensed:.4f} if all gal lensed)')

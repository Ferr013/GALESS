# pylint: disable-msg=C0103,C0302,E0611,W0611,W0613,R0914,R1705,R0913
"""Module providing a set of functions to evaluate the distributions of
rare configuration of lenses and lensed sources in a given survey."""

import os.path
import numpy as np
from scipy import integrate as integral
from scipy import signal, stats
from scipy.special import gamma as gammafunc
from scipy.special import ellipe as E_ell
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM

# import galess.LensStat.lens_stat
# import galess.Utils.ls_utils as utils
import sys
path_root = os.path.split(os.path.abspath(''))[0]
sys.path.append(str(path_root) + '/galess/LensStat/')
sys.path.append(str(path_root) + '/galess/Utils/')
from lens_stat import *
import ls_utils as utils


cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

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
        if zl_array[0] == 0:
            Ngal_zl_sigma[:,0] = 0
            Ngal_zl_zs1[:,0] = 0
            Ngal_zl_zs2[:,0] = 0
        if zs_array[0] == 0:
            Ngal_zl_zs1[0,:] = 0
            Ngal_zl_zs2[0,:] = 0
            Ngal_sigma_zs1[0,:] = 0
            Ngal_sigma_zs2[0,:] = 0
            Ngal_zs1_zs2[0,:] = 0
            Ngal_zs1_zs2[:,0] = 0
        P_zs1 = np.append(P_zs1[0], np.convolve(P_zs1[1:], np.ones(3)/3, mode='same'))
        P_zs2 = np.append(P_zs2[0], np.convolve(P_zs2[1:], np.ones(3)/3, mode='same'))
        P_zl  = np.append(P_zl[0],  np.convolve(P_zl[1:],  np.ones(3)/3, mode='same'))
        P_sg  = np.append(P_sg[0],  np.convolve(P_sg[1:],  np.ones(3)/3, mode='same'))
    return Ngal_zl_sigma, Ngal_zl_sigma, Ngal_zl_zs1, Ngal_zl_zs2, Ngal_sigma_zs1, Ngal_sigma_zs2, Ngal_zs1_zs2, P_zs1, P_zs2, P_zl, P_sg

def get_param_space_idx_from_obs_constraints(CW_ER_zl, CW_ER_zs, E_ring_rad,
                                                zs_array, zl_array, sigma_array):
    m_sg = np.zeros((len(zs_array), len(zl_array)))
    for izs, _zs in enumerate(zs_array):
        for izl, _zl in enumerate(zl_array):
            if _zs>_zl and _zs>CW_ER_zs[0]-CW_ER_zs[2] and _zs<CW_ER_zs[0]+CW_ER_zs[1]:
                if _zl>CW_ER_zl[0]-CW_ER_zl[2] and _zl<CW_ER_zl[0]+CW_ER_zl[1]:
                    m_sg[izs][izl] = sigma_from_R_Ein(_zs, _zl, E_ring_rad)
    sig_nozero_idx = np.zeros(0).astype(int)
    for sg_from_RE in m_sg[np.where(m_sg > 0)]:
            ijk = int(np.argmin(np.abs(sigma_array-sg_from_RE)))
            sig_nozero_idx = np.append(sig_nozero_idx, ijk)
    zs_nozero_idx, zl_nozero_idx = np.where(m_sg > 0)[0], np.where(m_sg > 0)[1]
    return zl_nozero_idx, zs_nozero_idx, sig_nozero_idx

def prob_for_obs_conf_in_param_space_per_sq_degree(survey_title,
                                                    CW_ER_zl, CW_ER_zs, E_ring_rad,
                                                    zs_array, zl_array, sigma_array, CHECK_LL = True):
    survey_params = utils.read_survey_params(survey_title, VERBOSE = 0)
    area = survey_params['area']
    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL =\
         utils.load_pickled_files(survey_title)
    Ngal_zl_sigma_LL, Ngal_zs_sigma_LL, Ngal_zs_zl_LL, _ , __ , ___ =\
         get_N_and_P_projections(matrix_LL, sigma_array, zl_array, zs_array, SMOOTH=1)
    mat = matrix_LL if CHECK_LL else matrix_noLL
    res = 0
    zl_nozero_idx, zs_nozero_idx, sig_nozero_idx =\
         get_param_space_idx_from_obs_constraints(CW_ER_zl, CW_ER_zs, E_ring_rad,
                                                    zs_array, zl_array, sigma_array)
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
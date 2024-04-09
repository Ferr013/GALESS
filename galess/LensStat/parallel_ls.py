import sys
import numpy as np
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import uuid
import galess.LensStat.lens_stat as ls

def init(mem):
    global mem_id
    mem_id = mem
    return

def calculate_num_lenses_and_prob(sigma_array, zl_array, zs_array, M_array_UV, app_magn_limit,
                                    survey_area_sq_degrees, seeing_arcsec, SNR, exp_time_sec,
                                    sky_bckgnd_m_per_arcsec_sq, zero_point_m, photo_band,
                                    mag_cut = None, arc_mu_threshold = 3, seeing_trsh = 1.5,
                                    num_exposures = 1, Phi_vel_disp = ls.Phi_vel_disp_Mason,
                                    LF_func = ls.schechter_LF, restframe_band = 'galex_NUV',
                                    LENS_LIGHT_FLAG = False, SIE_FLAG = True, ncores = None):
    '''
    Returns an two arrays: The first is the number of lensed galaxies per bin of velocity
    dispersion, redshift of the lens and the source, integrated over the source magnitude
    up to the magnitude limit of the survey.
    The second is the value of the Einstein Radius of a SIS lens per bin of velocity
    dispersion, redshift of the lens and of the source.
    The third is the number of lensed galaxies per bin of velocity dispersion, redshift
    of the lens and the source, and abs magnitude of the source.

            Parameters:
                    sigma_array: ndarray(dtype=float, ndim=1)
                        Velocity dispersion array of the lens
                    zl_array: ndarray(dtype=float, ndim=1)
                        Redshift array of the lens
                    zs_array: ndarray(dtype=float, ndim=1)
                        Redshift array of the source
                    M_array_UV: ndarray(dtype=float, ndim=1)
                        Intrisic UV abs magnitude array of the source
                    app_magn_limit: (float)
                        Magnitude limit of the survey
                    survey_area_sq_degrees: (float)
                        Area in sq. degrees of the survey
                    seeing_arcsec: (float)
                        Seeing of the telescope in arcseconds
                    SNR: (float)
                        Minimum signal-to-noise ration required for lens identification
                    exp_time_sec: (float)
                        Exposure time of the survey tiles
                    sky_bckgnd_m_per_arcsec_sq: (float)
                        Sky backfround in mag/arcsec^2
                    zero_point_m: (float)
                        Zero point of the survey in photo_band
                    photo_band: (string)
                        Observing photmetric band
                    mag_cut: (float)
                        Cut in magnitude on the survey data
                    arc_mu_threshold: (float)
                        Minimum magnification of brightest image (i.e., arc stretch factor)
                    seeing_trsh: (float)
                        Factor that fixes the minimum ratio of Einstein radius to seeing
                        required for lens identification
                    num_exposures: (int)
                        Number of exposures
                    Phi_vel_disp: (function)
                        Velocity Dispersion Function function
                    LF_func: (function)
                        Luminosity Function function
                    restframe_band: (string)
                        Restframe band associated to the source magnitude M_array_UV
                    LENS_LIGHT_FLAG: (boolean)
                        Flag to use account for the non-subtraction of the lens light profile
                    SIE_FLAG: (boolean)
                        Flag to use Singular Isothermal Ellipsoid (SIE) for the lens mass profile
            Returns:
                    Ngal_matrix: ndarray(dtype=float, ndim=3)
                        Number of lensed galaxies per bin of velocity dispersion, redshift
                        of the lens and the source, integrated over the source magnitude
                        up to the magnitude limit of the survey.
                    Theta_E_mat: ndarray(dtype=float, ndim=3)
                        Value of the Einstein Radius of a SIS lens per bin of velocity
                        dispersion, redshift of the lens and of the source.
                    Ngal_tensor: ndarray(dtype=float, ndim=4)
                        Number of lensed galaxies per bin of velocity dispersion, redshift
                        of the lens and the source, and abs magnitude of the source.
    '''
    supported_Lens_Light_photo_bands = [
        'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0',
        'ukirt_wfcam_Y', 'ukirt_wfcam_J', 'ukirt_wfcam_H', 'ukirt_wfcam_K']
    if photo_band not in supported_Lens_Light_photo_bands and LENS_LIGHT_FLAG is True:
        print('Photo band not supported for lens light fitting')
        return 0, 0, 0
    if mag_cut is None:
        mag_cut = app_magn_limit
    M_array_UV   = M_array_UV[::-1] if (M_array_UV[0]>M_array_UV[-1]) else M_array_UV
    # Reduce the evaluation to the redshift range for which the rest frame Ly\alpha can be seen by
    # the photometric filter in use
    zs_array = zs_array[zs_array<=ls.get_highest_LYA_rest_fram_observable(photo_band)]
    _dzs = zs_array[1]-zs_array[0]
    # Loop over zs, sigma and zl
    if cores is None:
        #TODO: Use 2/3 of cores and loop over the remainder
        cores = len(zs_array)
        #cores = multiprocessing.cpu_count()
    _args_ = sigma_array, zl_array, M_array_UV, app_magn_limit, photo_band, mag_cut, LENS_LIGHT_FLAG,\
             survey_area_sq_degrees, seeing_arcsec, SNR, exp_time_sec, sky_bckgnd_m_per_arcsec_sq,\
             zero_point_m, arc_mu_threshold, seeing_trsh, num_exposures, Phi_vel_disp, LF_func,\
             restframe_band, SIE_FLAG, _dzs
    shape = (len(zs_array), len(sigma_array), len(zl_array))
    args =  [(i, shape, zs_array[i], _args_) for i in range(int(cores))]
    exit = False
    try:
        global mem_id
        mem_id = str(uuid.uuid1())[:30] #Avoid OSError: [Errno 63]
        ###### Make Share Memory for Ngal ######
        nbytes = (len(zs_array) * len(sigma_array) * len(zl_array)) * np.float64(1).nbytes
        shd_mem = SharedMemory(name=f'{mem_id}', create=True, size=nbytes)
        method = 'spawn'
        if sys.platform.startswith('linux'):
            method = 'fork'
        ctx = multiprocessing.get_context(method)
        pool = ctx.Pool(processes=cores, maxtasksperchild=1,
                        initializer=init, initargs=(mem_id,))
        try:
            pool.map_async(cal_num_lens_prob_in_single, args, chunksize=1).get(timeout=10_000)
        except KeyboardInterrupt:
            print("Caught kbd interrupt")
            pool.close()
            exit = True
        else:
            pool.close()
            pool.join()
            Ngal_matrix = np.ndarray(shape, buffer=shd_mem.buf, dtype=np.float64).copy()
    finally:
        shd_mem.close()
        shd_mem.unlink()
        if exit:
            sys.exit(1)
    return Ngal_matrix

def cal_num_lens_prob_in_single(args):
    job_id, shape, zs, _args_ = args
    sigma_array, zl_array, M_array_UV, app_magn_limit, photo_band, mag_cut, LENS_LIGHT_FLAG,\
    survey_area_sq_deg, seeing_arcsec, SNR, exp_time_sec, sky_bckgnd_m_per_arcsec_sq,\
    zero_point_m, arc_mu_threshold, seeing_trsh, num_exposures, Phi_vel_disp, LF_func,\
    restframe_band, SIE_FLAG, _dzs = _args_
    shmem = SharedMemory(name=f'{mem_id}', create=False)
    shres = np.ndarray(shape, buffer=shmem.buf, dtype=np.float64)
    shres[job_id, :, :] = inner_integral(zs, _dzs, sigma_array, zl_array, M_array_UV,
                                         app_magn_limit, photo_band, mag_cut, LENS_LIGHT_FLAG,
                                         survey_area_sq_deg, seeing_arcsec, SNR, exp_time_sec,
                                         sky_bckgnd_m_per_arcsec_sq, zero_point_m, arc_mu_threshold,
                                         seeing_trsh, num_exposures, Phi_vel_disp, LF_func,
                                         restframe_band, SIE_FLAG)
    return

def inner_integral(zs, _dzs, sigma_array, zl_array, M_array_UV, app_magn_limit,
                photo_band, mag_cut, LENS_LIGHT_FLAG, survey_area_sq_degrees, seeing_arcsec,
                SNR, exp_time_sec, sky_bckgnd_m_per_arcsec_sq, zero_point_m,
                arc_mu_threshold = 3, seeing_trsh = 1.5,
                num_exposures = 1, Phi_vel_disp = ls.Phi_vel_disp_Mason,
                LF_func = ls.schechter_LF, restframe_band = 'galex_NUV', SIE_FLAG = True):
    # Ngal_tensor will store the number of lenses for each (zs, zl, sigma, Mag_UV) combination
    Ngal_tensor  = np.zeros((len(sigma_array), len(zl_array), len(M_array_UV)))
    # idxM_matrix will store the magnitude at which we should evaluate the cumulative probability
    idxM_matrix  = np.zeros((len(sigma_array), len(zl_array))).astype('int')
    # N_gal_matrix is prob_matrix \times the number of galaxies in the sampled volume with a given
    # sigma, evaluated at the magnitude described in idxM_matrix
    Ngal_matrix_per_zs  = np.zeros((len(sigma_array), len(zl_array)))
    if zs!=0: #avoid division by 0
        #correcting for distance modulus and K-correction
        obs_band_to_intr_UV_corr = 5 * np.log10(ls.cosmo_luminosity_distance(zs) * 1e5)+\
                ls.K_correction(zs, photo_band, restframe_band, M_array_UV)
        m_array = M_array_UV + obs_band_to_intr_UV_corr
        M_lim_b = app_magn_limit - 5 * np.log10(ls.cosmo_luminosity_distance(zs) * 1e5)
        M_lim   = M_lim_b - ls.K_correction(zs, photo_band, restframe_band, M_lim_b)
        idxM_matrix[:][:] = int(np.argmin(np.power(m_array-mag_cut,2)))
        # Calculate the probability (at each mag bin) that the first image arc is stretched at
        # least arc_mu_threshold and that the second image is brighter than M_lim
        if SIE_FLAG:
            frac_arc = ls.Fraction_1st_image_arc_SIE(arc_mu_threshold, M_array_UV, LF_func, zs)
            frac_2nd_img = ls.Fraction_Nth_image_above_Mlim_SIE(2, M_array_UV, M_lim, LF_func, zs)
        else:
            frac_arc = ls.Fraction_1st_image_arc(arc_mu_threshold, M_array_UV, LF_func, zs)
            frac_2nd_img = ls.Fraction_2nd_image_above_Mlim(M_array_UV, M_lim, LF_func, zs)
        #TODO: if(SIE_FLAG): frac_3rd_img, frac_4th_img = Fraction_Nth_image_above_Mlim_SIE(3, ...)
        for isg, sigma in enumerate(sigma_array):
            _dsg = sigma_array[1]-sigma_array[0] if isg==0 else sigma-sigma_array[isg-1]
            for izl, zl in enumerate(zl_array):
                _dzl = zl_array[1]-zl_array[0] if izl == 0 else zl-zl_array[izl-1]
                if zl==0:
                    continue #avoid division by 0
                # The (\Theta_e > c*seeing) condition is a first order approximation that works
                # well in the JWST/EUCLID cases (small seeing).
                #TODO: A complete treatment would involve finding which lensed sources can be seen
                #TODO: after the deconvolution of the seeing
                if zs > zl and ls.Theta_E(sigma, zl, zs) > seeing_trsh * seeing_arcsec:
                    # We approximate the selection of the arc strect OR the second image above
                    # M_lim with the max (eval at each mag bin)
                    SNR_1img = ls.Signal_to_noise_ratio(m_array - 2.5 * np.log10(3),
                                                    ls.Source_size_arcsec(M_array_UV, zs),
                                                    sky_bckgnd_m_per_arcsec_sq,
                                                    zero_point_m, exp_time_sec,
                                                    num_exposures = num_exposures) >= SNR
                    SNR_2img = ls.Signal_to_noise_ratio(m_array,
                                                    ls.Source_size_arcsec(M_array_UV, zs),
                                                    sky_bckgnd_m_per_arcsec_sq,
                                                    zero_point_m, exp_time_sec,
                                                    num_exposures = num_exposures) >= SNR
                    if LENS_LIGHT_FLAG:
                        weight_1img, weight_2img = ls.Check_R_from_sigma_FP(
                                                    sigma, zl, zs, m_array, M_array_UV, photo_band)
                    else:
                        weight_1img, weight_2img = (1,1)
                    weight_left = frac_arc * weight_1img * SNR_1img
                    weight_right = frac_2nd_img * weight_2img * SNR_2img
                    weight_final = np.vstack((weight_left, weight_right))
                    prob_lens = ls.get_prob_lensed_bckgnd(sigma, zl, zs, M_array_UV, dzs = _dzs,
                                                        LF_func = LF_func, SIE_FLAG = SIE_FLAG)
                    weighted_prob_lens = prob_lens * np.max(weight_final, axis=0)
                    Cone = ls.Lens_cone_volume_diff(zl, survey_area_sq_degrees, dz = _dzl)
                    IVD = (Phi_vel_disp(sigma-_dsg/2, zl)+Phi_vel_disp(sigma+_dsg/2, zl))*_dsg/2
                    number_of_ETGs = Cone * IVD
                    Ngal_tensor[isg][izl][:] = weighted_prob_lens * number_of_ETGs
                    ijk = idxM_matrix[isg][izl]
                    Ngal_matrix_per_zs[isg][izl] = np.cumsum(Ngal_tensor[isg][izl][:])[ijk]
    return Ngal_matrix_per_zs
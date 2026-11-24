"""Integrate independent source redshifts on a pool of worker processes."""
from functools import partial
import multiprocessing
import sys

import numpy as np
import galess.LensStat.lens_stat as ls


def _source_counts(plane, sigma_array, zl_array, lens_counts, settings):
    zs, dzs = plane
    matrix = np.zeros(lens_counts.shape)
    if zs != 0:
        for izl, isg, theta, prob, mag_index in ls._survey_source_plane(
            sigma_array, zl_array, zs, dzs, *settings
        ):
            weighted = ls._counts_per_magnitude(prob, lens_counts[isg, izl])
            matrix[isg, izl] = np.cumsum(weighted[:, :mag_index+1], axis=1)[:, -1]
    return matrix


def _foreground_counts(sigma_array, zl_array, area, Phi_vel_disp, VDF_args, max_zs, min_theta):
    if VDF_args is None:
        return ls._lens_counts(sigma_array, zl_array, area, Phi_vel_disp, max_zs, min_theta)
    return ls._lens_counts(sigma_array, zl_array, area,
        lambda sigma, zl: Phi_vel_disp(sigma, zl, *VDF_args), max_zs, min_theta)


def inner_integral(zs, _dzs, sigma_array, zl_array, M_array_UV, app_magn_limit,
    photo_band, mag_cut, LENS_LIGHT_FLAG, survey_area_sq_degrees, seeing_arcsec,
    SNR, exp_time_sec, sky_bckgnd_m_per_arcsec_sq, zero_point_m,
    arc_mu_threshold=3, seeing_trsh=1.5, num_exposures=1, Phi_vel_disp=ls.Phi_vel_disp_Mason,
    VDF_args=None, LF_func=ls.schechter_LF, restframe_band='galex_NUV', SIE_FLAG=True
):
    """Integrate one source redshift and return counts by velocity dispersion and lens redshift."""
    counts = _foreground_counts(sigma_array, zl_array, survey_area_sq_degrees,
        Phi_vel_disp, VDF_args, zs, seeing_trsh * seeing_arcsec)
    settings = (M_array_UV, app_magn_limit, seeing_arcsec, SNR, exp_time_sec,
        sky_bckgnd_m_per_arcsec_sq, zero_point_m, photo_band, mag_cut,
        arc_mu_threshold, seeing_trsh, num_exposures, LF_func, restframe_band,
        LENS_LIGHT_FLAG, SIE_FLAG)
    return _source_counts((zs, _dzs), sigma_array, zl_array, counts, settings)


def calculate_num_lenses_and_prob(sigma_array, zl_array, zs_array, M_array_UV, app_magn_limit,
    survey_area_sq_degrees, seeing_arcsec, SNR, exp_time_sec, sky_bckgnd_m_per_arcsec_sq,
    zero_point_m, photo_band, mag_cut=None, arc_mu_threshold=3, seeing_trsh=1.5,
    num_exposures=1, Phi_vel_disp=ls.Phi_vel_disp_Mason, VDF_args=None,
    LF_func=ls.schechter_LF, restframe_band='galex_NUV', LENS_LIGHT_FLAG=False,
    SIE_FLAG=True, ncores=None
):
    """Return counts on (observable source redshift, velocity dispersion, lens redshift).

    The selection and population arguments have the same meaning as in lens_stat.
    VDF_args are passed positionally after (sigma, zl). Each worker integrates a
    complete magnitude distribution before returning its count for a foreground bin.
    """
    if LENS_LIGHT_FLAG and photo_band not in (
        'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0', 'ukirt_wfcam_Y',
        'ukirt_wfcam_J', 'ukirt_wfcam_H', 'ukirt_wfcam_K'
    ):
        print('Photo band not supported for lens light fitting')
        return 0, 0, 0
    if mag_cut is None:
        mag_cut = app_magn_limit
    if M_array_UV[0] > M_array_UV[-1]:
        M_array_UV = M_array_UV[::-1]
    visible = zs_array <= ls.get_highest_LYA_rest_fram_observable(photo_band)
    if not np.any(visible):
        return np.zeros((0, len(sigma_array), len(zl_array)))
    dzs = np.diff(zs_array)
    planes = list(zip(zs_array[visible], np.r_[dzs[0], dzs][visible]))
    counts = _foreground_counts(sigma_array, zl_array, survey_area_sq_degrees,
        Phi_vel_disp, VDF_args, max(zs_array[visible]),
        seeing_trsh * seeing_arcsec)
    settings = (M_array_UV, app_magn_limit, seeing_arcsec, SNR, exp_time_sec,
        sky_bckgnd_m_per_arcsec_sq, zero_point_m, photo_band, mag_cut,
        arc_mu_threshold, seeing_trsh, num_exposures, LF_func, restframe_band,
        LENS_LIGHT_FLAG, SIE_FLAG)
    integrate = partial(_source_counts, sigma_array=sigma_array, zl_array=zl_array,
        lens_counts=counts, settings=settings)
    if ncores is None:
        ncores = multiprocessing.cpu_count()
    ncores = min(ncores, len(planes))
    if ncores == 1:
        return np.stack([integrate(plane) for plane in planes])
    method = 'fork' if sys.platform.startswith('linux') else 'spawn'
    with multiprocessing.get_context(method).Pool(ncores) as pool:
        return np.stack(pool.map(integrate, planes))

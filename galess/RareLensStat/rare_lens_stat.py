"""Survey forecasts for cusp, quad and double-source-plane lenses."""

import numpy as np
from scipy import signal
from tqdm import tqdm

from galess.LensStat import lens_stat as ls
from galess.Utils import ls_utils as utils


def _foreground_counts(sigma_array, zl_array, area, Phi_vel_disp):
    """Count early-type galaxies in each survey volume and velocity bin."""
    counts = np.zeros((len(sigma_array), len(zl_array)))
    dsigma = np.diff(sigma_array)
    dsigma = np.insert(dsigma, 0, dsigma[0])
    dzl = np.diff(zl_array)
    for izl, zl in enumerate(zl_array):
        if zl == 0:
            continue
        volume = ls.Lens_cone_volume_diff(zl, area, dz=dzl[max(0, izl-1)])
        for isg, (sigma, width) in enumerate(zip(sigma_array, dsigma)):
            density = Phi_vel_disp(sigma - width/2, zl) + Phi_vel_disp(sigma + width/2, zl)
            counts[isg, izl] = volume * density * width/2
    return counts


def calculate_num_quad_cusps_lenses(sigma_array, zl_array, zs_array, M_array_UV,
    app_magn_limit, survey_area_sq_degrees, seeing_arcsec, SNR, exp_time_sec,
    sky_bckgnd_m_per_arcsec_sq, zero_point_m, photo_band, mag_cut=None, arc_mu_threshold=3,
    seeing_trsh=1.5, num_exposures=1, Phi_vel_disp=ls.Phi_vel_disp_Mason,
    LF_func=ls.schechter_LF, restframe_band='galex_NUV', LENS_LIGHT_FLAG=False, SIE_FLAG=True):
    """Count detectable third and fourth images of SIE lenses.

    Returns two count grids and the Einstein radius in arcsec, all with axes
    (source redshift, velocity dispersion, lens redshift). This model assumes
    subtracted lens light and selects on image brightness and resolution alone.
    The other survey parameters allow the same inputs as the ordinary lens forecast.
    """
    if mag_cut is None:
        mag_cut = app_magn_limit
    if M_array_UV[0] > M_array_UV[-1]:
        M_array_UV = M_array_UV[::-1]
    shape = (len(zs_array), len(sigma_array), len(zl_array))
    cusp, quad, radius = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    foreground = _foreground_counts(sigma_array, zl_array, survey_area_sq_degrees, Phi_vel_disp)
    dzs = np.diff(zs_array)
    zmax = ls.get_highest_LYA_rest_fram_observable(photo_band)
    for izs in tqdm(np.flatnonzero((zs_array > 0) & (zs_array <= zmax))):
        zs = zs_array[izs]
        width = dzs[max(0, izs-1)]
        dist_mod = 5 * np.log10(ls.cosmo_luminosity_distance(zs) * 1e5)
        m = M_array_UV + (dist_mod + ls.K_correction(zs, photo_band, restframe_band, M_array_UV))
        M_lim = app_magn_limit - dist_mod
        M_lim -= ls.K_correction(zs, photo_band, restframe_band, M_lim)
        idx = np.argmin(np.power(m - mag_cut, 2))
        frac3 = ls.Fraction_Nth_image_above_Mlim_SIE(3, M_array_UV, M_lim, LF_func, zs)
        frac4 = ls.Fraction_Nth_image_above_Mlim_SIE(4, M_array_UV, M_lim, LF_func, zs)
        for izl, isg, theta, prob in ls._source_geometry(
            sigma_array, zl_array, zs, width, M_array_UV, LF_func, True,
            seeing_trsh * seeing_arcsec):
            count = foreground[isg, izl]
            cusp[izs, isg, izl] = np.cumsum(prob * frac3, axis=1)[:, idx] * count
            quad[izs, isg, izl] = np.cumsum(prob * frac4, axis=1)[:, idx] * count
            radius[izs, isg, izl] = theta
    return cusp, quad, radius


def calculate_num_double_lenses_and_prob(sigma_array, zl_array, zs_array, M_array_UV,
    app_magn_limit1, survey_area_sq_degrees1, seeing_arcsec1, SNR1, exp_time_sec1,
    sky_bckgnd_m_per_arcsec_sq1, zero_point_m1, photo_band1, mag_cut1=None,
    app_magn_limit2=None, survey_area_sq_degrees2=None, seeing_arcsec2=None, SNR2=None,
    exp_time_sec2=None, sky_bckgnd_m_per_arcsec_sq2=None, zero_point_m2=None,
    photo_band2=None, mag_cut2=None, arc_mu_threshold=3, seeing_trsh=1.5, num_exposures=1,
    LENS_LIGHT_FLAG=False, SIE_FLAG=True, FLAG_KCORRECTION=True):
    """Forecast two detectable sources at different redshifts behind the same deflector.

    The first survey sets the sky area. The second observation follows the same
    foreground galaxies and can override the depth, band and exposure for the
    farther source. Each source has its own flux and signal-to-noise selection.
    The nearer Einstein radius must resolve the poorer of the two seeing values.

    Returns the expected counts (zs1, zs2, sigma, zl), two Einstein-radius
    grids (zs, sigma, zl) in arcsec and the magnitude-resolved tensor. The
    latter weights same-magnitude pairs by the conditional probability of the
    farther source in that magnitude bin. Counts integrate the two source
    magnitudes independently before multiplying their probabilities.
    """
    if M_array_UV[0] > M_array_UV[-1]:
        M_array_UV = M_array_UV[::-1]
    grid = (len(sigma_array), len(zl_array))
    shape = (len(zs_array), len(zs_array), *grid)
    matrix = np.zeros(shape)
    tensor = np.zeros((*shape, len(M_array_UV)))
    theta1, theta2 = np.zeros(shape[1:]), np.zeros(shape[1:])
    norm2 = np.zeros(len(M_array_UV))
    if mag_cut1 is None:
        mag_cut1 = app_magn_limit1
    if mag_cut2 is None:
        mag_cut2 = mag_cut1 if app_magn_limit2 is None else app_magn_limit2
    first = (app_magn_limit1, seeing_arcsec1, SNR1, exp_time_sec1,
        sky_bckgnd_m_per_arcsec_sq1, zero_point_m1, photo_band1, mag_cut1)
    overrides = (app_magn_limit2, seeing_arcsec2, SNR2, exp_time_sec2,
        sky_bckgnd_m_per_arcsec_sq2, zero_point_m2, photo_band2, mag_cut2)
    second = tuple(a if b is None else b for a, b in zip(first, overrides))
    foreground = _foreground_counts(sigma_array, zl_array, survey_area_sq_degrees1,
        ls.Phi_vel_disp_Mason)
    seeing = max(first[1], second[1])
    # Each (survey, source redshift) has the same selection in every allowed pair.
    dzs = np.diff(zs_array)
    planes = {}
    for survey in (first, second):
        if survey in planes:
            continue
        limit, _, snr, exposure, sky, zp, band, cut = survey
        if LENS_LIGHT_FLAG:
            assert band in ('sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0', 'ukirt_wfcam_Y',
                'ukirt_wfcam_J', 'ukirt_wfcam_H', 'ukirt_wfcam_K'), 'Unsupported lens light band'
        planes[survey] = {}
        zmax = ls.get_highest_LYA_rest_fram_observable(band)
        for izs in np.flatnonzero((zs_array > 0) & (zs_array <= zmax)):
            zs = zs_array[izs]
            width = dzs[max(0, izs-1)]
            radius, integrated = np.zeros(grid), np.zeros(grid)
            weights = np.zeros((*grid, len(M_array_UV)))
            for izl, isg, theta, prob, idx in ls._survey_source_plane(
                sigma_array, zl_array, zs, width, M_array_UV, limit, seeing, snr, exposure,
                sky, zp, band, cut, arc_mu_threshold, seeing_trsh, num_exposures,
                ls.schechter_LF, 'galex_NUV', LENS_LIGHT_FLAG, SIE_FLAG, FLAG_KCORRECTION):
                radius[isg, izl] = theta
                weights[isg, izl] = prob
                integrated[isg, izl] = np.cumsum(prob, axis=1)[:, idx]
            planes[survey][izs] = radius, weights, integrated

    for near, (r1, w1, p1) in tqdm(planes[first].items()):
        for far, (r2, w2, p2) in planes[second].items():
            if zs_array[far] <= zs_array[near]:
                continue
            selected = (r1 > 0) & (r2 > 0)
            theta1[near][selected] = r1[selected]
            theta2[far][selected] = r2[selected]
            matrix[near, far] = p1 * p2 * foreground
            np.multiply(w1, w2, out=tensor[near, far])
            tensor[near, far] *= foreground[:, :, None]
            # Accumulate in the reference order (near, far, sigma, lens redshift).
            for weight in w2[selected]:
                norm2 += weight
    # No farther source is detectable in an empty magnitude bin.
    np.divide(tensor, norm2, out=tensor, where=norm2 != 0)
    return matrix, theta1, theta2, tensor


def get_N_and_P_projections_double_lens(N_gal_matrix, sigma_array, zl_array, zs_array,
    SMOOTH=True):
    """Project double-source counts onto pairs of axes and normalize the four marginals.

    The first two entries repeat the (sigma, zl) grid for compatibility with
    dev. Next are (zs1, zl), (zs2, zl), (zs1, sigma), (zs2, sigma), (zs1, zs2),
    then the densities in zs1, zs2, zl and sigma, in that order.
    """
    pairs = [N_gal_matrix.sum(axis=ax) for ax in
        ((0, 1), (1, 2), (0, 2), (1, 3), (0, 3), (2, 3))]
    total = N_gal_matrix.sum()
    axes = ((1, 2, 3), (0, 2, 3), (0, 1, 2), (0, 1, 3))
    grids = (zs_array, zs_array, zl_array, sigma_array)
    densities = [N_gal_matrix.sum(axis=ax) / total * (1 / (grid[1] - grid[0]))
        for ax, grid in zip(axes, grids)]
    if SMOOTH:
        kernel = np.ones((3, 3)) / 9
        pairs = [signal.convolve2d(pair, kernel, mode='same') for pair in pairs]
        if zl_array[0] == 0:
            for pair in pairs[:3]:
                pair[:, 0] = 0
        if zs_array[0] == 0:
            for pair in pairs[1:]:
                pair[0] = 0
            pairs[-1][:, 0] = 0
        kernel = np.ones(3) / 3
        densities = [np.append(p[0], signal.convolve(p[1:], kernel, mode='same', method='direct'))
            for p in densities]
    return (pairs[0], *pairs, *densities)


def get_param_space_idx_from_obs_constraints(CW_ER_zl, CW_ER_zs, E_ring_rad,
    zs_array, zl_array, sigma_array):
    """Find (zl, zs, sigma) indices consistent with an observed Einstein radius in arcsec.

    Redshifts are given as (central value, upper error, lower error). For each
    allowed redshift pair use the nearest velocity-dispersion bin.
    """
    zl, dzl_hi, dzl_lo = CW_ER_zl
    zs, dzs_hi, dzs_lo = CW_ER_zs
    source = (zs_array > zs - dzs_lo) & (zs_array < zs + dzs_hi)
    lens = (zl_array > zl - dzl_lo) & (zl_array < zl + dzl_hi)
    izs, izl = np.nonzero(source[:, None] & lens & (zs_array[:, None] > zl_array))
    # Evaluate only the allowed pairs, retaining redshift order and nearest-bin ties.
    sigmas = np.array([ls.sigma_from_R_Ein(zs_array[i], zl_array[j], E_ring_rad)
        for i, j in zip(izs, izl)])
    use = sigmas > 0
    isg = np.abs(sigmas[use, None] - sigma_array).argmin(axis=1)
    return izl[use], izs[use], isg


def prob_for_obs_conf_in_param_space_per_sq_degree(survey_title, CW_ER_zl, CW_ER_zs,
    E_ring_rad, zs_array, zl_array, sigma_array, CHECK_LL=True):
    """Return the expected number per square degree compatible with an observed ring."""
    area = utils.read_survey_params(survey_title)['area']
    matrix = utils._load_survey_result(survey_title, 'matrix_LL' if CHECK_LL else 'matrix_noLL')
    izl, izs, isg = get_param_space_idx_from_obs_constraints(CW_ER_zl, CW_ER_zs, E_ring_rad,
        zs_array, zl_array, sigma_array)
    return sum(matrix[izs, isg, izl]) / area


def get_src_magnitude_distr(m_obs, m_cut, zs_array, prob, M_array_UV, obs_band='sdss_i0'):
    """Bin magnified sources brighter than m_cut by their nearest apparent magnitude.

    prob is the ordinary lens forecast tensor with axes (zs, sigma, zl, M).
    The arc magnitude assumes a factor of three magnification.
    """
    if M_array_UV[0] > M_array_UV[-1]:
        M_array_UV = M_array_UV[::-1]
    counts = np.zeros(len(m_obs))
    per_magnitude = np.sum(prob, axis=(1, 2))
    for izs in np.flatnonzero(zs_array > 0):
        zs = zs_array[izs]
        corr = 5 * np.log10(ls.cosmo_luminosity_distance(zs) * 1e5)
        corr += ls.K_correction_from_UV(zs, obs_band, M_array_UV)
        m = M_array_UV + corr - 2.5 * np.log10(3)
        bright = m <= m_cut
        bins = np.abs(m[bright, None] - m_obs).argmin(axis=1)
        np.add.at(counts, bins, per_magnitude[izs, bright])
    return counts

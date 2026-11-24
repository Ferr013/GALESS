"""Module providing a set of plotting functions."""

import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from astropy.cosmology import FlatLambdaCDM

from galess.ComputeSurveys import compute_surveys as surveys
import galess.LensStat.lens_stat as ls
import galess.Utils.ls_utils as utils

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

def set_plt_param(PLOT_FOR_KEYNOTE = 0):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['figure.figsize'  ] = (3.3,2.0)
    plt.rcParams['font.size'       ] = 8
    plt.rcParams['axes.labelsize'  ] = 14
    plt.rcParams['legend.fontsize' ] = 13
    plt.rcParams['legend.frameon'  ] = False
    plt.rcParams['legend.title_fontsize'] = 20
    plt.rcParams['font.family'     ] = 'STIXGeneral'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['xtick.direction' ] = 'in'
    plt.rcParams['ytick.direction' ] = 'in'
    plt.rcParams['xtick.top'       ] = True
    plt.rcParams['ytick.right'     ] = True
    plt.rcParams['xtick.labelsize' ] = 16
    plt.rcParams['ytick.labelsize' ] = 16
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['xtick.major.width'] = 1.25
    plt.rcParams['xtick.minor.width'] = 0.75
    plt.rcParams['ytick.major.width'] = 1.25
    plt.rcParams['ytick.minor.width'] = 0.75
    params_keynote = {
        "lines.color": "white",
        "patch.edgecolor": "white",
        "text.color": "white",
        "axes.facecolor": '#222222',
        "axes.edgecolor": "lightgray",
        "axes.labelcolor": 'white',
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "lightgray",
        "figure.facecolor": '#222222',
        "figure.edgecolor": 'lightgray',
        "savefig.facecolor":'#222222',
        "savefig.edgecolor": 'lightgray',
        }
    line_c = 'k'
    cmap_c = cm.inferno
    _col_  = None
    col_A  = 'k'
    col_B  = 'r'
    col_C  = 'm'
    col_D  = 'orange'
    fn_prefix = ''
    if PLOT_FOR_KEYNOTE:
        plt.rcParams.update(params_keynote)
        line_c = 'w'
        cmap_c = cm.cool
        _col_  = iter(cmap_c(np.linspace(0, 1, 4)))
        col_A  = next(_col_)
        col_B  = next(_col_)
        col_C  = next(_col_)
        col_D  = next(_col_)
        fn_prefix = 'KEY_'
    return line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix

def plot_ALL_distributions(title, zl_array, zs_array, sigma_array,
                               Theta_E_LL, matrix_LL, Theta_E_noLL, matrix_noLL,
                               PLOT_FOR_KEYNOTE = 0, SMOOTH = 1, SAVE = 0, LEGEND = 0):
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)

    Ngal_zl_sigma_noLL, Ngal_zs_sigma_noLL, Ngal_zs_zl_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH)
    Ngal_zl_sigma_LL, Ngal_zs_sigma_LL, Ngal_zs_zl_LL, P_zs_LL, P_zl_LL, P_sg_LL = ls.get_N_and_P_projections(matrix_LL, sigma_array, zl_array, zs_array, SMOOTH)
    _nbins_Re = np.arange(0  , 4  , 0.25)

    fig, ax = plt.subplots(2, 3, figsize=(17, 10), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.15, hspace=.2)
    fig.suptitle(title, fontsize=25)
    ax[0,0].plot(zl_array, P_zl_LL, c=col_A, ls=':')
    ax[0,0].plot(zs_array, P_zs_LL, c=col_B, ls=':' , label='w/ lens light')
    ax[0,0].plot(zl_array, P_zl_noLL, c=col_A, ls='-')
    ax[0,0].plot(zs_array, P_zs_noLL, c=col_B, ls='-', label='No lens light')
    ax[0,0].set_ylabel(r'$P$', fontsize=20)
    ax[0,0].set_xlabel(r'$z$', fontsize=20)
    ax[0,0].set_xlim((0,5.2))
    ax[0,1].plot(sigma_array, P_sg_LL, c=col_C, ls = ':')
    ax[0,1].plot(sigma_array, P_sg_noLL, c=col_C, ls = '-')
    ax[0,1].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
    ax[0,2].hist(np.ravel(Theta_E_LL), weights=np.ravel(matrix_LL), bins=_nbins_Re, range=(0, 3), density=True, histtype='step', color=col_D, ls = ':')
    ax[0,2].hist(np.ravel(Theta_E_noLL), weights=np.ravel(matrix_noLL), bins=_nbins_Re, range=(0, 3), density=True, histtype='step', color=col_D, ls = '-')
    ax[0,2].set_xlabel(r'$\Theta_E$ [arcsec]', fontsize=20)

    level_array = [0.01, 0.05, 0.2, 0.4, 0.8]
    norm = np.sum(matrix_noLL)
    plotting_now = Ngal_zs_zl_noLL/norm
    _zs, _zl = np.meshgrid(zs_array, zl_array)
    levels   = np.asarray(level_array)*(np.power(10,(np.log10(np.max(plotting_now)))))
    contours = ax[1,0].contour(_zs, _zl, plotting_now.T, levels, cmap=cmap_c, norm=colors.Normalize(vmin=np.min(levels), vmax=np.max(plotting_now)), linestyles='-')
    ax[1,0].clabel(contours, inline=True, fontsize=8, fmt='%.0e')
    ax[1,0].set_xlabel(r'$z_s$', fontsize=20)
    ax[1,0].set_ylabel(r'$z_l$', fontsize=20)
    plotting_now = Ngal_zl_sigma_noLL/norm
    _sigma, _zl = np.meshgrid(sigma_array, zl_array)
    levels   = np.asarray(level_array)*(np.power(10,(np.log10(np.max(plotting_now)))))
    contours = ax[1,1].contour(_sigma, _zl,plotting_now.T, levels, cmap=cmap_c, norm=colors.Normalize(vmin=np.min(levels), vmax=np.max(plotting_now)), linestyles='-')
    ax[1,1].scatter(200,1.0, label='', alpha=0)
    ax[1,1].clabel(contours, inline=True, fontsize=8, fmt='%.0e')
    ax[1,1].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
    ax[1,1].set_ylabel(r'$z_l$', fontsize=20)
    plotting_now = Ngal_zs_sigma_noLL/norm
    _sigma, _zs = np.meshgrid(sigma_array, zs_array)
    levels   = np.asarray(level_array)*(np.power(10,(np.log10(np.max(plotting_now)))))
    contours = ax[1,2].contour(_sigma, _zs, plotting_now, levels, cmap=cmap_c, norm=colors.Normalize(vmin=np.min(levels), vmax=np.max(plotting_now)), linestyles='-')
    ax[1,2].clabel(contours, inline=True, fontsize=8, fmt='%.0e')
    ax[1,2].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
    ax[1,2].set_ylabel(r'$z_s$', fontsize=20)

    norm = np.sum(matrix_LL)
    plotting_now = Ngal_zs_zl_LL/norm
    _zs, _zl = np.meshgrid(zs_array, zl_array)
    levels   = np.asarray(level_array)*(np.power(10,(np.log10(np.max(plotting_now)))))
    contours = ax[1,0].contour(_zs, _zl, plotting_now.T, levels, cmap=cmap_c, norm=colors.Normalize(vmin=np.min(levels), vmax=np.max(plotting_now)), linestyles=':')
    # ax[1,0].clabel(contours, inline=True, fontsize=8)
    ax[1,0].set_xlabel(r'$z_s$', fontsize=20)
    ax[1,0].set_ylabel(r'$z_l$', fontsize=20)
    plotting_now = Ngal_zl_sigma_LL/norm
    _sigma, _zl = np.meshgrid(sigma_array, zl_array)
    levels   = np.asarray(level_array)*(np.power(10,(np.log10(np.max(plotting_now)))))
    contours = ax[1,1].contour(_sigma, _zl,plotting_now.T, levels, cmap=cmap_c, norm=colors.Normalize(vmin=np.min(levels), vmax=np.max(plotting_now)), linestyles=':')
    ax[1,1].scatter(200,1.0, label='', alpha=0)
    # ax[1,1].clabel(contours, inline=True, fontsize=8)
    ax[1,1].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
    ax[1,1].set_ylabel(r'$z_l$', fontsize=20)
    plotting_now = Ngal_zs_sigma_LL/norm
    _sigma, _zs = np.meshgrid(sigma_array, zs_array)
    levels   = np.asarray(level_array)*(np.power(10,(np.log10(np.max(plotting_now)))))
    contours = ax[1,2].contour(_sigma, _zs, plotting_now, levels, cmap=cmap_c, norm=colors.Normalize(vmin=np.min(levels), vmax=np.max(plotting_now)), linestyles=':')
    # ax[1,2].clabel(contours, inline=True, fontsize=8)
    ax[1,2].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
    ax[1,2].set_ylabel(r'$z_s$', fontsize=20)

    if LEGEND:
        if(np.sum(matrix_noLL))>10_000:
            ax[1,0].legend([f'#Lenses w/ LL: {np.sum(matrix_LL):.1e}', f'#Lenses no LL: {np.sum(matrix_noLL):.1e}'], fontsize=20)
        else:
            ax[1,0].legend([f'#Lenses w/ LL: {np.sum(matrix_LL):.0f}', f'#Lenses no LL: {np.sum(matrix_noLL):.0f}'], fontsize=20)
    # ax[1,1].set_xlim((100,400))
    # ax[1,0].set_xlim((100,400))
    # ax[1,0].set_ylim((0,2.5))
    plt.tight_layout()
    if (SAVE):
        folderpath = 'img/'+utils.remove_spaces_from_string(title)
        if not os.path.exists(folderpath): os.makedirs(folderpath)
        plt.savefig(folderpath+'/'+fn_prefix+'all_plts.jpg', dpi=200)
    plt.show()

def plot_effect_vel_disp_function(zl_array, zs_array, sigma_array,
                                    PLOT_FOR_KEYNOTE = 0, LENS_LIGHT = 1,
                                    SMOOTH = 0, SAVE = 0, READ_FILES = 1):
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.15, hspace=.2)
    for title, lstyle, label, VDF in zip(['EUCLID Wide VIS', 'EUCLID Wide VIS VDF Choi', 'EUCLID Wide VIS VDF Geng'],
                                ['-', '--', ':'],
                                ['VDF Mason et al. 2015', 'VDF Choi et al. 2007', 'VDF Geng et al. 2021'],
                                [ls.Phi_vel_disp_Mason, ls.Phi_vel_disp_SDSS, ls.Phi_vel_disp_Geng]):
        try:
            if READ_FILES:
                matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
            else:
                raise(ValueError)
        except ValueError:
            M_array     = np.linspace(-13 , -25 , 25)
            min_SNR     = 20
            arc_mu_thr  = 3
            survey_params = utils.read_survey_params('EUCLID Wide VIS', VERBOSE = 0)
            limit    = survey_params['limit']
            cut      = survey_params['cut']
            area     = survey_params['area']
            seeing   = survey_params['seeing']
            exp_time_sec = survey_params['exp_time_sec']
            zero_point_m = survey_params['zero_point_m']
            sky_bckgnd_m = survey_params['sky_bckgnd_m']
            photo_band   = survey_params['photo_band']
            print('FILE do NOT exist - RUNNING MODEL')
            if LENS_LIGHT:
                matrix_LL, Theta_E_LL, prob_LL = ls.calculate_num_lenses_and_prob(
                                                                            sigma_array, zl_array, zs_array, M_array, limit, area,
                                                                            seeing, min_SNR, exp_time_sec, sky_bckgnd_m, zero_point_m,
                                                                            photo_band = photo_band, mag_cut=cut, arc_mu_threshold = arc_mu_thr,
                                                                            Phi_vel_disp = VDF, LENS_LIGHT_FLAG = True, SIE_FLAG = True)
            else:
                matrix_noLL, Theta_E_noLL, prob_noLL = ls.calculate_num_lenses_and_prob(
                                                                            sigma_array, zl_array, zs_array, M_array, limit, area,
                                                                            seeing, min_SNR, exp_time_sec, sky_bckgnd_m, zero_point_m,
                                                                            photo_band = photo_band, mag_cut=cut, arc_mu_threshold = arc_mu_thr,
                                                                            Phi_vel_disp = VDF, LENS_LIGHT_FLAG = False, SIE_FLAG = True)
        matrix, theta = (matrix_LL, Theta_E_LL) if LENS_LIGHT else (matrix_noLL, Theta_E_noLL)
        *_, P_zs_noLL, P_zl_noLL, P_sg_noLL = ls.get_N_and_P_projections(
            matrix, sigma_array, zl_array, zs_array, SMOOTH)
        ax[0].plot(zl_array, P_zl_noLL, c=col_A, ls=lstyle, label=f'{label}')
        ax[0].plot(zs_array, P_zs_noLL, c=col_B, ls=lstyle)
        ax[0].set_ylabel(r'$dP/dz$', fontsize=20)
        ax[0].set_xlabel(r'$z$', fontsize=20)
        ax[0].set_xlim((0,5.2))
        ax[1].plot(sigma_array, P_sg_noLL, c=col_C, ls = lstyle)
        ax[1].set_ylabel(r'$dP/d\sigma$', fontsize=20)
        ax[1].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
        _nbins_Re = np.arange(0  , 4  , 0.25)
        ax[2].hist(theta.ravel(), weights=matrix.ravel(), bins=_nbins_Re, density=True,
            histtype='step', color=col_D, ls=lstyle, label=f'# Lenses: {np.sum(matrix):.1e}')
        ax[2].set_ylabel(r'$dP/d\Theta_E$', fontsize=20)
        ax[2].set_xlabel(r'$\Theta_E$ [arcsec]', fontsize=20)
    ax[0].legend(fontsize=14, frameon=False)
    ax[2].legend(fontsize=14, frameon=False)
    plt.tight_layout()
    if (SAVE):
        folderpath = 'img/'+utils.remove_spaces_from_string(title)
        if not os.path.exists(folderpath): os.makedirs(folderpath)
        plt.savefig(folderpath+'/VDF_effect_corner_plts.jpg', dpi=200)
    plt.show()

def _survey_results(title, sigma_array, zl_array, zs_array, read=True, survey=None,
    VDF=ls.Phi_vel_disp_Mason, arc_mu_threshold=3):
    """Load the saved light/no-light pair, or calculate it on the supplied grid."""
    standard = (np.array_equal(sigma_array, surveys.sigma_array)
        and np.array_equal(zl_array, surveys.zl_array)
        and np.array_equal(zs_array, surveys.zs_array))
    # Existing named results are explicit inputs to the plot. Fresh plot forecasts
    # use SIE in both light treatments. Keep them apart from the batch SIS-light model.
    plot_title = title + ' SIE lens light'
    if standard and read:
        for key in (title, plot_title):
            try:
                result = utils.load_pickled_files(key)
            except ValueError:
                continue
            if result[0].shape == (len(zs_array), len(sigma_array), len(zl_array)):
                return result
    p = utils.read_survey_params(survey or title)
    args = (sigma_array, zl_array, zs_array, np.linspace(-13, -25, 25), p['limit'],
        p['area'], p['seeing'], 20, p['exp_time_sec'], p['sky_bckgnd_m'], p['zero_point_m'],
        p['photo_band'])
    result = tuple(value for light in (True, False)
        for value in ls.calculate_num_lenses_and_prob(*args, mag_cut=p['cut'],
            Phi_vel_disp=VDF, LENS_LIGHT_FLAG=light, arc_mu_threshold=arc_mu_threshold))
    if standard:
        utils.save_pickled_files(plot_title, *result)
    return result


def compare_ALL_distributions_surveys(surveys_selection, sigma_array, zl_array, zs_array,
    LENS_LIGHT=1, PLOT_FOR_KEYNOTE=0, SMOOTH=1, SAVE=0, LABEL_NUMBERS=0, LOG=0):
    """Compare the redshift, velocity dispersion and radius distributions of surveys."""
    _, cmap, *_ = set_plt_param(PLOT_FOR_KEYNOTE)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    fig.subplots_adjust(wspace=.235)
    colors = cmap(np.linspace(0, 1, len(surveys_selection) + 1))
    for title, color in zip(surveys_selection, colors):
        result = _survey_results(title, sigma_array, zl_array, zs_array)
        matrix, theta, _ = result[:3] if LENS_LIGHT else result[3:]
        *_, P_zs, P_zl, P_sg = ls.get_N_and_P_projections(
            matrix, sigma_array, zl_array, zs_array, SMOOTH)
        ax[0].plot(zl_array, P_zl, c=color, label=title)
        ax[0].plot(zs_array, P_zs, c=color, ls=':')
        ax[1].plot(sigma_array, P_sg, c=color, label=title)
        label = f'{title} ({matrix.sum():.1e})' if LABEL_NUMBERS else title
        plot_R_distribution_in_ax(ax[2], title, color, np.arange(0, 4, .25), matrix, theta,
            label=label)
    for a, x, y in zip(ax, ('$z$', r'$\sigma$', r'$\Theta_E$ [arcsec]'),
        ('$dP/dz$', r'$dP/d\sigma$', r'$dP/d\Theta_E$')):
        a.set_xlabel(x, fontsize=20)
        a.set_ylabel(y, fontsize=20)
        if LOG:
            a.set_yscale('log')
    ax[0].legend(fontsize=12)
    if LABEL_NUMBERS:
        ax[2].legend(fontsize=12)
    if SAVE:
        os.makedirs('img', exist_ok=True)
        fig.savefig('img/comp_surveys.png', dpi=200, bbox_inches='tight')
    plt.show()


def plot_z_distribution_in_ax(ax, title, color, zl_array, zs_array, sigma_array, matrix,
    SMOOTH=1, lw=None):
    *_, P_zs, P_zl, P_sg = ls.get_N_and_P_projections(matrix, sigma_array, zl_array,
        zs_array, SMOOTH)
    ax.plot(zl_array, P_zl, c=color, lw=lw, label=title)
    ax.plot(zs_array, P_zs, c=color, lw=lw, ls=':')


def plot_s_distribution_in_ax(ax, title, color, zl_array, zs_array, sigma_array, matrix,
    SMOOTH=1, lw=None):
    P_sg = ls.get_N_and_P_projections(matrix, sigma_array, zl_array, zs_array, SMOOTH)[-1]
    ax.plot(sigma_array, P_sg, c=color, label=title, lw=lw)


def plot_R_distribution_in_ax(ax, title, color, _nbins_Re, matrix, Theta_E, SMOOTH=1,
    label='', lw=None):
    ax.hist(Theta_E.ravel(), weights=matrix.ravel(), bins=_nbins_Re, density=True,
        histtype='step', color=color, lw=lw, label=label or title)


def compare_SL2S(zl_array, zs_array, sigma_array, LENS_LIGHT=1, PLOT_FOR_KEYNOTE=0,
    SMOOTH=1, SAVE=0, BEST_SAMPLE=False):
    """Compare the SL2S measurements with the Mason and Geng VDF forecasts."""
    folder = os.path.join(utils.BASEPATH, 'LENS_SEARCHES', 'SL2S_Sonnenfeld')
    spectra = pd.read_csv(os.path.join(folder, 'redshifts_sigma.csv'))
    radii = pd.read_csv(os.path.join(folder, 'data.csv'))
    line, _, _, _, col_B, *_ = set_plt_param(PLOT_FOR_KEYNOTE)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    fig.subplots_adjust(wspace=.23)
    samples = [(spectra, radii, 'Full sample', 'darkorange' if PLOT_FOR_KEYNOTE else 'forestgreen')]
    if BEST_SAMPLE:
        best = spectra[spectra['Grade'] == 'A']
        samples.append((best, radii[radii['Name'].isin(best['Name'])], 'Grade A', 'orange'))
    for data, lenses, label, color in samples:
        for a, values, bins, style, name in (
            (ax[0], data['z_l'], np.arange(0, 1.2, .3), '-', label),
            (ax[0], data['z_s'], np.arange(.5, 4, .5), '--', ''),
            (ax[1], data['sigma'], np.arange(100, 400, 25), '-', ''),
            (ax[2], lenses['R_Ein'], np.arange(0, 4, .25), '-', '')
        ):
            a.hist(values.dropna(), bins=bins, density=True, histtype='step',
                lw=3, color=color, ls=style, label=name)
    for title, vdf, color in (('CFHTLS i band', ls.Phi_vel_disp_Mason, line),
        ('CFHTLS i band VDF Geng', ls.Phi_vel_disp_Geng, col_B)):
        result = _survey_results(title, sigma_array, zl_array, zs_array, survey='CFHTLS i band',
            VDF=vdf)
        matrix, theta, _ = result[:3] if LENS_LIGHT else result[3:]
        *_, P_zs, P_zl, P_sg = ls.get_N_and_P_projections(matrix, sigma_array,
            zl_array, zs_array, SMOOTH)
        ax[0].plot(zl_array, P_zl, c=color, label=title)
        ax[0].plot(zs_array, P_zs, c=color, ls='--')
        ax[1].plot(sigma_array, P_sg, c=color)
        plot_R_distribution_in_ax(ax[2], title, color, np.arange(0, 4, .2), matrix, theta)
    ax[0].set(xlabel='$z$', ylabel='$dP/dz$', xlim=(0, 5.2))
    ax[1].set(xlabel=r'$\sigma$ [km/s]', ylabel=r'$dP/d\sigma$')
    ax[2].set(xlabel=r'$\Theta_E$ [arcsec]', ylabel=r'$dP/d\Theta_E$', xlim=(0, 4))
    ax[0].legend(fontsize=10)
    if SAVE:
        os.makedirs('img', exist_ok=True)
        fig.savefig('img/SL2S_Sonnenfeld.png', dpi=200, bbox_inches='tight')
    plt.show()


def model_response_m_cut():
    ### COMPUTING effect of m_cut on the number of lenses and the parameters distributions ###
    ___PLOT_FOR_KEYNOTE___ = 0
    LENS_DENSITY  = 0
    title = 'PEARLS NEP F115W'
    survey_params = utils.read_survey_params(title, VERBOSE = 0)
    arc_mu_thr = 3
    min_SNR    = 20
    limit    = survey_params['limit']
    cut      = survey_params['cut']
    area     = survey_params['area']
    seeing   = survey_params['seeing']
    exp_time_sec = survey_params['exp_time_sec']
    zero_point_m = survey_params['zero_point_m']
    sky_bckgnd_m = survey_params['sky_bckgnd_m']
    photo_band   = survey_params['photo_band']
    M_array = np.linspace(-13 , -25 , 25)
    zl_array_CFHTLS = np.arange(0.0 , 2.1 , 0.1)
    zs_array_CFHTLS = np.arange(0.0 , 5.6 , 0.2)
    sg_array_CFHTLS = np.linspace(100 , 400 , 31)
    delta_cut_limit = np.arange(0, 5.25, 0.5)
    gal_num_vs_mcut, gal_num_vs_mcut_LL = np.zeros(0), np.zeros(0)

    _title_ = 'PEARLS NEP F115W band_mcut_'
    for iid, dlt in enumerate(delta_cut_limit):
        cut   = limit - dlt
        title = _title_ + str(iid)
        try:
            matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
        except ValueError:
            print('FILE do NOT exist - RUNNING MODEL')
            matrix_noLL, Theta_E_noLL, prob_noLL = ls.calculate_num_lenses_and_prob(
                                                                        sg_array_CFHTLS, zl_array_CFHTLS, zs_array_CFHTLS, M_array, limit, area,
                                                                        seeing, min_SNR, exp_time_sec, sky_bckgnd_m, zero_point_m,
                                                                        photo_band = photo_band, mag_cut=cut, arc_mu_threshold = arc_mu_thr,
                                                                        LENS_LIGHT_FLAG = False, SIE_FLAG = True)
            print('FILE do NOT exist - RUNNING MODEL LL')
            matrix_LL, Theta_E_LL, prob_LL = ls.calculate_num_lenses_and_prob(
                                                                        sg_array_CFHTLS, zl_array_CFHTLS, zs_array_CFHTLS, M_array, limit, area,
                                                                        seeing, min_SNR, exp_time_sec, sky_bckgnd_m, zero_point_m,
                                                                        photo_band = photo_band, mag_cut=cut, arc_mu_threshold = arc_mu_thr,
                                                                        LENS_LIGHT_FLAG = True, SIE_FLAG = False)

            utils.save_pickled_files(title,  matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL)
        gal_num_vs_mcut = np.append(gal_num_vs_mcut, np.sum(matrix_noLL))
        gal_num_vs_mcut_LL = np.append(gal_num_vs_mcut_LL, np.sum(matrix_LL))

    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(___PLOT_FOR_KEYNOTE___)
    _c_ = 'w' if ___PLOT_FOR_KEYNOTE___ else 'k'
    fig = plt.figure(figsize=(17, 7))
    grid = plt.GridSpec(2, 3, wspace=0.2, hspace=0.35)
    ax1 = fig.add_subplot(grid[0, :])
    ax2 = fig.add_subplot(grid[1, 0])
    ax3 = fig.add_subplot(grid[1, 1])
    ax4 = fig.add_subplot(grid[1, 2])
    if LENS_DENSITY:
        ax1.plot(limit - delta_cut_limit, gal_num_vs_mcut/area, c = _c_)
        ax1.plot(limit - delta_cut_limit, gal_num_vs_mcut_LL/area, c = _c_, ls='--')
    else:
        ax1.plot(limit - delta_cut_limit, gal_num_vs_mcut, c = _c_)
        ax1.plot(limit - delta_cut_limit, gal_num_vs_mcut_LL, c = _c_, ls='--')
    _col_  = iter(cmap_c(np.linspace(0, 1, len(delta_cut_limit)+1)))
    for iid, dlt in enumerate(delta_cut_limit):
        ccc = next(_col_)
        title = _title_ + str(iid)
        matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
        _ , __  , ___, P_zs_LL   , P_zl_LL  , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL, sg_array_CFHTLS, zl_array_CFHTLS, zs_array_CFHTLS, SMOOTH=1)
        _ , __  , ___, P_zs_noLL , P_zl_noLL  , P_sg_noLL   = ls.get_N_and_P_projections(matrix_noLL, sg_array_CFHTLS, zl_array_CFHTLS, zs_array_CFHTLS, SMOOTH=1)
        if LENS_DENSITY:
            ax1.plot(limit - dlt, np.sum(matrix_noLL)/area, marker='o', c = ccc, ms = 12)
        else:
            ax1.plot(limit - dlt, np.sum(matrix_noLL), marker='o', c = ccc, ms = 12)
        ax2.plot(np.append(0,zl_array_CFHTLS), np.append(0,P_zl_noLL), c=ccc, ls='-', label=str(limit - dlt))
        ax3.plot(np.append(0,zs_array_CFHTLS), np.append(0,P_zs_noLL), c=ccc, ls='-')
        ax4.plot(sg_array_CFHTLS, P_sg_noLL, c=ccc, ls='-')
    ax1.set_yscale('log')
    ax2.set_xlim((0, 2.0))
    ax3.set_xlim((0, 5.5))
    __size_labels__, __size_ticks__ = 20, 15
    ax1.set_xlabel(r'm$_\text{cut}$ [mag]', fontsize=__size_labels__)
    if LENS_DENSITY:
        ax1.set_ylabel(r'N lenses [deg$^{-2}$]' , fontsize=__size_labels__)
    else:
        ax1.set_ylabel(r'N lenses' , fontsize=__size_labels__)
    ax2.set_xlabel(r'$z_l$'    , fontsize=__size_labels__)
    ax3.set_xlabel(r'$z_s$'    , fontsize=__size_labels__)
    ax4.set_xlabel(r'$\sigma$ [km/s]', fontsize=__size_labels__)
    ax2.set_ylabel(r'Probability distr.', fontsize=__size_labels__)
    ax1.tick_params(axis='both', which = 'major', labelsize=__size_ticks__, direction = 'in', length = 8)
    ax1.tick_params(axis='both', which = 'minor', labelsize=__size_ticks__, direction = 'in', length = 5)
    ax2.tick_params(axis='both', which = 'both', labelsize=__size_ticks__, direction = 'in', length = 5)
    ax3.tick_params(axis='both', which = 'both', labelsize=__size_ticks__, direction = 'in', length = 5)
    ax4.tick_params(axis='both', which = 'both', labelsize=__size_ticks__, direction = 'in', length = 5)
    plt.show()

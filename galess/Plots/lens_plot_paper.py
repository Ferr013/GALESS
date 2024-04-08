import os.path
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import kstest
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from astropy.cosmology import FlatLambdaCDM

# import galess.LensStat.lens_stat as ls
# import galess.Utils.ls_utils as utils
import sys
path_root = os.path.split(os.path.abspath(''))[0]
sys.path.append(str(path_root) + '/galess/LensStat/')
sys.path.append(str(path_root) + '/galess/Utils/')
sys.path.append(str(path_root) + '/galess/Plots/')
sys.path.append(str(path_root) + '/galess/RareLensStat/')
import lens_stat as ls
import rare_lens_stat as rls
import ls_utils as utils
from lens_plot import *


def plot_Lens_Fraction(m_lim = 28.5, mu_arc_SIE = 3,
                       M_array = 0, zs_array_plot = 0, schechter_plot = ls.schechter_LF,
                       PLOT_FOR_KEYNOTE = 0, SAVE = 0):
    if zs_array_plot == 0: zs_array_plot = np.asarray((1,2,3,4,5,6,7,8,9))
    if M_array == 0: M_array = np.linspace(-14,-26,37)
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    f, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True, squeeze=True)
    plt.subplots_adjust(wspace=.075, hspace=.075)
    color = iter(cmap_c(np.linspace(0, 1, len(zs_array_plot))))
    l_width = 1.75
    for zs in zs_array_plot:
        _col = next(color)
        M_lim = m_lim - 5 * np.log10(cosmo.luminosity_distance(zs).value * 1e5)
        frl   = ls.Fraction_lensed_SIE(1, M_array, schechter_plot, ls.Phi_vel_disp_Mason, zs)
        ax[0, 0].plot(M_array, frl,        c=_col,  label = r'$z=$'+str(zs), lw=l_width)
        C = 0.8979070411803386 #Correction for elliptical caustic area averaged over axis ratio distribution
        #ax[0, 0].axhline(Tau(zs)*C, c=_col, ls='--')
        F_arc = ls.Fraction_1st_image_arc_SIE(mu_arc_SIE, M_array, schechter_plot, zs)
        ax[0, 1].plot(M_array, frl*F_arc,  c=_col,  label = r'$z=$'+str(zs), lw=l_width)
        img_N = 2
        F_Nth = ls.Fraction_Nth_image_above_Mlim_SIE(img_N, M_array, M_lim, schechter_plot, zs)
        ax[1, 0].plot(M_array, frl*F_Nth,  c=_col,  label = r'$z=$'+str(zs), lw=l_width)
        img_N = 3
        F_Nth = ls.Fraction_Nth_image_above_Mlim_SIE(img_N, M_array, M_lim, schechter_plot, zs)
        ax[1, 1].plot(M_array, frl*F_Nth,  c=_col, ls=':')
        img_N = 4
        F_Nth = ls.Fraction_Nth_image_above_Mlim_SIE(img_N, M_array, M_lim, schechter_plot, zs)
        ax[1, 1].plot(M_array, frl*F_Nth,  c=_col,  label = r'$z=$'+str(zs), lw=l_width)
    # ax[0, 0].set_title(r'Fraction of lensed galaxies',                              fontsize=15)
    # ax[0, 1].set_title(r'Fraction arcs stretched more than $\mu=$'+str(mu_arc_SIE), fontsize=15)
    # ax[1, 0].set_title(r'Fraction of 2nd images above $M_{lim}$',                   fontsize=15)
    # ax[1, 1].set_title(r'Fraction of cusp/quad images above $M_{lim}$',             fontsize=15)
    ax[0, 0].set_ylabel(r'$F_{lens}$', fontsize=22)
    ax[0, 1].set_ylabel(r'$F_{arc}$', fontsize=22)
    ax[1, 0].set_ylabel(r'$F_\text{2nd}$', fontsize=22)
    ax[1, 1].set_ylabel(r'$F_\text{cusp}$ | $F_\text{quad}$', fontsize=22)
    for i in range(2):
        for j in range(2):
            ax[i, j].set_ylim((7e-4,1.2e0))
            ax[i, j].set_xlabel(r'$M_{AB,1}$ ', fontsize=22)
            ax[i, j].set_xlim((-24,-16))
            ax[i, j].set_yscale('log')
            ax[i, j].xaxis.set_major_locator(plt.MultipleLocator(2))
            ax[i, j].xaxis.set_minor_locator(plt.MultipleLocator(1))
            ax[i, j].tick_params(axis='both', which='major', direction='in', right=1, top=1, labelsize=15, length=8, width=1.5)
            ax[i, j].tick_params(axis='both', which='minor', direction='in', right=1, top=1, labelsize=15, length=5, width=1)
    ax[1, 1].legend(title=r'Source $z_s$', loc='upper right', fontsize=20)
    plt.tight_layout()
    if SAVE: plt.savefig('img/'+fn_prefix+'frac_lens_SIE.png', dpi=200, bbox_inches='tight')
    plt.show()

def get_distributions_from_title(title, sigma_array, zl_array, zs_array, LENS_LIGHT = 1, SMOOTH = 1, PRINT = 0,
                                 zl_lim = (0.0, 99), zs_lim = (0.0, 99), sg_lim = (0, 1000), Rein_lim = 0):
    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
    zl_lower, zl_upper = zl_lim
    zs_lower, zs_upper = zs_lim
    sg_lower, sg_upper = sg_lim
    matrix_noLL[np.logical_or(zs_array <= zs_lower, zs_array >= zs_upper), :, :] *= 0
    matrix_noLL[:, np.logical_or(sigma_array <= sg_lower, sigma_array >= sg_upper), :] *= 0
    matrix_noLL[:, :, np.logical_or(zl_array <= zl_lower, zl_array >= zl_upper)] *= 0
    matrix_LL[np.logical_or(zs_array <= zs_lower, zs_array >= zs_upper), :, :] *= 0
    matrix_LL[:, np.logical_or(sigma_array <= sg_lower, sigma_array >= sg_upper), :] *= 0
    matrix_LL[:, :, np.logical_or(zl_array <= zl_lower, zl_array >= zl_upper)] *= 0
    MASK_REGRx = [] if Rein_lim == 0 else Theta_E_LL < Rein_lim
    matrix_LL[MASK_REGRx] *= 0
    matrix_noLL[MASK_REGRx] *= 0
    _ , __  , ___, P_zs_LL   , P_zl_LL   , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH=1)
    _ , __  , ___, P_zs_noLL , P_zl_noLL , P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH=1)
    if PRINT:
        print('EXPECTED NUMBER OF LENSES AFTER PRIOR ON ZL, ZS, AND SIGMA:')
        if np.sum(matrix_noLL) > 1e4:
            print(f'   {title}: {np.sum(matrix_noLL):.1e} ({np.sum(matrix_LL):.1e})')
        else:
            print(f'   {title}: {np.sum(matrix_noLL):.0f} ({np.sum(matrix_LL):.0f})')
    if LENS_LIGHT:
        return matrix_LL, Theta_E_LL, prob_LL, P_zs_LL, P_zl_LL, P_sg_LL
    return matrix_noLL, Theta_E_noLL, prob_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL

def plot_survey_ALL_distributions(surveys, READ_FILES = 1, _PLOT_FOR_KEYNOTE = 0):
    set_plt_param(PLOT_FOR_KEYNOTE = _PLOT_FOR_KEYNOTE)
    M_array     = np.linspace(-13 , -25 , 25)
    sigma_array = np.linspace(100 , 400 , 31)
    zl_array    = np.arange(0.  , 2.5 , 0.1)
    zs_array    = np.arange(0.  , 5.4 , 0.2)
    min_SNR     = 20
    arc_mu_thr  = 3
    Phi_vel_dis = ls.Phi_vel_disp_Mason

    for title in surveys:
        survey_params = utils.read_survey_params(title, VERBOSE = 0)
        limit    = survey_params['limit']
        cut      = survey_params['cut']
        area     = survey_params['area']
        seeing   = survey_params['seeing']
        exp_time_sec = survey_params['exp_time_sec']
        zero_point_m = survey_params['zero_point_m']
        sky_bckgnd_m = survey_params['sky_bckgnd_m']
        photo_band   = survey_params['photo_band']
        try:
            if READ_FILES:
                matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
            else:
                raise ValueError
        except ValueError:
            print('FILE do NOT exist - RUNNING MODEL')
            matrix_noLL, Theta_E_noLL, prob_noLL = ls.calculate_num_lenses_and_prob(
                                                                        sigma_array, zl_array, zs_array, M_array, limit, area,
                                                                        seeing, min_SNR, exp_time_sec, sky_bckgnd_m, zero_point_m,
                                                                        photo_band = photo_band, mag_cut=cut, arc_mu_threshold = arc_mu_thr,
                                                                        Phi_vel_disp = Phi_vel_dis, LENS_LIGHT_FLAG = False, SIE_FLAG = True)
            matrix_LL, Theta_E_LL, prob_LL = ls.calculate_num_lenses_and_prob(
                                                                        sigma_array, zl_array, zs_array, M_array, limit, area,
                                                                        seeing, min_SNR, exp_time_sec, sky_bckgnd_m, zero_point_m,
                                                                        photo_band = photo_band, mag_cut=cut, arc_mu_threshold = arc_mu_thr,
                                                                        Phi_vel_disp = Phi_vel_dis, LENS_LIGHT_FLAG = True, SIE_FLAG = True)
            utils.save_pickled_files(title,  matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL)
        utils.print_summary_surveys([title])
        plot_ALL_distributions(title, zl_array, zs_array, sigma_array,
                                    Theta_E_LL, matrix_LL, Theta_E_noLL, matrix_noLL,
                                    PLOT_FOR_KEYNOTE = _PLOT_FOR_KEYNOTE, SMOOTH = 1, SAVE = 0)

def plot_z_sigma_distributions_double_lenses(title, zl_array, zs_array, sigma_array, matrix_LL, matrix_noLL,
                               PLOT_FOR_KEYNOTE = 0, SMOOTH = 1, SAVE = 0):
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.25, hspace=.2)
    Ngal_zl_sigma_noLL, Ngal_zl_sigma_noLL, Ngal_zl_zs1_noLL, Ngal_zl_zs2_noLL, Ngal_sigma_zs1_noLL, Ngal_sigma_zs2_noLL, Ngal_zs1_zs2_noLL, P_zs_noLL, P_zs2_noLL, P_zl_noLL, P_sg_noLL = rls.get_N_and_P_projections_double_lens(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH)
    Ngal_zl_sigma_LL, Ngal_zl_sigma_LL, Ngal_zl_zs1_LL, Ngal_zl_zs2_LL, Ngal_sigma_zs1_LL, Ngal_sigma_zs2_LL, Ngal_zs1_zs2_LL, P_zs_LL, P_zs2_LL, P_zl_LL, P_sg_LL = rls.get_N_and_P_projections_double_lens(matrix_LL, sigma_array, zl_array, zs_array, SMOOTH)
    ax[0].plot(zl_array, P_zl_noLL, c=col_A, ls='-')
    ax[0].plot(zl_array, P_zl_LL,   c=col_A, ls=':')
    ax[0].plot(zs_array, P_zs_noLL, c=col_B, ls='-' , label='No lens light')
    ax[0].plot(zs_array, P_zs_LL,   c=col_B, ls=':' , label='w/ lens light')
    ax[0].plot(zs_array, P_zs2_noLL,c=col_C, ls='-' )
    ax[0].plot(zs_array, P_zs2_LL,  c=col_C, ls=':' )
    ax[0].set_ylabel(r'$dP/dz$', fontsize=20)
    ax[0].set_xlabel(r'$z$', fontsize=20)
    ax[0].set_xlim((0,5.2))
    ax[1].plot(sigma_array, P_sg_noLL, c=col_A, ls = '-')
    ax[1].plot(sigma_array, P_sg_LL  , c=col_A, ls = ':')
    ax[1].set_ylabel(r'$dP/d\sigma$', fontsize=20)
    ax[1].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
    if SAVE:
        # folderpath = 'img/'+utils.remove_spaces_from_string(title)
        # if not os.path.exists(folderpath): os.makedirs(folderpath)
        fn_prefix = fn_prefix + 'DBL_'
        plt.savefig('img/'+fn_prefix+'corner_plts.png', dpi=200, bbox_inches='tight')
    plt.show()

def plot_z_sigma_distributions(fig, ax, title, zl_array, zs_array, sigma_array,
                               Theta_E_LL, matrix_LL, Theta_E_noLL, matrix_noLL,
                               PLOT_FOR_KEYNOTE = 0, CONTOUR = 1, LOG = 0, SMOOTH = 0, SAVE = 0,
                               LEGEND = 1, DOUBLE_LENS = 0):
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    if DOUBLE_LENS:
        Ngal_zl_sigma_noLL, Ngal_zl_sigma_noLL, Ngal_zl_zs1_noLL, Ngal_zl_zs2_noLL, Ngal_sigma_zs1_noLL, Ngal_sigma_zs2_noLL, Ngal_zs1_zs2_noLL, P_zs_noLL, P_zs2_noLL, P_zl_noLL, P_sg_noLL = rls.get_N_and_P_projections_double_lens(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH)
        Ngal_zl_sigma_LL, Ngal_zl_sigma_LL, Ngal_zl_zs1_LL, Ngal_zl_zs2_LL, Ngal_sigma_zs1_LL, Ngal_sigma_zs2_LL, Ngal_zs1_zs2_LL, P_zs_LL, P_zs2_LL, P_zl_LL, P_sg_LL = rls.get_N_and_P_projections_double_lens(matrix_LL, sigma_array, zl_array, zs_array, SMOOTH)
    else:
        Ngal_zl_sigma_noLL, Ngal_zs_sigma_noLL, Ngal_zs_zl_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH)
        Ngal_zl_sigma_LL, Ngal_zs_sigma_LL, Ngal_zs_zl_LL, P_zs_LL, P_zl_LL, P_sg_LL = ls.get_N_and_P_projections(matrix_LL, sigma_array, zl_array, zs_array, SMOOTH)
    _nbins_Re = np.arange(0  , 4  , 0.25)
    fig.suptitle(title, fontsize=25)
    if LOG:
        CONTOUR = 0
        ax[0,0].set_yscale('log')
        ax[0,1].set_yscale('log')
        ax[0,0].set_ylim((1e-3,2))
        ax[0,1].set_ylim((1e-3,2))
    if CONTOUR:
        _zs, _zl = np.meshgrid(zs_array, zl_array)
        levels   = np.asarray([0.01, 0.1, 0.5, 1, 2, 3, 5])*(np.power(10,np.ceil(np.log10(np.max(Ngal_zs_zl_LL))-1)))
        contours = ax[0,0].contour(_zl, _zs, Ngal_zs_zl_LL.T, levels, cmap=cmap_c, norm=colors.Normalize(vmin=np.min(Ngal_zs_zl_LL), vmax=np.max(Ngal_zs_zl_LL)), linestyles=':')
        ax[0,0].scatter(200,1.0, label='', alpha=0)
        ax[0,0].clabel(contours, inline=True, fontsize=8)
        levels   = np.asarray([0.01, 0.1, 0.5, 1, 2, 3, 5])*(np.power(10,np.ceil(np.log10(np.max(Ngal_zs_zl_noLL))-1)))
        contours = ax[0,0].contour(_zl, _zs,Ngal_zs_zl_noLL.T, levels, cmap=cmap_c, norm=colors.Normalize(vmin=np.min(Ngal_zs_zl_noLL), vmax=np.max(Ngal_zs_zl_noLL)), linestyles='-')
        ax[0,0].scatter(200,1.0, label='', alpha=0)
        ax[0,0].clabel(contours, inline=True, fontsize=8)
        ax[0,0].set_xlabel(r'$z_l', fontsize=20)
        ax[0,0].set_ylabel(r'$z_s$', fontsize=20)
        ax[0,0].set_xlim((0,3.2))
        ax[0,0].set_ylim((0,6.2))
    else:
        ax[0,0].plot(zl_array, P_zl_LL, c=col_A, ls=':')
        ax[0,0].plot(zs_array, P_zs_LL, c=col_B, ls=':' , label='w/ lens light')
        ax[0,0].plot(zl_array, P_zl_noLL, c=col_A, ls='-')
        ax[0,0].plot(zs_array, P_zs_noLL, c=col_B, ls='-', label='No lens light')
        if DOUBLE_LENS:
            ax[0,0].plot(zs_array, P_zs2_LL, c=col_C, ls=':' , label='w/ lens light')
            ax[0,0].plot(zs_array, P_zs2_noLL, c=col_C, ls='-', label='No lens light')
        ax[0,0].set_ylabel(r'$P$', fontsize=20)
        ax[0,0].set_xlabel(r'$z$', fontsize=20)
        ax[0,0].set_xlim((0,5.2))
    ax[1,1].plot(sigma_array, P_sg_LL, c=col_C, ls = ':')
    ax[1,1].plot(sigma_array, P_sg_noLL, c=col_C, ls = '-')
    ax[1,1].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)

    if DOUBLE_LENS:
        pass
        # ax[0,1].hist(np.ravel(Theta_E_LL[0]), weights=np.ravel(matrix_LL), bins=_nbins_Re, range=(0, 3), density=True, histtype='step', color=col_C, ls = ':')
        # ax[0,1].hist(np.ravel(Theta_E_noLL[0]), weights=np.ravel(matrix_noLL), bins=_nbins_Re, range=(0, 3), density=True, histtype='step', color=col_C, ls = '-')
        # ax[0,1].hist(np.ravel(Theta_E_LL[1]), weights=np.ravel(matrix_LL), bins=_nbins_Re, range=(0, 3), density=True, histtype='step', color=col_D, ls = ':')
        # ax[0,1].hist(np.ravel(Theta_E_noLL[1]), weights=np.ravel(matrix_noLL), bins=_nbins_Re, range=(0, 3), density=True, histtype='step', color=col_D, ls = '-')
    else:
        ax[0,1].hist(np.ravel(Theta_E_LL), weights=np.ravel(matrix_LL), bins=_nbins_Re, range=(0, 3), density=True, histtype='step', color=col_D, ls = ':')
        ax[0,1].hist(np.ravel(Theta_E_noLL), weights=np.ravel(matrix_noLL), bins=_nbins_Re, range=(0, 3), density=True, histtype='step', color=col_D, ls = '-')
    ax[0,1].set_xlabel(r'$\Theta_E$ [arcsec]', fontsize=20)

    _sigma, _zl = np.meshgrid(sigma_array, zl_array)
    levels   = np.asarray([0.01, 0.1, 0.5, 1, 2, 3, 5])*(np.power(10,np.ceil(np.log10(np.max(Ngal_zl_sigma_LL))-1)))
    contours = ax[1,0].contour(_sigma, _zl,Ngal_zl_sigma_LL.T, levels, cmap=cmap_c, norm=colors.Normalize(vmin=np.min(Ngal_zl_sigma_LL), vmax=np.max(Ngal_zl_sigma_LL)), linestyles=':')
    ax[1,0].scatter(200,1.0, label='', alpha=0)
    ax[1,0].clabel(contours, inline=True, fontsize=8)
    ax[1,0].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
    ax[1,0].set_ylabel(r'$z_l$', fontsize=20)
    levels   = np.asarray([0.01, 0.1, 0.5, 1, 2, 3, 5])*(np.power(10,np.ceil(np.log10(np.max(Ngal_zl_sigma_noLL))-1)))
    contours = ax[1,0].contour(_sigma, _zl,Ngal_zl_sigma_noLL.T, levels, cmap=cmap_c, norm=colors.Normalize(vmin=np.min(Ngal_zl_sigma_noLL), vmax=np.max(Ngal_zl_sigma_noLL)), linestyles='-')
    ax[1,0].scatter(200,1.0, label='', alpha=0)
    ax[1,0].clabel(contours, inline=True, fontsize=8)
    ax[1,0].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
    ax[1,0].set_ylabel(r'$z_l$', fontsize=20)
    if LEGEND:
        if(np.sum(matrix_noLL))>10_000:
            ax[1,0].legend([f'#Lenses w/ LL: {np.sum(matrix_LL):.1e}', f'#Lenses no LL: {np.sum(matrix_noLL):.1e}'], fontsize=20)
        else:
            ax[1,0].legend([f'#Lenses w/ LL: {np.sum(matrix_LL):.0f}', f'#Lenses no LL: {np.sum(matrix_noLL):.0f}'], fontsize=20)
    ax[1,1].set_xlim((100,400))
    ax[1,0].set_xlim((100,400))
    ax[1,0].set_ylim((0,2.5))
    plt.tight_layout()
    if SAVE:
        # folderpath = 'img/'+utils.remove_spaces_from_string(title)
        # if not os.path.exists(folderpath): os.makedirs(folderpath)
        if DOUBLE_LENS: fn_prefix = fn_prefix + 'DBL_'
        plt.savefig('img/'+fn_prefix+'corner_plts.png', dpi=200, bbox_inches='tight')
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
        if LENS_LIGHT:
            Ngal_zl_sigma_noLL, Ngal_zs_sigma_noLL, Ngal_zs_zl_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL = ls.get_N_and_P_projections(matrix_LL, sigma_array, zl_array, zs_array, SMOOTH)
        else:
            Ngal_zl_sigma_noLL, Ngal_zs_sigma_noLL, Ngal_zs_zl_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH)
        ax[0].plot(zl_array, P_zl_noLL, c=col_A, ls=lstyle, label=f'{label}')
        ax[0].plot(zs_array, P_zs_noLL, c=col_B, ls=lstyle)
        ax[0].set_ylabel(r'$dP/dz$', fontsize=20)
        ax[0].set_xlabel(r'$z$', fontsize=20)
        ax[0].set_xlim((0,5.2))
        ax[1].plot(sigma_array, P_sg_noLL, c=col_C, ls = lstyle)
        ax[1].set_ylabel(r'$dP/d\sigma$', fontsize=20)
        ax[1].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
        _nbins_Re = np.arange(0  , 4  , 0.25)
        ax[2].hist(np.ravel(Theta_E_noLL), weights=np.ravel(matrix_noLL), bins=_nbins_Re, range=(0, 3),
                   density=True, histtype='step', color=col_D, ls = lstyle, label=f'# Lenses: {np.sum(matrix_noLL):.1e}')
        ax[2].set_ylabel(r'$dP/d\Theta_E$', fontsize=20)
        ax[2].set_xlabel(r'$\Theta_E$ [arcsec]', fontsize=20)
    ax[0].legend(fontsize=14, frameon=False)
    ax[2].legend(fontsize=14, frameon=False)
    plt.tight_layout()
    if (SAVE):
        folderpath = 'img/'+utils.remove_spaces_from_string(title)
        if not os.path.exists(folderpath): os.makedirs(folderpath)
        plt.savefig(folderpath+'/VDF_effect_corner_plts.png', dpi=200, bbox_inches='tight')
    plt.show()


def compare_ALL_distributions_surveys(surveys_selection, sigma_array, zl_array, zs_array, LENS_LIGHT = 1, PLOT_FOR_KEYNOTE = 0, SMOOTH = 1):
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    _col_  = iter(cmap_c(np.linspace(0, 1, len(surveys_selection)+1)))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.235, hspace=.2)
    for title in surveys_selection:
        survey_params = utils.read_survey_params(title, VERBOSE = 0)
        limit    = survey_params['limit']
        cut      = survey_params['cut']
        area     = survey_params['area']
        seeing   = survey_params['seeing']
        exp_time_sec = survey_params['exp_time_sec']
        pixel_arcsec = survey_params['pixel_arcsec']
        zero_point_m = survey_params['zero_point_m']
        sky_bckgnd_m = survey_params['sky_bckgnd_m']
        photo_band   = survey_params['photo_band']
        matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
        matrix  = matrix_LL  if LENS_LIGHT else matrix_noLL
        Theta_E = Theta_E_LL if LENS_LIGHT else Theta_E_noLL
        __col__ = next(_col_)
        plot_z_distribution_in_ax(ax[0], title, __col__, zl_array, zs_array, sigma_array, matrix, SMOOTH = SMOOTH)
        plot_s_distribution_in_ax(ax[1], title, __col__, zl_array, zs_array, sigma_array, matrix, SMOOTH = SMOOTH)
        plot_R_distribution_in_ax(ax[2], title, __col__, np.arange(0  , 4  , 0.25), matrix, Theta_E, SMOOTH = SMOOTH)

    ax[0].set_xlabel(r'$z$', fontsize=20)
    ax[0].set_ylabel(r'$dP/dz$', fontsize=20)
    ax[1].set_xlabel(r'$\sigma$', fontsize=20)
    ax[1].set_ylabel(r'$dP/d\sigma$', fontsize=20)
    ax[2].set_xlabel(r'$\Theta_E$ [arcsec]', fontsize=20)
    ax[2].set_ylabel(r'$dP/d\Theta_E$', fontsize=20)
    ax[0].legend(fontsize=12)
    plt.show()


def single_compare_z_distributions_surveys(ax, title, color,
                                        zl_array, zs_array, sigma_array, matrix_LL, matrix_noLL,
                                        PLOT_FOR_KEYNOTE = 0, SMOOTH = 1, PLOT_ALL = 0):
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    _n, __n, ___n, P_zs_noLL, P_zl_noLL, P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH)
    _ , __  , ___, P_zs_LL  , P_zl_LL  , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH)
    if PLOT_ALL:
        ax.plot(zl_array, P_zl_noLL, c=color, ls='-' , label=title)
        ax.plot(zs_array, P_zs_noLL, c=color, ls=':' )
        ax.set_xlabel(r'$z$', fontsize=20)
        ax.set_ylabel(r'$dP/dz$', fontsize=20)
    else:
        ax[0].plot(zl_array, P_zl_noLL, c=color, ls='-' , label=title)
        ax[0].plot(zs_array, P_zs_noLL, c=color, ls=':')
        ax[1].plot(zl_array, P_zl_LL, c=color, ls='--', label=title)
        ax[1].plot(zs_array, P_zs_LL, c=color, ls=':')
        ax[0].set_xlabel(r'$z$', fontsize=20)
        ax[1].set_xlabel(r'$z$', fontsize=20)
        ax[0].set_ylabel(r'$dP/dz$', fontsize=20)

def single_compare_sigma_distributions_surveys(ax, title, color,
                                        zl_array, zs_array, sigma_array, matrix_LL, matrix_noLL,
                                        PLOT_FOR_KEYNOTE = 0, SMOOTH = 1, PLOT_ALL = 0):
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    _n, __n, ___n, P_zs_noLL, P_zl_noLL, P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH)
    _ , __  , ___, P_zs_LL  , P_zl_LL  , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH)
    if PLOT_ALL:
        ax.plot(sigma_array, P_sg_noLL, c=color, ls='-' , label=title)
        ax.set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
        ax.set_ylabel(r'$dP/d\sigma$'   , fontsize=20)
    else:
        ax[0].plot(sigma_array, P_sg_noLL, c=color, ls = '-', label=title)
        ax[1].plot(sigma_array, P_sg_LL  , c=color, ls = ':')
        ax[0].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
        ax[1].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
        ax[0].set_ylabel(r'$dP/d\sigma$', fontsize=20)


def single_compare_EinRing_distributions_surveys(ax, title, color,
                                        zl_array, zs_array, sigma_array, matrix_LL, matrix_noLL,
                                        PLOT_FOR_KEYNOTE = 0, SMOOTH = 1, PLOT_ALL = 0):
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    _n, __n, ___n, P_zs_noLL, P_zl_noLL, P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH)
    _ , __  , ___, P_zs_LL  , P_zl_LL  , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH)
    if PLOT_ALL:
        ax.hist(np.ravel(Theta_E_noLL), weights=np.ravel(matrix_noLL), bins=_nbins_Re, range=(0, 3), density=True, histtype='step', color=col_D, ls = '-')
        ax.set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
        ax.set_ylabel(r'$dP/d\sigma$'   , fontsize=20)
    else:
        ax[0].plot(sigma_array, P_sg_noLL, c=color, ls = '-', label=title)
        ax[1].plot(sigma_array, P_sg_LL  , c=color, ls = ':')
        ax[0].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
        ax[1].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
        ax[0].set_ylabel(r'$dP/d\sigma$', fontsize=20)

def compare_z_distributions_surveys(surveys_selection, sigma_array, zl_array, zs_array, PLOT_FOR_KEYNOTE = 0):
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    _col_  = iter(cmap_c(np.linspace(0, 1, len(surveys_selection)+1)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=False, sharey=True)
    plt.subplots_adjust(wspace=.02, hspace=.2)
    for title in surveys_selection:
        survey_params = utils.read_survey_params(title, VERBOSE = 0)
        limit    = survey_params['limit']
        cut      = survey_params['cut']
        area     = survey_params['area']
        seeing   = survey_params['seeing']
        exp_time_sec = survey_params['exp_time_sec']
        pixel_arcsec = survey_params['pixel_arcsec']
        zero_point_m = survey_params['zero_point_m']
        sky_bckgnd_m = survey_params['sky_bckgnd_m']
        photo_band   = survey_params['photo_band']
        matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
        single_compare_z_distributions_surveys(ax, title, next(_col_),
                                                        zl_array, zs_array, sigma_array, matrix_LL, matrix_noLL,
                                                        PLOT_FOR_KEYNOTE = PLOT_FOR_KEYNOTE, SMOOTH = 1)
    ax[0].set_xlim((0,5.2))
    ax[1].set_xlim((0,5.2))
    ax[0].legend()
    plt.show()

def compare_sigma_distributions_surveys(surveys_selection, sigma_array, zl_array, zs_array, PLOT_FOR_KEYNOTE = 0):
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    _col_  = iter(cmap_c(np.linspace(0, 1, len(surveys_selection)+1)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=False, sharey=True)
    plt.subplots_adjust(wspace=.02, hspace=.2)
    for title in surveys_selection:
        survey_params = utils.read_survey_params(title, VERBOSE = 0)
        limit    = survey_params['limit']
        cut      = survey_params['cut']
        area     = survey_params['area']
        seeing   = survey_params['seeing']
        exp_time_sec = survey_params['exp_time_sec']
        pixel_arcsec = survey_params['pixel_arcsec']
        zero_point_m = survey_params['zero_point_m']
        sky_bckgnd_m = survey_params['sky_bckgnd_m']
        photo_band   = survey_params['photo_band']
        matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
        single_compare_sigma_distributions_surveys(ax, title, next(_col_),
                                                        zl_array, zs_array, sigma_array, matrix_LL, matrix_noLL,
                                                        PLOT_FOR_KEYNOTE = PLOT_FOR_KEYNOTE, SMOOTH = 1)
    ax[0].set_xlim((100,400))
    ax[1].set_xlim((100,400))
    ax[0].legend()
    plt.show()

def plot_angular_separation(survey_title, _theta_arcsec, omega, cmap_c = cm.cool,
                            frac_lens = 0.1, PLOT_FOR_KEYNOTE = 0,
                            A_w = 0.4, beta = 0.6):
    set_plt_param(PLOT_FOR_KEYNOTE = PLOT_FOR_KEYNOTE)

    survey_params = utils.read_survey_params(survey_title, VERBOSE = 0)
    limit    = survey_params['limit']
    cut      = survey_params['cut']
    area     = survey_params['area']
    seeing   = survey_params['seeing']
    exp_time_sec = survey_params['exp_time_sec']
    pixel_arcsec = survey_params['pixel_arcsec']
    zero_point_m = survey_params['zero_point_m']
    sky_bckgnd_m = survey_params['sky_bckgnd_m']
    photo_band   = survey_params['photo_band']
    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(survey_title)

    intsteps = 11
    Theta_pos_noLL, Theta_pos_LL = np.zeros(Theta_E_noLL.shape), np.zeros(Theta_E_LL.shape)
    for _m in np.linspace(1,2, intsteps):
        Theta_pos_noLL = Theta_pos_noLL + Theta_E_noLL*_m
        Theta_pos_LL   = Theta_pos_LL   + Theta_E_LL*_m
    Theta_pos_noLL, Theta_pos_LL = Theta_pos_noLL/intsteps, Theta_pos_LL/intsteps

    omega[_theta_arcsec<1]  = 0
    omega[_theta_arcsec>10] = 0
    dPdt = 2 * np.pi * _theta_arcsec * np.diff(_theta_arcsec)[0] * (omega + 1)
    cPdt = np.cumsum(dPdt)/np.sum(dPdt)

    color = 'w' if PLOT_FOR_KEYNOTE else 'k'
    c1 = 'c' if PLOT_FOR_KEYNOTE else 'k'
    c2 = 'r' if PLOT_FOR_KEYNOTE else 'k'

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.15, hspace=.2)
    ax.set_ylabel(r'$P(<\theta$)', fontsize=20)
    ax.set_xlabel(r'$\theta$ [arcsec]', fontsize=20)
    ax.set_xlim((0,10))
    ax.set_ylim((0,1.1))
    counts, bins, _ = ax.hist(np.ravel(Theta_pos_noLL), weights=np.ravel(matrix_noLL), bins=_theta_arcsec, range=(0, 12), density=True, histtype='step', alpha=0,
                              color=color, ls = '--', cumulative=True)
    counts = np.append(counts, 1)
    ax.plot(_theta_arcsec, counts, c=c1, ls='--', label='Lenses only')
    ax.plot(_theta_arcsec, cPdt, c=c2, ls=':', label='1 halo term HOD')
    w_avg = np.array(counts) * frac_lens + cPdt * (1 - frac_lens)
    ax.plot(_theta_arcsec, w_avg, c=color, lw=2.5, label='Observed ACF')
    # plt.legend(fontsize=15, loc='lower right')
    plt.tight_layout()
    plt.show()

    if PLOT_FOR_KEYNOTE:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), sharex=False, sharey=False)
        plt.subplots_adjust(wspace=.15, hspace=.2)
        ax.set_ylabel(r'$P(<\theta$)', fontsize=20)
        ax.set_xlabel(r'$\theta$ [arcsec]', fontsize=20)
        ax.set_xlim((0,10))
        ax.set_ylim((0,1.1))
        counts, bins, _ = ax.hist(np.ravel(Theta_pos_noLL), weights=np.ravel(matrix_noLL), bins=_theta_arcsec, range=(0, 12), density=True, histtype='step', alpha=0,
                                color=color, ls = '--', cumulative=True)
        counts = np.append(counts, 1)
        ax.plot(_theta_arcsec, counts, c=c1, ls='--', label='Lenses only')
        # ax.plot(_theta_arcsec, cPdt, c=c2, ls=':', label='1 halo term HOD')
        # w_avg = np.array(counts) * frac_lens + cPdt * (1 - frac_lens)
        # ax.plot(_theta_arcsec, w_avg, c=color, lw=2.5, label='Observed ACF')
        # plt.legend(fontsize=15, loc='lower right')
        plt.tight_layout()
        plt.show()
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), sharex=False, sharey=False)
        plt.subplots_adjust(wspace=.15, hspace=.2)
        ax.set_ylabel(r'$P(<\theta$)', fontsize=20)
        ax.set_xlabel(r'$\theta$ [arcsec]', fontsize=20)
        ax.set_xlim((0,10))
        ax.set_ylim((0,1.1))
        counts, bins, _ = ax.hist(np.ravel(Theta_pos_noLL), weights=np.ravel(matrix_noLL), bins=_theta_arcsec, range=(0, 12), density=True, histtype='step', alpha=0,
                                color=color, ls = '--', cumulative=True)
        counts = np.append(counts, 1)
        ax.plot(_theta_arcsec, counts, c=c1, ls='--', label='Lenses only')
        ax.plot(_theta_arcsec, cPdt, c=c2, ls=':', label='1 halo term HOD')
        # w_avg = np.array(counts) * frac_lens + cPdt * (1 - frac_lens)
        # ax.plot(_theta_arcsec, w_avg, c=color, lw=2.5, label='Observed ACF')
        # plt.legend(fontsize=15, loc='lower right')
        plt.tight_layout()
        plt.show()

def compare_COSMOS_HST_Faure(zl_array, zs_array, sigma_array, M_array_UV, mag_cut,
                            ONLY_FULL_SAMPLE = 1, LENS_LIGHT = 1, __MAG_OVER_ARCSEC_SQ__ = 0,
                            PLOT_FOR_KEYNOTE = 0, SMOOTH = 1, SAVE = 0):
    ### FAURE DATA #################################################################################
    ### from FAURE+(2008) --- selection of high grade lenses from COSMOS-HST
    FAURE_title=['Name'       ,'zl' , 'zs', 'sig', 'Rein', 'mag_814W_len', 'mag_814W_src_arcsec_sq']
    FAURE_data =[['0012 + 2015', 0.41, 0.95, 215.3, 0.67 , 19.28         , 21.6],
                ['0018 + 3845', 0.71, 1.93, 303.1, 1.32  , 23.60         , 22.7],
                ['0038 + 4133', 0.89, 2.70, 225.3, 0.73  , 20.39         , 20.5],
                ['0047 + 5023', 0.85, 2.51, 313.0, 1.41  , 20.65         , 22.8],
                ['0049 + 5128', 0.33, 0.74, 380.0, 2.09  , 19.61         , 23.3],
                ['0050 + 4901', 1.01, 3.34, 342.3, 1.69  , 21.72         , 22.7],
                ['0056 + 1226', 0.44, 1.03, 337.4, 1.64  , 18.70         , 23.3],
                ['0124 + 5121', 0.84, 2.47, 245.0, 0.86  , 22.43         , 23.2],
                ['0211 + 1139', 0.90, 2.76, 466.3, 3.14  , 21.09         , 21.6],
                ['0216 + 2955', 0.67, 1.77, 348.5, 1.75  , 19.98         , 22.0],
                ['0227 + 0451', 0.89, 2.70, 428.3, 2.64  , 21.94         , 22.3],
                ['5857 + 5949', 0.39, 0.89, 398.1, 2.28  , 20.05         , 21.9],
                ['5914 + 1219', 1.05, 3.57, 338.6, 1.65  , 23.25         , 22.8],
                ['5921 + 0638', 0.45, 1.06, 221.0, 0.70  , 20.34         , 20.6],
                ['5941 + 3628', 0.90, 2.76, 285.0, 1.17  , 20.91         , 22.8],
                ['5947 + 4752', 0.28, 0.61, 370.0, 1.97  , 19.83         , 22.8]]
    FAURE_data  = np.asarray(FAURE_data)
    FAURE_names = FAURE_data[:,0]
    FAURE_zl,   FAURE_zs    = np.asarray(FAURE_data[:,1], dtype='float'), np.asarray(FAURE_data[:,2], dtype='float')
    FAURE_sig,  FAURE_Rein  = np.asarray(FAURE_data[:,3], dtype='float'), np.asarray(FAURE_data[:,4], dtype='float')
    FAURE_m_Ib, FAURE_m_src = np.asarray(FAURE_data[:,5], dtype='float'), np.asarray(FAURE_data[:,6], dtype='float')

    FAURE_A_data  = pd.read_csv('../galess/data/LENS_SEARCHES/FAURE_2008/FAURE_A.csv')
    FAURE_A_names = FAURE_A_data['Cosmos Name'].to_numpy()
    FAURE_A_zl    = FAURE_A_data['z_l'].to_numpy()
    FAURE_A_Rarc  = FAURE_A_data['R_arc'].to_numpy()
    FAURE_A_Reff  = FAURE_A_data['Reff'].to_numpy()
    FAURE_A_m_Ib  = FAURE_A_data['mag_814W'].to_numpy()
    FAURE_A_ell   = FAURE_A_data['ell'].to_numpy()
    FAURE_A_m_src = FAURE_A_data['mag_814W_src'].to_numpy()

    FAURE_B_data  = pd.read_csv('../galess/data/LENS_SEARCHES/FAURE_2008/FAURE_B.csv')
    FAURE_B_names = FAURE_B_data['Cosmos Name'].to_numpy()
    FAURE_B_zl    = FAURE_B_data['z_l'].to_numpy()
    FAURE_B_Rarc  = FAURE_B_data['R_arc'].to_numpy()
    FAURE_B_Reff  = FAURE_B_data['Reff'].to_numpy()
    FAURE_B_m_Ib  = FAURE_B_data['mag_814W'].to_numpy()
    FAURE_B_ell   = FAURE_B_data['ell'].to_numpy()
    FAURE_B_m_src = FAURE_B_data['mag_814W_src'].to_numpy()
    ### PLOT DATA #################################################################################
    set_plt_param(PLOT_FOR_KEYNOTE)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.23, hspace=.2)
    ccc = 'w' if PLOT_FOR_KEYNOTE else 'k'
    cc2 = 'r'
    _nbins_zl = np.arange(0.0, 1.6, 0.2 )
    _nbins_zs = np.arange(0.0, 5  , 0.5 )
    _nbins_sg = np.linspace(100 , 400 , 31)
    _nbins_Re = np.arange(0  , 4  , 0.25)
    m_obs = np.linspace(15, 30, 31)
    line_thick = 3
    if PLOT_FOR_KEYNOTE:
        ER_col1, ER_col2  = 'darkorange', 'lime'
        _ALPHA_ = 1
    else:
        ER_col1, ER_col2  = 'forestgreen', 'orange'
        _ALPHA_ = 1
    ### FULL SAMPLE ###
    _nbins_zl = np.histogram_bin_edges(np.append(FAURE_A_zl,  FAURE_B_zl), bins='fd', range=(0,1.5))
    F_zl_hist, F_zl_bins, _ = ax[0].hist(np.append(FAURE_A_zl,  FAURE_B_zl), bins=_nbins_zl, density=True, histtype='step',
                                        lw=line_thick, color=ER_col1, alpha = _ALPHA_,
                                        label=f'Faure et al. 2008\n Full Sample ({len(np.append(FAURE_A_zl,  FAURE_B_zl))})')
    _nbins_Re = np.histogram_bin_edges(np.append(FAURE_A_Rarc/1.5,FAURE_B_Rarc/1.5), bins='fd', range=(0,4))
    F_Ra_hist, F_Ra_bins, _ = ax[1].hist(np.append(FAURE_A_Rarc/1.5,FAURE_B_Rarc/1.5), bins=_nbins_Re, density=True, histtype='step',
                                        lw=line_thick, color=ER_col1, alpha = _ALPHA_,
                                        label=f'Faure et al. 2008\n Full Sample ({len(np.append(FAURE_A_zl,  FAURE_B_zl))})')
    # if DENSITY: _nbins_zl = np.histogram_bin_edges(FAURE_A_zl, bins='fd', range=(0,1.5))
    F_mI_hist, F_mI_bins, _ = ax[2].hist(np.append(FAURE_A_m_Ib, FAURE_B_m_Ib), bins=m_obs, density=True, histtype='step',
                                        lw=line_thick, color=ER_col1, alpha = _ALPHA_,
                                        label=f'Faure et al. 2008\n Full Sample ({len(np.append(FAURE_A_zl,  FAURE_B_zl))})')
    ### BEST SAMPLE ###
    _nbins_zl = np.histogram_bin_edges(FAURE_A_zl, bins='fd', range=(0,1.5))
    F_zl_hist, F_zl_bins, _ = ax[0].hist(FAURE_A_zl, bins=_nbins_zl, density=True, histtype='step', lw=line_thick,
                                        color=ER_col2, alpha = _ALPHA_, label=f'Faure et al. 2008\n Best Sample ({len(FAURE_A_zl)})')
    _nbins_Re = np.histogram_bin_edges(FAURE_A_Rarc/1.5, bins='fd', range=(0,4))
    F_Ra_hist, F_Ra_bins, _ = ax[1].hist(FAURE_A_Rarc/1.5, bins=_nbins_Re, density=True, histtype='step', lw=line_thick,
                                        color=ER_col2, alpha = _ALPHA_, label=f'Faure et al. 2008\n Best Sample ({len(FAURE_A_zl)})')
    # if DENSITY: _nbins_zl = np.histogram_bin_edges(FAURE_A_zl, bins='fd', range=(0,1.5))
    F_mI_hist, F_mI_bins, _ = ax[2].hist(FAURE_A_m_Ib, bins=m_obs, density=True, histtype='step', lw=line_thick,
                                        color=ER_col2, alpha = _ALPHA_, label=f'Faure et al. 2008\n Best Sample ({len(FAURE_A_zl)})')
    title = 'COSMOS HST i band'
    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
    zl_lower, zl_upper = 0.2, 1
    zs_lower, zs_upper = 0.0, 99
    sg_lower, sg_upper = 160, 1000
    matrix_noLL[np.logical_or(zs_array <= zs_lower, zs_array >= zs_upper), :, :] *= 0
    matrix_noLL[:, np.logical_or(sigma_array <= sg_lower, sigma_array >= sg_upper), :] *= 0
    matrix_noLL[:, :, np.logical_or(zl_array <= zl_lower, zl_array >= zl_upper)] *= 0
    matrix_LL[np.logical_or(zs_array <= zs_lower, zs_array >= zs_upper), :, :] *= 0
    matrix_LL[:, np.logical_or(sigma_array <= sg_lower, sigma_array >= sg_upper), :] *= 0
    matrix_LL[:, :, np.logical_or(zl_array <= zl_lower, zl_array >= zl_upper)] *= 0
    print('EXPECTED NUMBER OF LENSES AFTER PRIOR ON ZL, ZS, AND SIGMA:')
    print(f'   COSMOS HST i band Mason VDF: {np.sum(matrix_noLL):.0f} ({np.sum(matrix_LL):.0f})')
    _ , __  , ___, P_zs_LL  , P_zl_LL  , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL, sigma_array, zl_array, zs_array, SMOOTH=SMOOTH)
    _ , __  , ___, P_zs_noLL  , P_zl_noLL  , P_sg_noLL   = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH=SMOOTH)
    if LENS_LIGHT:
        matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_LL, Theta_E_LL, prob_LL, P_zs_LL, P_zl_LL, P_sg_LL
    else:
        matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_noLL, Theta_E_noLL, prob_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL
    ax[0].plot(zl_array, P_zl, c=ccc, ls='-', label='COSMOS HST\n i band\n Mason VDF')
    ax[0].plot(zs_array, P_zs, c=ccc, ls='--')
    ax[0].set_xlim((0,4.2))
    ax[0].set_xlabel(r'$z$', fontsize=20)
    ax[0].set_ylabel(r'$dP/dz$', fontsize=20)
    T_hist_Mason, T_bins_Mason, _ = ax[1].hist(np.ravel(Theta_E), weights=np.ravel(matrix),
                                            bins=np.arange(0, 4, 0.2), range=(0, 3), density=True,
                                            histtype='step', color=ccc, ls = '-', label=title)
    ax[1].set_xlabel(r'$\Theta_E$ ["]', fontsize=20)
    ax[1].set_ylabel(r'$dP/d\Theta_E$', fontsize=20)
    ax[1].set_xlim((0,4))
    m_lens = ls.get_len_magnitude_distr(m_obs, zl_array, sigma_array, matrix)
    norm = integrate.simps(m_lens, m_obs)
    ax[2].plot(m_obs, m_lens/norm, color=ccc, label=title)
    ax[2].set_xlabel(r'$m_\text{I814W}^\text{len}$ [mag]', fontsize=20)
    ax[2].set_ylabel(r'$dP/dm$', fontsize=20)
    ax[2].set_xlim((15,25))
    ax[2].set_ylim((0,0.4))
    pval_zl_Mason = kstest(F_zl_hist, P_zl)[1]
    pval_Ra_Mason = kstest(F_Ra_hist, T_hist_Mason)[1]
    pval_mI_Mason = kstest(F_mI_hist, m_lens/norm)[1]
    title = 'COSMOS HST i band VDFGeng'
    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
    matrix_noLL[np.logical_or(zs_array <= zs_lower, zs_array >= zs_upper), :, :] *= 0
    matrix_noLL[:, np.logical_or(sigma_array <= sg_lower, sigma_array >= sg_upper), :] *= 0
    matrix_noLL[:, :, np.logical_or(zl_array <= zl_lower, zl_array >= zl_upper)] *= 0
    matrix_LL[np.logical_or(zs_array <= zs_lower, zs_array >= zs_upper), :, :] *= 0
    matrix_LL[:, np.logical_or(sigma_array <= sg_lower, sigma_array >= sg_upper), :] *= 0
    matrix_LL[:, :, np.logical_or(zl_array <= zl_lower, zl_array >= zl_upper)] *= 0
    _ , __  , ___, P_zs_LL  , P_zl_LL  , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL, sigma_array, zl_array, zs_array, SMOOTH=SMOOTH)
    _ , __  , ___, P_zs_noLL  , P_zl_noLL  , P_sg_noLL   = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH=SMOOTH)
    if LENS_LIGHT:
        matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_LL, Theta_E_LL, prob_LL, P_zs_LL, P_zl_LL, P_sg_LL
    else:
        matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_noLL, Theta_E_noLL, prob_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL
    print(f'   COSMOS HST i band Geng VDF: {np.sum(matrix_noLL):.0f} ({np.sum(matrix_LL):.0f})')
    ax[0].plot(zl_array, P_zl, c=cc2, ls='-', label='COSMOS HST\n i band\n Geng VDF')
    ax[0].plot(zs_array, P_zs, c=cc2, ls='--')
    ax[1].hist(np.ravel(Theta_E), weights=np.ravel(matrix), bins = np.arange(0, 4, 0.2),
                range=(0, 3), density=True, histtype='step', color=cc2, ls = '-')
    m_lens = ls.get_len_magnitude_distr(m_obs, zl_array, sigma_array, matrix)
    norm = integrate.simps(m_lens, m_obs)
    ax[2].plot(m_obs, m_lens/norm, color=cc2, label=title)
    ax[0].legend(fontsize=13)
    if (SAVE):
        # folderpath = 'img/'+utils.remove_spaces_from_string(title)
        # if not os.path.exists(folderpath): os.makedirs(folderpath)
        plt.savefig('img/COSMOS_HST_Faure.png', dpi=200, bbox_inches='tight')
    plt.show()



def compare_JACOBS_CNN_DES(zl_array, zs_array, sigma_array,
                        LENS_LIGHT = 1, PLOT_FOR_KEYNOTE = 0, SAVE = 0, DENSITY = 1, MU5 = 0,
                        legend_size = 18):
    JAC_DES_data       = pd.read_csv('../galess/data/LENS_SEARCHES/Jacobs_CNN/JACOBS_2019_DES_CNN.tsv', sep=';')
    JAC_DES_data_zl    = JAC_DES_data['z'].to_numpy()
    JAC_DES_data_imag  = JAC_DES_data['imag'].to_numpy()

    AGEL_data = pd.read_csv('../galess/data/LENS_SEARCHES/AGEL/AGEL.csv')
    AGEL_zl   = AGEL_data['z_l'].to_numpy()
    AGEL_zs   = AGEL_data['z_s'].to_numpy()
    AGEL_zl, AGEL_zs = AGEL_zl[AGEL_zl<100], AGEL_zs[AGEL_zs<100]
    ### PLOT DATA #################################################################################
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    ccc = 'w' if PLOT_FOR_KEYNOTE else 'k'
    cc2 = 'r'
    if PLOT_FOR_KEYNOTE:
        ER_col1, ER_col2, _ALPHA_  = 'darkorange', 'lime', 1
    else:
        ER_col1, ER_col2, _ALPHA_  = 'forestgreen', 'orange', 1
    title = 'DES i band'
    if MU5: title = title + '_muarc5'

    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
    MASK_REGR2 = Theta_E_LL < 2 if MU5 else []
    matrix_LL[MASK_REGR2] *= 0
    matrix_noLL[MASK_REGR2] *= 0
    print('EXPECTED NUMBER OF LENSES AFTER PRIOR ON ZL, ZS, AND SIGMA:')
    print(f'   DES i band Mason VDF: {np.sum(matrix_noLL):.0f} ({np.sum(matrix_LL):.0f})')
    _ , __  , ___, P_zs_LL   , P_zl_LL   , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH=1)
    _ , __  , ___, P_zs_noLL , P_zl_noLL , P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH=1)
    if LENS_LIGHT:
        matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_LL, Theta_E_LL, prob_LL, P_zs_LL, P_zl_LL, P_sg_LL
    else:
        matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_noLL, Theta_E_noLL, prob_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL
    N_LENSES = 1 if DENSITY else np.sum(matrix)*np.abs(zl_array[1]-zl_array[0])
    m_obs = np.linspace(15, 30, 31)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.23, hspace=.2)
    line_thick = 3
    _nbins_zl = np.arange(0.0, 2.2, 0.1 )
    ### SAMPLE ###
    _nbins_zl = np.histogram_bin_edges(JAC_DES_data_zl, bins='fd', range=(0,1.5))
    ax[0].hist( JAC_DES_data_zl, bins=_nbins_zl, density=DENSITY , histtype='step' , lw = line_thick, color=ER_col1, alpha = _ALPHA_,
                label=f'Jacobs et al. 2019\n Phot. sample ({len(JAC_DES_data_zl)})')
    _nbins_zl = np.histogram_bin_edges(AGEL_zl, bins='fd', range=(0,1.5))
    ax[0].hist( AGEL_zl, bins=_nbins_zl, density=DENSITY , histtype='step' , lw = line_thick, color=ER_col2, alpha = _ALPHA_,
                label=f'Tran et al. 2022\n Spec. sample (68)')
    _nbins_zs = np.histogram_bin_edges(AGEL_zs, bins='fd', range=(0.5,3))
    ax[0].hist( AGEL_zs, bins=_nbins_zs, density=DENSITY , histtype='step' , lw = line_thick, color=ER_col2, alpha = _ALPHA_, ls='--')
    # _nbins_zl = np.histogram_bin_edges(JAC_DES_data_zl, bins='fd', range=(0,2.2))
    ax[1].hist( JAC_DES_data_imag, bins=m_obs  , density=DENSITY, histtype='step', lw = line_thick, color=ER_col1, alpha = _ALPHA_)
    ax[0].plot(zl_array, P_zl* N_LENSES, c=ccc, ls='-', label='DES i band\n Mason VDF')
    ax[0].plot(zs_array, P_zs* N_LENSES, c=ccc, ls='--')
    ax[0].set_xlabel(r'$z$', fontsize=20)
    ax[0].set_ylabel(r'$dN/dz$', fontsize=20)
    if DENSITY:
        ax[0].set_ylim((0,3.3))
        ax[0].set_ylabel(r'$dP/dz$', fontsize=20)
    ax[0].set_xlim((0,3.6))
    m_lens = ls.get_len_magnitude_distr(m_obs, zl_array, sigma_array, matrix, obs_band = 'sdss_i0')
    norm = integrate.simps(m_lens, m_obs)
    ax[1].plot(m_obs, m_lens/norm, color=ccc)
    ax[1].set_xlabel(r'$m_\text{I814W}^\text{len}$ [mag]', fontsize=20)
    ax[1].set_ylim((0,0.5))
    ax[1].set_ylabel(r'$dP/dm$', fontsize=20)
    ax[1].set_xlim((15,25))
    title = 'DES i band Geng'
    if MU5: title = title + '_muarc5'
    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
    MASK_REGR2 = Theta_E_LL < 2 if MU5 else Theta_E_LL < 0
    matrix_LL[MASK_REGR2] *= 0
    matrix_noLL[MASK_REGR2] *= 0
    print(f'   DES i band Geng VDF: {np.sum(matrix_noLL):.0f} ({np.sum(matrix_LL):.0f})')
    _ , __  , ___, P_zs_LL   , P_zl_LL   , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH=1)
    _ , __  , ___, P_zs_noLL , P_zl_noLL , P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH=1)
    if LENS_LIGHT:
        matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_LL, Theta_E_LL, prob_LL, P_zs_LL, P_zl_LL, P_sg_LL
    else:
        matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_noLL, Theta_E_noLL, prob_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL
    N_LENSES = 1 if DENSITY else np.sum(matrix)*np.abs(zl_array[1]-zl_array[0])
    ax[0].plot(zl_array, P_zl * N_LENSES, c=cc2, ls='-', label='DES i band\n Geng VDF')
    ax[0].plot(zs_array, P_zs * N_LENSES, c=cc2, ls='--')
    m_lens = ls.get_len_magnitude_distr(m_obs, zl_array, sigma_array, matrix, obs_band = 'sdss_i0')
    norm = integrate.simps(m_lens, m_obs)
    ax[1].plot(m_obs, m_lens/norm, color=cc2)
    ax[0].legend(fontsize=legend_size)
    if SAVE:
        # folderpath = 'img/'+utils.remove_spaces_from_string(title)
        # if not os.path.exists(folderpath): os.makedirs(folderpath)
        plt.savefig('img/Jacobs_DES.png', dpi=200, bbox_inches='tight')
    plt.show()


def compare_SUGOHI(zl_array, zs_array, sigma_array, LENS_LIGHT = 1, PLOT_FOR_KEYNOTE = 0,
                   SAVE = 0, DENSITY = 1, SMOOTH = 1, legend_size = 18):
    # SUGOHI_upto_2020      = pd.read_csv('../galess/data/LENS_SEARCHES/SUGOHI/SUGOHI_SONNENFELD_2020.tsv', sep=';')
    # SUGOHI_SONN_2018      = SUGOHI_upto_2020[SUGOHI_upto_2020['Ref'] == 'c']
    # SUGOHI_SONN_2020      = SUGOHI_upto_2020[SUGOHI_upto_2020['Ref'] == 'i']
    # SUGOHI_W_data_AB      = pd.read_csv('../galess/data/LENS_SEARCHES/SUGOHI/SUGOHI_Wong_A_B.tsv', sep=';')
    # SUGOHI_W_data_C       = pd.read_csv('../galess/data/LENS_SEARCHES/SUGOHI/SUGOHI_Wong_C.tsv', sep=';')
    # SUGOHI_SONN_2018_zl    = SUGOHI_SONN_2018['zl'].to_numpy().astype('float')
    # # SUGOHI_SONN_2018_zs    = SUGOHI_SONN_2018['zs'].to_numpy().astype('float')
    # # SUGOHI_SONN_2020_zl    = SUGOHI_SONN_2020['zl'].to_numpy().astype('float')
    # # SUGOHI_SONN_2020_zs    = SUGOHI_SONN_2020['zs'].to_numpy().astype('float')
    # SUGOHI_W_data_AB_zl    = SUGOHI_W_data_AB['zL'].to_numpy().astype('float')
    # # SUGOHI_W_data_AB_zs    = SUGOHI_W_data_AB['zS'].to_numpy().astype('float')
    # SUGOHI_W_data_C_zl     = SUGOHI_W_data_C['zL'].to_numpy().astype('float')

    SUGOHI_A = pd.read_csv('../galess/data/LENS_SEARCHES/SUGOHI/SuGOHI_gradeA.csv')
    SUGOHI_B = pd.read_csv('../galess/data/LENS_SEARCHES/SUGOHI/SuGOHI_gradeB.csv')
    SUGOHI_C = pd.read_csv('../galess/data/LENS_SEARCHES/SUGOHI/SuGOHI_gradeC.csv')

    SUGOHI_A_zl = SUGOHI_A['z_lens'].to_numpy().astype('float')
    SUGOHI_A_zl = SUGOHI_A_zl[SUGOHI_A_zl > 0]
    SUGOHI_B_zl = SUGOHI_B['z_lens'].to_numpy().astype('float')
    SUGOHI_B_zl = SUGOHI_B_zl[SUGOHI_B_zl > 0]
    SUGOHI_C_zl = SUGOHI_C['z_lens'].to_numpy().astype('float')
    SUGOHI_C_zl = SUGOHI_C_zl[SUGOHI_C_zl > 0]

    SUGOHI_A_zs = SUGOHI_A['z_source'].to_numpy().astype('float')
    SUGOHI_A_zs = SUGOHI_A_zs[SUGOHI_A_zs > 0]
    SUGOHI_B_zs = SUGOHI_B['z_source'].to_numpy().astype('float')
    SUGOHI_B_zs = SUGOHI_B_zs[SUGOHI_B_zs > 0]
    SUGOHI_C_zs = SUGOHI_C['z_source'].to_numpy().astype('float')
    SUGOHI_C_zs = SUGOHI_C_zs[SUGOHI_C_zs > 0]

    SUGOHI_A_RE = SUGOHI_A['Rein'].to_numpy().astype('float')
    SUGOHI_A_RE = SUGOHI_A_RE[SUGOHI_A_RE > 0]
    SUGOHI_B_RE = SUGOHI_B['Rein'].to_numpy().astype('float')
    SUGOHI_B_RE = SUGOHI_B_RE[SUGOHI_B_RE > 0]
    SUGOHI_C_RE = SUGOHI_C['Rein'].to_numpy().astype('float')
    SUGOHI_C_RE = SUGOHI_C_RE[SUGOHI_C_RE > 0]

    SUGOHI_A_imag = SUGOHI_A['lens_mag_i'].to_numpy().astype('float')
    SUGOHI_A_imag = SUGOHI_A_imag[SUGOHI_A_imag > 0]
    SUGOHI_B_imag = SUGOHI_B['lens_mag_i'].to_numpy().astype('float')
    SUGOHI_B_imag = SUGOHI_B_imag[SUGOHI_B_imag > 0]
    SUGOHI_C_imag = SUGOHI_C['lens_mag_i'].to_numpy().astype('float')
    SUGOHI_C_imag = SUGOHI_C_imag[SUGOHI_C_imag > 0]

    ### PLOT DATA #################################################################################
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    ccc = 'w' if PLOT_FOR_KEYNOTE else 'k'
    cc2 = 'r'
    if PLOT_FOR_KEYNOTE:
        ER_col1, ER_col2, ER_col3, ER_col4, _ALPHA_  = 'darkorange', 'lime', 'magenta', 'cyan', 1
    else:
        ER_col1, ER_col2, ER_col3, ER_col4, _ALPHA_  = 'forestgreen', 'teal', 'orange', 'firebrick', 1
    line_thick = 3
    _nbins_zl = np.arange(0.0, 1.5, 0.1 )
    _nbins_zs = np.arange(0.0, 5  , 0.2 )
    m_obs = np.linspace(15, 30, 31)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.23, hspace=.2)
    _nbins_zl = np.histogram_bin_edges(SUGOHI_A_zl, bins='fd', range=(0,1.5))
    ax[0].hist(SUGOHI_A_zl, bins=_nbins_zl, density=DENSITY, histtype='step', color=ER_col1,
     lw=line_thick, alpha=_ALPHA_, label=f'SuGOHI grade A ({len(SUGOHI_A_zl)})')
    _nbins_zl = np.histogram_bin_edges(SUGOHI_B_zl, bins='fd')
    ax[0].hist(SUGOHI_B_zl, bins=_nbins_zl, density=DENSITY, histtype='step', color=ER_col3,
     lw=line_thick, alpha=_ALPHA_, label=f'SuGOHI grade B ({len(SUGOHI_B_zl)})')
    _nbins_zl = np.histogram_bin_edges(SUGOHI_C_zl, bins='fd')
    ax[0].hist(SUGOHI_C_zl, bins=_nbins_zl, density=DENSITY, histtype='step', color=ER_col2,
     lw=line_thick, alpha=_ALPHA_/2, label=f'SuGOHI grade C ({len(SUGOHI_C_zl)})')

    # _nbins_zs = np.histogram_bin_edges(SUGOHI_A_zs, bins='fd', range=(0,1.5))
    # ax[0].hist(SUGOHI_A_zs, bins=_nbins_zs, density=DENSITY, histtype='step', color=ER_col1, ls='--',
    #  lw=line_thick, alpha=_ALPHA_, label=f'SuGOHI grade A ({len(SUGOHI_A_zs)})')
    # _nbins_zs = np.histogram_bin_edges(SUGOHI_B_zs, bins='fd')
    # ax[0].hist(SUGOHI_B_zs, bins=_nbins_zs, density=DENSITY, histtype='step', color=ER_col3, ls='--',
    #  lw=line_thick, alpha=_ALPHA_, label=f'SuGOHI grade B ({len(SUGOHI_B_zs)})')
    # _nbins_zs = np.histogram_bin_edges(SUGOHI_C_zs, bins='fd')
    # ax[0].hist(SUGOHI_C_zs, bins=_nbins_zs, density=DENSITY, histtype='step', color=ER_col2, ls='--',
    #  lw=line_thick, alpha=_ALPHA_/2, label=f'SuGOHI grade C ({len(SUGOHI_C_zs)})')

    ax[1].hist(SUGOHI_A_imag, bins=m_obs, density=DENSITY, histtype='step', lw = line_thick, color=ER_col1,
               alpha = _ALPHA_, label=f'SuGOHI grade A ({len(SUGOHI_A_imag)})')
    ax[1].hist(SUGOHI_B_imag, bins=m_obs, density=DENSITY, histtype='step', lw = line_thick, color=ER_col3,
               alpha = _ALPHA_, label=f'SuGOHI grade B ({len(SUGOHI_B_imag)})')
    ax[1].hist(SUGOHI_C_imag, bins=m_obs, density=DENSITY, histtype='step', lw = line_thick, color=ER_col2,
               alpha = _ALPHA_/2, label=f'SuGOHI grade C ({len(SUGOHI_C_imag)})')

    title = 'SUBARU HSC SuGOHI i band'
    matrix, Theta_E, prob, P_zs, P_zl, P_sg = get_distributions_from_title(title, sigma_array, zl_array, zs_array,
                                                                           LENS_LIGHT = 1, SMOOTH = SMOOTH, PRINT = 1,
                                                                           zl_lim = (0.2, 1.0), zs_lim = (0, 15), sg_lim = (230, 1000), Rein_lim=0.2)
    N_LENSES = 1 if DENSITY else np.sum(matrix)*np.abs(zl_array[1]-zl_array[0])#*0.025
    ax[0].plot(zl_array, P_zl * N_LENSES, c=ccc, ls='-', label='SUBARU HSC i band\n Mason VDF')
    # ax[0].plot(zs_array, P_zs * N_LENSES, c=ccc, ls='--')
    m_lens = ls.get_len_magnitude_distr(m_obs, zl_array, sigma_array, matrix, obs_band = 'sdss_i0')
    norm = integrate.simps(m_lens, m_obs)
    ax[1].plot(m_obs, m_lens/norm, color=ccc)

    title = 'SUBARU HSC SuGOHI i band VDF Geng'
    matrix, Theta_E, prob, P_zs, P_zl, P_sg = get_distributions_from_title(title, sigma_array, zl_array, zs_array,
                                                                           LENS_LIGHT = 1, SMOOTH = SMOOTH, PRINT = 1,
                                                                           zl_lim = (0.2, 1.0), zs_lim = (0, 15), sg_lim = (230, 1000), Rein_lim=0.2)
    N_LENSES = 1 if DENSITY else np.sum(matrix)*np.abs(zl_array[1]-zl_array[0])#*0.025
    ax[0].plot(zl_array, P_zl * N_LENSES, c=cc2, ls='-', label='SUBARU HSC i band\n Geng VDF')
    # ax[0].plot(zs_array, P_zs * N_LENSES, c=cc2, ls='--')
    m_lens = ls.get_len_magnitude_distr(m_obs, zl_array, sigma_array, matrix, obs_band = 'sdss_i0')
    norm = integrate.simps(m_lens, m_obs)
    ax[1].plot(m_obs, m_lens/norm, color=cc2)

    # # Limit sigma array to >210 km/s
    # reduced_sigma_array = np.linspace(210, 400, 20)
    # title = 'SUBARU HSC SuGOHI i band_sgr210'
    # matrix, Theta_E, prob, P_zs, P_zl, P_sg = get_distributions_from_title(title, reduced_sigma_array, zl_array, zs_array, LENS_LIGHT = 1, SMOOTH = 1)
    # N_LENSES = 1 if DENSITY else np.sum(matrix)*np.abs(zl_array[1]-zl_array[0])#*0.025
    # ax[0].plot(zl_array, P_zl * N_LENSES, c=ccc, ls='--')#, label='SUBARU HSC i band\n Mason VDF')
    # m_lens = ls.get_len_magnitude_distr(m_obs, zl_array, reduced_sigma_array, matrix, obs_band = 'sdss_i0')
    # norm = integrate.simps(m_lens, m_obs)
    # ax[1].plot(m_obs, m_lens/norm, color=ccc, ls='--')

    # title = 'SUBARU HSC SuGOHI i band VDF Geng_sgr210'
    # matrix, Theta_E, prob, P_zs, P_zl, P_sg = get_distributions_from_title(title, reduced_sigma_array, zl_array, zs_array, LENS_LIGHT = 1, SMOOTH = 1)
    # N_LENSES = 1 if DENSITY else np.sum(matrix)*np.abs(zl_array[1]-zl_array[0])#*0.025
    # ax[0].plot(zl_array, P_zl * N_LENSES, c=cc2, ls='--')#, label='SUBARU HSC i band\n Geng VDF')
    # m_lens = ls.get_len_magnitude_distr(m_obs, zl_array, reduced_sigma_array, matrix, obs_band = 'sdss_i0')
    # norm = integrate.simps(m_lens, m_obs)
    # ax[1].plot(m_obs, m_lens/norm, color=cc2, ls='--')


    ax[0].set_xlabel(r'$z_l$', fontsize=20)
    ax[0].set_ylabel(r'$dN/dz$', fontsize=20)
    if DENSITY:
        ax[0].set_ylim((0,5.5))
        ax[0].set_ylabel(r'$dP/dz$', fontsize=20)
    ax[0].set_xlim((0,3.2))
    ax[1].set_xlabel(r'$m_\text{I814W}^\text{len}$ [mag]', fontsize=20)
    ax[1].set_ylim((0,0.55))
    ax[1].set_ylabel(r'$dP/dm$', fontsize=20)
    ax[1].set_xlim((15,25))

    ax[0].legend(fontsize=legend_size, loc=1)
    # ax[1].legend(fontsize=13, loc=1)
    if SAVE:
        # folderpath = 'img/'+utils.remove_spaces_from_string(title)
        # if not os.path.exists(folderpath): os.makedirs(folderpath)
        plt.savefig('img/SUGOHI.png', dpi=200, bbox_inches='tight')
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

    zl_array_CFHTLS = np.arange(0.0 , 2.1 , 0.1)
    zs_array_CFHTLS = np.arange(0.0 , 5.6 , 0.2)
    sg_array_CFHTLS = np.linspace(100 , 400 , 31)


    _title_ = 'PEARLS NEP F115W band_mcut_'
    delta_cut_limit = np.arange(0, 5.25, 0.5)
    gal_num_vs_mcut, gal_num_vs_mcut_LL = np.zeros(0), np.zeros(0)

    for iid, dlt in enumerate(delta_cut_limit):
        cut   = limit - dlt
        title = _title_ + str(iid)
        try:
            #raise(ValueError)
            matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
        except ValueError:
            #print('FILE do NOT exist - RUNNING MODEL')
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

        if(1):
            ax2.plot(np.append(0,zl_array_CFHTLS), np.append(0,P_zl_noLL), c=ccc, ls='-', label=str(limit - dlt))
            ax3.plot(np.append(0,zs_array_CFHTLS), np.append(0,P_zs_noLL), c=ccc, ls='-')
            ax4.plot(sg_array_CFHTLS, P_sg_noLL, c=ccc, ls='-')
        else:
            ax2.plot(np.append(0,zl_array_CFHTLS), np.append(0,P_zl_LL), c=ccc, ls='--')
            ax3.plot(np.append(0,zs_array_CFHTLS), np.append(0,P_zs_LL), c=ccc, ls='--')
            ax4.plot(sg_array_CFHTLS, P_sg_LL, c=ccc, ls='--')
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
    if(0):
        ax2.set_ylabel(r'$dP/dz_l$', fontsize=__size_labels__)
        ax3.set_ylabel(r'$dP/dz_s$', fontsize=__size_labels__)
        ax4.set_ylabel(r'$dP/d\sigma$'   , fontsize=__size_labels__)
    else:
        ax2.set_ylabel(r'Probability distr.', fontsize=__size_labels__)
    ax1.tick_params(axis='both', which = 'major', labelsize=__size_ticks__, direction = 'in', length = 8)
    ax1.tick_params(axis='both', which = 'minor', labelsize=__size_ticks__, direction = 'in', length = 5)
    ax2.tick_params(axis='both', which = 'both', labelsize=__size_ticks__, direction = 'in', length = 5)
    ax3.tick_params(axis='both', which = 'both', labelsize=__size_ticks__, direction = 'in', length = 5)
    ax4.tick_params(axis='both', which = 'both', labelsize=__size_ticks__, direction = 'in', length = 5)
    plt.show()

def compare_all_searches_z(zl_array, zs_array, sigma_array, LENS_LIGHT = 1, PLOT_FOR_KEYNOTE = 0, SAVE = 0, DENSITY = 1, SMOOTH = 1):
    ### DATA ##################################################################################
    FAURE_A_data  = pd.read_csv('../galess/data/LENS_SEARCHES/FAURE_2008/FAURE_A.csv')
    FAURE_B_data  = pd.read_csv('../galess/data/LENS_SEARCHES/FAURE_2008/FAURE_B.csv')
    FAURE_A_zl    = FAURE_A_data['z_l'].to_numpy()
    FAURE_B_zl    = FAURE_B_data['z_l'].to_numpy()

    SL2S_data       = pd.read_csv('../galess/data/LENS_SEARCHES/SL2S_Sonnenfeld/redshifts_sigma.csv')
    SL2S_data_names = SL2S_data['Name'].to_numpy()
    SL2S_data_zl    = SL2S_data['z_l'].to_numpy()
    SL2S_data_zs    = SL2S_data['z_s'].to_numpy()
    SL2S_data_sigma = SL2S_data['sigma'].to_numpy()
    SL2S_data_grade = SL2S_data['Grade'].to_numpy()
    id = np.where(SL2S_data['Name'].notnull())
    SL2S_data_sigma = SL2S_data_sigma[id[0]]
    MASK_A = SL2S_data_grade == 'A'
    SL2S_data_zl_A = SL2S_data_zl[MASK_A]
    SL2S_data_zs_A = SL2S_data_zs[MASK_A]
    SL2S_data_sg_A = SL2S_data_sigma[MASK_A]
    SL2S_data_2013a = pd.read_csv('../galess/data/LENS_SEARCHES/SL2S_Sonnenfeld/data.csv')
    SL2S_data_name2 = SL2S_data_2013a['Name'].to_numpy()
    MASK_bL = np.intersect1d(SL2S_data_name2, SL2S_data_names, return_indices=True)[1]
    MASK_bA = np.intersect1d(SL2S_data_name2, SL2S_data_names[MASK_A], return_indices=True)[1]
    SL2S_data_Rein  = SL2S_data_2013a['R_Ein'].to_numpy()
    SL2S_data_Rein_A= SL2S_data_Rein[MASK_bA]
    SL2S_data_Rein  = SL2S_data_Rein[MASK_bL]

    JAC_DES_data       = pd.read_csv('../galess/data/LENS_SEARCHES/Jacobs_CNN/JACOBS_2019_DES_CNN.tsv', sep=';')
    JAC_DES_data_zl    = JAC_DES_data['z'].to_numpy()
    JAC_DES_data_imag  = JAC_DES_data['imag'].to_numpy()
    AGEL_data = pd.read_csv('../galess/data/LENS_SEARCHES/AGEL/AGEL.csv')
    AGEL_zl   = AGEL_data['z_l'].to_numpy()
    AGEL_zs   = AGEL_data['z_s'].to_numpy()
    AGEL_zl, AGEL_zs = AGEL_zl[AGEL_zl<100], AGEL_zs[AGEL_zs<100]

    SUGOHI_upto_2020      = pd.read_csv('../galess/data/LENS_SEARCHES/SUGOHI/SUGOHI_SONNENFELD_2020.tsv', sep=';')
    SUGOHI_SONN_2018      = SUGOHI_upto_2020[SUGOHI_upto_2020['Ref'] == 'c']
    SUGOHI_W_data_AB      = pd.read_csv('../galess/data/LENS_SEARCHES/SUGOHI/SUGOHI_Wong_A_B.tsv', sep=';')
    SUGOHI_W_data_C       = pd.read_csv('../galess/data/LENS_SEARCHES/SUGOHI/SUGOHI_Wong_C.tsv', sep=';')
    SUGOHI_SONN_2018_zl    = SUGOHI_SONN_2018['zl'].to_numpy().astype('float')
    SUGOHI_W_data_AB_zl    = SUGOHI_W_data_AB['zL'].to_numpy().astype('float')
    SUGOHI_W_data_C_zl     = SUGOHI_W_data_C['zL'].to_numpy().astype('float')

    ### PLOTS #################################################################################
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    ccc = 'w' if PLOT_FOR_KEYNOTE else 'k'
    cc2 = 'r'
    ER_col1, ER_col2, ER_col3, ER_col4, _ALPHA_  = 'forestgreen', 'teal', 'orange', 'sienna', 1
    if PLOT_FOR_KEYNOTE:
        ER_col1, ER_col2, ER_col3, ER_col4, _ALPHA_  = 'darkorange', 'lime', 'magenta', 'cyan', 1
    line_thick = 3

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.13, hspace=.1)
    ### PLOT Faure COSMOS ###############################################################################
    _nbins_zl = zl_array
    if DENSITY: _nbins_zl = np.histogram_bin_edges(np.append(FAURE_A_zl,FAURE_B_zl), bins='fd', range=(0,1.5))
    ax[0].hist(np.append(FAURE_A_zl,FAURE_B_zl), bins=_nbins_zl, density=DENSITY, histtype='step',
                        lw=line_thick, color=ER_col1, alpha = _ALPHA_, label=f'Faure 2008\n Full Sample ({len(np.append(FAURE_A_zl,  FAURE_B_zl))})')
    if DENSITY: _nbins_zl = np.histogram_bin_edges(FAURE_A_zl, bins='fd', range=(0,1.5))
    ax[0].hist(FAURE_A_zl, bins=_nbins_zl, density=DENSITY, histtype='step',
                        lw=line_thick, color=ER_col2, alpha = _ALPHA_, label=f'Faure 2008\n Best Sample ({len(FAURE_A_zl)})')
    title = 'COSMOS HST i band'
    matrix, Theta_E, prob, P_zs, P_zl, P_sg = get_distributions_from_title(title, sigma_array, zl_array, zs_array, LENS_LIGHT = 1, SMOOTH = SMOOTH)
    zl_lower, zl_upper = 0.2, 1
    zs_lower, zs_upper = 0.0, 99
    sg_lower, sg_upper = 160, 1000
    matrix[np.logical_or(zs_array <= zs_lower, zs_array >= zs_upper), :, :] *= 0
    matrix[:, np.logical_or(sigma_array <= sg_lower, sigma_array >= sg_upper), :] *= 0
    matrix[:, :, np.logical_or(zl_array <= zl_lower, zl_array >= zl_upper)] *= 0
    _, __, ___, P_zs, P_zl, P_sg = ls.get_N_and_P_projections(matrix  , sigma_array, zl_array, zs_array, SMOOTH=SMOOTH)
    N_LENSES = 1 if DENSITY else np.sum(matrix)*np.abs(zl_array[1]-zl_array[0])
    ax[0].plot(zl_array, P_zl * N_LENSES, c=ccc, ls='-', label = 'COSMOS HST\n i band\n Mason VDF')
    title = 'COSMOS HST i band VDF Geng'
    matrix, Theta_E, prob, P_zs, P_zl, P_sg = get_distributions_from_title(title, sigma_array, zl_array, zs_array, LENS_LIGHT = 1, SMOOTH = SMOOTH)
    zl_lower, zl_upper = 0.2, 1
    zs_lower, zs_upper = 0.0, 99
    sg_lower, sg_upper = 160, 1000
    matrix[np.logical_or(zs_array <= zs_lower, zs_array >= zs_upper), :, :] *= 0
    matrix[:, np.logical_or(sigma_array <= sg_lower, sigma_array >= sg_upper), :] *= 0
    matrix[:, :, np.logical_or(zl_array <= zl_lower, zl_array >= zl_upper)] *= 0
    _, __, ___, P_zs, P_zl, P_sg = ls.get_N_and_P_projections(matrix  , sigma_array, zl_array, zs_array, SMOOTH=SMOOTH)
    N_LENSES = 1 if DENSITY else np.sum(matrix)*np.abs(zl_array[1]-zl_array[0])
    ax[0].plot(zl_array, P_zl * N_LENSES, c=cc2, ls='-', label = 'COSMOS HST\n i band\n Geng VDF')
    ax[0].set_xlabel(r'$z_l$', fontsize=20)
    ax[0].set_xlim((0,3))

    ### PLOT SL2S #######################################################################################
    # if DENSITY: _nbins_zl = np.histogram_bin_edges(SL2S_data_zl, bins='fd', range=(np.nanmin(SL2S_data_zl),np.nanmax(SL2S_data_zl)))
    if DENSITY: _nbins_zl = np.histogram_bin_edges(SL2S_data_zl, bins='fd', range=(np.nanmin(SL2S_data_zl),np.nanmax(SL2S_data_zl)))
    ax[1].hist(SL2S_data_zl, bins=_nbins_zl, density=DENSITY, histtype='step', lw=line_thick, color=ER_col1, alpha = _ALPHA_,
                label=f'Sonnenfeld et al. 2013\n Full Sample ({len(SL2S_data_zl)})')
    if DENSITY: _nbins_zl = np.histogram_bin_edges(SL2S_data_zl_A, bins='fd', range=(np.nanmin(SL2S_data_zl_A),np.nanmax(SL2S_data_zl_A)))
    ax[1].hist( SL2S_data_zl_A, bins=_nbins_zl, density=DENSITY, histtype='step' , lw=line_thick, color=ER_col2, alpha = _ALPHA_,
                label=f'Sonnenfeld et al. 2013\n Best Sample ({len(SL2S_data_zl_A)})')
    title = 'CFHTLS i band'
    matrix, Theta_E, prob, P_zs, P_zl, P_sg = get_distributions_from_title(title, sigma_array, zl_array, zs_array, LENS_LIGHT = 1, SMOOTH = SMOOTH)
    zl_lower, zl_upper = 0.1, 0.8
    zs_lower, zs_upper = 0.0, 99
    sg_lower, sg_upper = 180, 1000
    matrix[np.logical_or(zs_array <= zs_lower, zs_array >= zs_upper), :, :] *= 0
    matrix[:, np.logical_or(sigma_array <= sg_lower, sigma_array >= sg_upper), :] *= 0
    matrix[:, :, np.logical_or(zl_array <= zl_lower, zl_array >= zl_upper)] *= 0
    _, __, ___, P_zs, P_zl, P_sg = ls.get_N_and_P_projections(matrix  , sigma_array, zl_array, zs_array, SMOOTH=SMOOTH)
    N_LENSES = 1 if DENSITY else np.sum(matrix)*np.abs(zl_array[1]-zl_array[0])*0.3*0.5
    ax[1].plot(zl_array, P_zl * N_LENSES, c=ccc, ls='-', label ='CFHTLS i band\n Mason VDF')
    title = 'CFHTLS i band VDF Geng'
    matrix, Theta_E, prob, P_zs, P_zl, P_sg = get_distributions_from_title(title, sigma_array, zl_array, zs_array, LENS_LIGHT = 1, SMOOTH = SMOOTH)
    zl_lower, zl_upper = 0.1, 0.8
    zs_lower, zs_upper = 0.0, 99
    sg_lower, sg_upper = 180, 1000
    matrix[np.logical_or(zs_array <= zs_lower, zs_array >= zs_upper), :, :] *= 0
    matrix[:, np.logical_or(sigma_array <= sg_lower, sigma_array >= sg_upper), :] *= 0
    matrix[:, :, np.logical_or(zl_array <= zl_lower, zl_array >= zl_upper)] *= 0
    _, __, ___, P_zs, P_zl, P_sg = ls.get_N_and_P_projections(matrix  , sigma_array, zl_array, zs_array, SMOOTH=SMOOTH)
    N_LENSES = 1 if DENSITY else np.sum(matrix)*np.abs(zl_array[1]-zl_array[0])*0.3*0.5
    ax[1].plot(zl_array, P_zl * N_LENSES, c=cc2, ls='-', label ='CFHTLS i band\n Geng VDF')
    ax[1].set_xlabel(r'$z_l$', fontsize=20)
    ax[1].set_xlim((0,2.0))

    ### PLOT Jacobs DES #################################################################################
    # if DENSITY: _nbins_zl = np.histogram_bin_edges(JAC_DES_data_zl, bins='fd', range=(0,np.nanmax(JAC_DES_data_zl)))
    if DENSITY:
        _nbins_zl = np.histogram_bin_edges(JAC_DES_data_zl, bins='fd', range=(0,1.5))
        ax[2].hist(JAC_DES_data_zl, bins=_nbins_zl, density=DENSITY , histtype='step' , lw = line_thick, color=ER_col1, alpha = _ALPHA_,
                    label=f'Jacobs et al. 2019 ({len(JAC_DES_data_zl)})')
    if DENSITY: _nbins_zl = np.histogram_bin_edges(AGEL_zl, bins='fd', range=(0,1.5))
    ax[2].hist(AGEL_zl, bins=_nbins_zl, density=DENSITY , histtype='step' , lw = line_thick, color=ER_col3, alpha = _ALPHA_,
                label=f'Tran et al. 2022 (68)')
    title = 'DES i band'
    matrix, Theta_E, prob, P_zs, P_zl, P_sg = get_distributions_from_title(title, sigma_array, zl_array, zs_array, LENS_LIGHT = 1, SMOOTH = SMOOTH)
    N_LENSES = 1 if DENSITY else np.sum(matrix)*np.abs(zl_array[1]-zl_array[0])*0.88*0.15
    ax[2].plot(zl_array, P_zl* N_LENSES, c=ccc, ls='-', label='DES i band\n Mason VDF')
    title = 'DES i band VDF Geng'
    matrix, Theta_E, prob, P_zs, P_zl, P_sg = get_distributions_from_title(title, sigma_array, zl_array, zs_array, LENS_LIGHT = 1, SMOOTH = SMOOTH)
    N_LENSES = 1 if DENSITY else np.sum(matrix)*np.abs(zl_array[1]-zl_array[0])*0.88*0.15
    ax[2].plot(zl_array, P_zl * N_LENSES, c=cc2, ls='-', label='DES i band\n Geng VDF')
    ax[2].set_xlabel(r'$z_l$', fontsize=20)
    ax[2].set_xlim((0,2.0))

    # ### PLOT SUGOHI #################################################################################
    # # if DENSITY: _nbins_zl = np.histogram_bin_edges(SUGOHI_SONN_2018_zl, bins='fd', range=(0,np.nanmax(SUGOHI_SONN_2018_zl)))
    # if DENSITY: _nbins_zl = np.histogram_bin_edges(SUGOHI_SONN_2018_zl, bins='fd', range=(0,1.5))
    # ax[3].hist(SUGOHI_SONN_2018_zl, bins=_nbins_zl, density=DENSITY, histtype='step', color=ER_col3,
    #  lw=line_thick, alpha=_ALPHA_, label=f'Sonnenfeld et al. 2018\n ({len(SUGOHI_SONN_2018_zl)})')

    # # if DENSITY: _nbins_zl = np.histogram_bin_edges(SUGOHI_W_data_AB_zl, bins='fd')
    # if DENSITY: _nbins_zl = np.histogram_bin_edges(SUGOHI_W_data_AB_zl, bins='fd', range=(0,1.5))
    # ax[3].hist(SUGOHI_W_data_AB_zl, bins=_nbins_zl, density=DENSITY, histtype='step', color=ER_col1,
    #  lw=line_thick, alpha=_ALPHA_, label=f'Wong et al. 2022\n Grade A+B ({len(SUGOHI_W_data_AB_zl)})')

    # # if DENSITY: _nbins_zl = np.histogram_bin_edges(SUGOHI_W_data_C_zl, bins='fd')
    # if DENSITY: _nbins_zl = np.histogram_bin_edges(SUGOHI_W_data_C_zl, bins='fd', range=(0,1.5))
    # ax[3].hist(SUGOHI_W_data_C_zl, bins=_nbins_zl, density=DENSITY, histtype='step', color=ER_col2,
    #  lw=line_thick, alpha=_ALPHA_/2, label=f'Wong et al. 2022\n Grade C ({len(SUGOHI_W_data_C_zl)})')

    # all_SuGOHI = np.append(SUGOHI_W_data_AB_zl, SUGOHI_W_data_C_zl)
    # all_SuGOHI = np.append(all_SuGOHI, SUGOHI_SONN_2018_zl)
    # # if DENSITY: _nbins_zl = np.histogram_bin_edges(all_SuGOHI, bins='fd')
    # if DENSITY: _nbins_zl = np.histogram_bin_edges(all_SuGOHI, bins='fd', range=(0,1.5))
    # ax[3].hist(all_SuGOHI, bins=_nbins_zl,
    #             density=DENSITY, histtype='step', color=ER_col4,
    #             lw=line_thick, alpha=_ALPHA_, label=f'All SuGOHI\n samples ({len(all_SuGOHI)})')
    # title = 'SUBARU HSC SuGOHI i band'
    # matrix, Theta_E, prob, P_zs, P_zl, P_sg = get_distributions_from_title(title, sigma_array, zl_array, zs_array, LENS_LIGHT = 1, SMOOTH = 1)
    # N_LENSES = 1 if DENSITY else np.sum(matrix)*np.abs(zl_array[1]-zl_array[0])
    # ax[3].plot(zl_array, P_zl * N_LENSES, c=ccc, ls='-', label='SUBARU HSC i band\n Mason VDF')
    # title = 'SUBARU HSC SuGOHI i band VDF Geng'
    # matrix, Theta_E, prob, P_zs, P_zl, P_sg = get_distributions_from_title(title, sigma_array, zl_array, zs_array, LENS_LIGHT = 1, SMOOTH = 1)
    # N_LENSES = 1 if DENSITY else np.sum(matrix)*np.abs(zl_array[1]-zl_array[0])
    # ax[3].plot(zl_array, P_zl * N_LENSES, c=cc2, ls='-', label='SUBARU HSC i band\n Geng VDF')
    # ax[3].set_xlabel(r'$z_l$', fontsize=20)
    # ax[3].set_xlim((0,2.5))
    ax[0].set_ylabel(r'$dN/dz$', fontsize=20)
    if DENSITY:
        ax[0].set_ylabel(r'$dP/dz$', fontsize=20)
    for j in range(3):
        ax[j].legend(fontsize=13, loc=1)
        if DENSITY:
            ax[j].set_ylim((0,3.0))
            if j == 0: ax[j].set_ylim((0,2))
            if j == 1: ax[j].set_ylim((0,2.5))
            if j == 3: ax[j].set_ylim((0,6.5))

    if SAVE:
        suffix = 'DENS' if DENSITY else 'NUMBER'
        plt.savefig('img/Searches_z_'+suffix+'.png', dpi=200, bbox_inches='tight')
    plt.show()

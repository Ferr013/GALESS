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

import galess.LensStat.lens_stat as ls
import galess.Utils.ls_utils as utils

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

def plot_survey_ALL_distributions(surveys, READ_FILES = 1, _PLOT_FOR_KEYNOTE = 1):
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

def set_plt_param(PLOT_FOR_KEYNOTE = 1):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['figure.figsize'  ] = (3.3,2.0)
    plt.rcParams['font.size'       ] = 8
    plt.rcParams['axes.labelsize'  ] = 14
    plt.rcParams['legend.fontsize' ] = 8
    plt.rcParams['legend.frameon'  ] = False
    plt.rcParams['legend.title_fontsize'] = 20
    plt.rcParams['font.family'     ] = 'STIXGeneral'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['xtick.direction' ] = 'in'
    plt.rcParams['ytick.direction' ] = 'in'
    plt.rcParams['xtick.top'       ] = True
    plt.rcParams['ytick.right'     ] = True
    plt.rcParams['xtick.labelsize' ] = 15
    plt.rcParams['ytick.labelsize' ] = 15
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

    
def plot_Lens_Fraction(m_lim = 28.5, mu_arc_SIE = 3, 
                       M_array = 0, zs_array_plot = 0, schechter_plot = ls.schechter_LF, 
                       PLOT_FOR_KEYNOTE = 1, SAVE = 0):
    if zs_array_plot == 0: zs_array_plot = np.asarray((1,2,3,4,5,6,7,8,9))
    if M_array == 0: M_array = np.linspace(-14,-26,37)
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    cmap_c = cm.viridis
    f, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True, squeeze=True)
    plt.subplots_adjust(wspace=.075, hspace=.075)
    color = iter(cmap_c(np.linspace(0, 1, len(zs_array_plot))))
    for zs in zs_array_plot:
        _col = next(color)
        M_lim = m_lim - 5 * np.log10(cosmo.luminosity_distance(zs).value * 1e5)
        frl   = ls.Fraction_lensed_SIE(1, M_array, schechter_plot, ls.Phi_vel_disp_Mason, zs)
        ax[0, 0].plot(M_array, frl,        c=_col,  label = r'$z=$'+str(zs))
        C = 0.8979070411803386 #Correction for elliptical caustic area averaged over axis ratio distribution
        #ax[0, 0].axhline(Tau(zs)*C, c=_col, ls='--')
        F_arc = ls.Fraction_1st_image_arc_SIE(mu_arc_SIE, M_array, schechter_plot, zs)
        ax[0, 1].plot(M_array, frl*F_arc,  c=_col,  label = r'$z=$'+str(zs))
        img_N = 2
        F_Nth = ls.Fraction_Nth_image_above_Mlim_SIE(img_N, M_array, M_lim, schechter_plot, zs)
        ax[1, 0].plot(M_array, frl*F_Nth,  c=_col,  label = r'$z=$'+str(zs))
        img_N = 3
        F_Nth = ls.Fraction_Nth_image_above_Mlim_SIE(img_N, M_array, M_lim, schechter_plot, zs)
        ax[1, 1].plot(M_array, frl*F_Nth,  c=_col, ls=':')
        img_N = 4
        F_Nth = ls.Fraction_Nth_image_above_Mlim_SIE(img_N, M_array, M_lim, schechter_plot, zs)
        ax[1, 1].plot(M_array, frl*F_Nth,  c=_col,  label = r'$z=$'+str(zs))    
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
    if SAVE: plt.savefig('img/'+fn_prefix+'frac_lens_SIE.jpg', dpi=100)
    plt.show()

def plot_ALL_distributions(title, zl_array, zs_array, sigma_array,
                               Theta_E_LL, matrix_LL, Theta_E_noLL, matrix_noLL,
                               PLOT_FOR_KEYNOTE = 1, SMOOTH = 1, SAVE = 0, LEGEND = 0):
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

def plot_z_sigma_distributions_double_lenses(title, zl_array, zs_array, sigma_array, matrix_LL, matrix_noLL,
                               PLOT_FOR_KEYNOTE = 1, SMOOTH = 1, SAVE = 0):
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.25, hspace=.2)
    Ngal_zl_sigma_noLL, Ngal_zl_sigma_noLL, Ngal_zl_zs1_noLL, Ngal_zl_zs2_noLL, Ngal_sigma_zs1_noLL, Ngal_sigma_zs2_noLL, Ngal_zs1_zs2_noLL, P_zs_noLL, P_zs2_noLL, P_zl_noLL, P_sg_noLL = ls.get_N_and_P_projections_double_lens(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH)
    Ngal_zl_sigma_LL, Ngal_zl_sigma_LL, Ngal_zl_zs1_LL, Ngal_zl_zs2_LL, Ngal_sigma_zs1_LL, Ngal_sigma_zs2_LL, Ngal_zs1_zs2_LL, P_zs_LL, P_zs2_LL, P_zl_LL, P_sg_LL = ls.get_N_and_P_projections_double_lens(matrix_LL, sigma_array, zl_array, zs_array, SMOOTH)
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
    if (SAVE):
        folderpath = 'img/'+utils.remove_spaces_from_string(title)
        if not os.path.exists(folderpath): os.makedirs(folderpath)
        plt.savefig(folderpath+'/'+fn_prefix+'corner_plts.jpg', dpi=200)
    plt.show()

def plot_z_sigma_distributions(fig, ax, title, zl_array, zs_array, sigma_array,
                               Theta_E_LL, matrix_LL, Theta_E_noLL, matrix_noLL,
                               PLOT_FOR_KEYNOTE = 1, CONTOUR = 1, LOG = 0, SMOOTH = 0, SAVE = 0, 
                               LEGEND = 1, DOUBLE_LENS = 0):
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    if DOUBLE_LENS:
        Ngal_zl_sigma_noLL, Ngal_zl_sigma_noLL, Ngal_zl_zs1_noLL, Ngal_zl_zs2_noLL, Ngal_sigma_zs1_noLL, Ngal_sigma_zs2_noLL, Ngal_zs1_zs2_noLL, P_zs_noLL, P_zs2_noLL, P_zl_noLL, P_sg_noLL = ls.get_N_and_P_projections_double_lens(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH)
        Ngal_zl_sigma_LL, Ngal_zl_sigma_LL, Ngal_zl_zs1_LL, Ngal_zl_zs2_LL, Ngal_sigma_zs1_LL, Ngal_sigma_zs2_LL, Ngal_zs1_zs2_LL, P_zs_LL, P_zs2_LL, P_zl_LL, P_sg_LL = ls.get_N_and_P_projections_double_lens(matrix_LL, sigma_array, zl_array, zs_array, SMOOTH)
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
    if (SAVE):
        folderpath = 'img/'+utils.remove_spaces_from_string(title)
        if not os.path.exists(folderpath): os.makedirs(folderpath)
        plt.savefig(folderpath+'/'+fn_prefix+'corner_plts.jpg', dpi=200)


def plot_effect_vel_disp_function(zl_array, zs_array, sigma_array, 
                                    PLOT_FOR_KEYNOTE = 1, LENS_LIGHT = 1,
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
        plt.savefig(folderpath+'/VDF_effect_corner_plts.jpg', dpi=200)
    plt.show()


def compare_ALL_distributions_surveys(surveys_selection, sigma_array, zl_array, zs_array, LENS_LIGHT = 1, PLOT_FOR_KEYNOTE = 1, SMOOTH = 1):
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

def plot_z_distribution_in_ax(ax, title, color, zl_array, zs_array, sigma_array, matrix, SMOOTH = 1):
    _n, __n, ___n, P_zs, P_zl, P_sg = ls.get_N_and_P_projections(matrix, sigma_array, zl_array, zs_array, SMOOTH)
    ax.plot(zl_array, P_zl, c=color, ls='-' , label=title)
    ax.plot(zs_array, P_zs, c=color, ls=':' )
def plot_s_distribution_in_ax(ax, title, color, zl_array, zs_array, sigma_array, matrix, SMOOTH = 1):
    _n, __n, ___n, P_zs, P_zl, P_sg = ls.get_N_and_P_projections(matrix, sigma_array, zl_array, zs_array, SMOOTH)
    ax.plot(sigma_array, P_sg, c=color, ls='-', label=title)
def plot_R_distribution_in_ax(ax, title, color, _nbins_Re, matrix, Theta_E, SMOOTH = 1):
    ax.hist(np.ravel(Theta_E), weights=np.ravel(matrix), bins=_nbins_Re, range=(0, 3), density=True, histtype='step', color=color, ls = '-', label=title)


def single_compare_z_distributions_surveys(ax, title, color,
                                        zl_array, zs_array, sigma_array, matrix_LL, matrix_noLL,
                                        PLOT_FOR_KEYNOTE = 1, SMOOTH = 1, PLOT_ALL = 0):
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
                                        PLOT_FOR_KEYNOTE = 1, SMOOTH = 1, PLOT_ALL = 0):
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
                                        PLOT_FOR_KEYNOTE = 1, SMOOTH = 1, PLOT_ALL = 0):
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

def compare_z_distributions_surveys(surveys_selection, sigma_array, zl_array, zs_array, PLOT_FOR_KEYNOTE = 1):
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

def compare_sigma_distributions_surveys(surveys_selection, sigma_array, zl_array, zs_array, PLOT_FOR_KEYNOTE = 1):
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
                            frac_lens = 0.1, PLOT_FOR_KEYNOTE = 1,
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



def plot_lens_src_magnitudes(survey_titles, zl_array, zs_array, sigma_array, M_array_UV, AVG_MAGNIF_3 = 0, LENS_LIGHT = 1, PLOT_FOR_KEYNOTE = 1):
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.23, hspace=.2)
    
    color = iter(cmap_c(np.linspace(0, 1, len(survey_titles)+1)))
    for iit, title in enumerate(survey_titles):
        ccc = next(color)
        survey_params = utils.read_survey_params(title, VERBOSE = 0)
        photo_band   = survey_params['photo_band']
        limit        = survey_params['limit']
        cut          = survey_params['cut']

        matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
        _ , __  , ___, P_zs_LL  , P_zl_LL  , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL, sigma_array, zl_array, zs_array, SMOOTH=1)
        _ , __  , ___, P_zs_noLL  , P_zl_noLL  , P_sg_noLL   = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH=1)

        if LENS_LIGHT:
            matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_LL, Theta_E_LL, prob_LL, P_zs_LL, P_zl_LL, P_sg_LL
        else:
            matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_noLL, Theta_E_noLL, prob_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL
    
        m_obs = np.arange(15, 35, 0.25)
        
        m_lens = ls.get_len_magnitude_distr(m_obs, zl_array, sigma_array, matrix, obs_band = photo_band)
        norm_lens = integrate.simps(m_lens, m_obs)
        ax[0].plot(m_obs, m_lens/norm_lens, color=ccc, label = title)
        ax[0].set_xlabel(r'$m_\text{len}$ [mag]', fontsize=20)
        ax[0].set_ylabel(r'$dP/dm_\text{lens}$', fontsize=20)
        ax[0].set_xlim((15,30))
        ax[0].set_ylim((0,0.5))
        ax[0].legend(fontsize = 12)

        m_src = ls.get_src_magnitude_distr(m_obs, cut, zs_array, prob, M_array_UV, obs_band = photo_band)
        norm_src = integrate.simps(m_src, m_obs)
        ax[1].plot(m_obs, m_src/norm_src, color=ccc)
        ax[1].set_xlabel(r'$m_\text{src}$ [mag]', fontsize=20)
        ax[1].set_ylabel(r'$dP/dm_\text{src}$', fontsize=20)
        ax[1].set_xlim((20,30))

        if AVG_MAGNIF_3 :
            ax[0].axvline(float(cut-2.5*np.log10(3)), color=ccc, ls = '--')
            ax[1].axvline(float(cut-2.5*np.log10(3)), color=ccc, ls = '--')
        else:
            ax[0].axvline(float(cut), color=ccc, ls = '--')
            ax[1].axvline(float(cut), color=ccc, ls = '--')
        
    plt.show()






def compare_COSMOS_Web_Ering(zl_array, zs_array, sigma_array, PLOT_FOR_KEYNOTE = 1):
    E_ring_rad     = 1.54/2 #" 
    zl, spzl, smzl = 1.94,   0.13,   0.17
    zs, spzs, smzs = 2.98,   0.42,   0.47
    Ml, spMl, smMl = 6.5e11, 3.7e11, 1.5e11
    sg, spsg, smsg = 336   , 145   , 55
    CW_ER_rd, CW_ER_zl, CW_ER_zs, CW_ER_Ml, CW_ER_sg =  E_ring_rad, [zl, spzl, smzl], [zs, spzs, smzs], [Ml, spMl, smMl], [sg, spsg, smsg]

    zl_nozero_idx, zs_nozero_idx, sig_nozero_idx = ls.get_param_space_idx_from_obs_constraints(CW_ER_zl, CW_ER_zs, E_ring_rad, zs_array, zl_array, sigma_array)
    zl_param_space = zl_array[zl_nozero_idx]
    zs_param_space = zs_array[zs_nozero_idx]
    sg_param_space = sigma_array[sig_nozero_idx]

    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.23, hspace=.2)
    COSMOS_JWST_surveys = ['COSMOS Web F115W', 'COSMOS Web F150W', 'COSMOS Web F277W']
    #COSMOS_JWST_surveys = ['COSMOS Web F115W']
    _col_  = iter(cmap_c(np.linspace(0, 1, len(COSMOS_JWST_surveys)+1)))
    for title in COSMOS_JWST_surveys:
        ccc = next(_col_)
        matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
        _ , __  , ___, P_zs_LL  , P_zl_LL  , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH=1)
        ax[0].plot(zl_array, P_zl_LL, c=ccc, ls='-', label=title)
        ax[0].plot(zs_array, P_zs_LL, c=ccc, ls='--')
        ax[0].set_xlabel(r'$z$', fontsize=20) 
        ax[0].set_ylabel(r'$dP/dz$', fontsize=20)
        ax[0].set_xlim((0,5.2))
        ax[1].plot(sigma_array, P_sg_LL  , c=ccc, ls = '-', label=title)
        ax[1].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
        ax[1].set_ylabel(r'$dP/d\sigma$', fontsize=20)
        ax[1].set_xlim((100,400))
        ax[2].hist(np.ravel(Theta_E_LL), weights=np.ravel(matrix_LL), bins = np.arange(0, 4, 0.2), 
                   range=(0, 3), density=True, histtype='step', color=ccc, ls = '-', label=title)
        ax[2].set_xlabel(r'$\Theta_E$ ["]', fontsize=20)
        ax[2].set_ylabel(r'$dP/d\Theta_E$', fontsize=20)
        ax[2].set_xlim((0,4))
    ER_col = 'orange' if PLOT_FOR_KEYNOTE else 'forestgreen'
    ax[0].axvline(CW_ER_zl[0], color=ER_col)
    ax[0].axvspan(CW_ER_zl[0]-CW_ER_zl[2], CW_ER_zl[0]+CW_ER_zl[1], alpha=0.25, color=ER_col)
    ax[0].axvline(CW_ER_zs[0], color=ER_col, ls='--')
    ax[0].axvspan(CW_ER_zs[0]-CW_ER_zs[2], CW_ER_zs[0]+CW_ER_zs[1], alpha=0.25, facecolor="none", edgecolor=ER_col, hatch='x')
    ax[1].axvline(CW_ER_sg[0], color=ER_col)
    ax[1].axvspan(CW_ER_sg[0]-CW_ER_sg[2], CW_ER_sg[0]+CW_ER_sg[1], alpha=0.25, color=ER_col)
    ax[2].axvline(CW_ER_rd, color=ER_col, label='E-Ring (van Dokkum+23)')
    #ax[0].legend()
    #ax[1].legend()
    ax[2].legend(fontsize=10)
    plt.show()

    ################################################################################################################################################################

    fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.23, hspace=.2)
    COSMOS_JWST_surveys = ['COSMOS Web F115W', 'COSMOS Web F150W', 'COSMOS Web F277W']
    _col_  = iter(cmap_c(np.linspace(0, 1, len(COSMOS_JWST_surveys)+1)))
    title = COSMOS_JWST_surveys[0]
    ccc = next(_col_)
    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
    Ngal_zl_sigma_LL, Ngal_zs_sigma_LL, Ngal_zs_zl_LL, _ , __ , ___ = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH=1)

    _sigma, _zl = np.meshgrid(sigma_array, zl_array)
    levels   = np.asarray([0.01, 0.1, 0.5, 1, 2, 3, 5])*(np.power(10,np.ceil(np.log10(np.max(Ngal_zl_sigma_LL))-1)))
    contours = ax[0].contour(_sigma, _zl, Ngal_zl_sigma_LL.T, levels, cmap=cmap_c, norm=colors.Normalize(vmin=np.min(Ngal_zl_sigma_LL), vmax=np.max(Ngal_zl_sigma_LL)), linestyles='-')
    ax[0].clabel(contours, inline=True, fontsize=8)
    ax[0].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
    ax[0].set_ylabel(r'$z_l$', fontsize=20)

    _sigma, _zs = np.meshgrid(sigma_array, zs_array)
    levels   = np.asarray([0.01, 0.1, 0.5, 1, 2, 3, 5])*(np.power(10,np.ceil(np.log10(np.max(Ngal_zs_sigma_LL))-1)))
    contours = ax[1].contour(_sigma, _zs, Ngal_zs_sigma_LL, levels, cmap=cmap_c, norm=colors.Normalize(vmin=np.min(Ngal_zs_sigma_LL), vmax=np.max(Ngal_zs_sigma_LL)), linestyles='-')
    ax[1].clabel(contours, inline=True, fontsize=8)
    ax[1].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
    ax[1].set_ylabel(r'$z_s$', fontsize=20)

    _zs, _zl = np.meshgrid(zs_array, zl_array)
    levels   = np.asarray([0.01, 0.1, 0.5, 1, 2, 3, 5])*(np.power(10,np.ceil(np.log10(np.max(Ngal_zs_zl_LL))-1)))
    contours = ax[2].contour(_zs, _zl, Ngal_zs_zl_LL.T, levels, cmap=cmap_c, norm=colors.Normalize(vmin=np.min(Ngal_zs_zl_LL), vmax=np.max(Ngal_zs_zl_LL)), linestyles='-')
    ax[2].clabel(contours, inline=True, fontsize=8)
    ax[2].set_xlabel(r'$z_s$', fontsize=20)
    ax[2].set_ylabel(r'$z_l$', fontsize=20)

    ER_col = 'orange' if PLOT_FOR_KEYNOTE else 'forestgreen'
    ax[0].scatter(CW_ER_sg[0], CW_ER_zl[0], color=ER_col, marker='*')
    ax[1].scatter(CW_ER_sg[0], CW_ER_zs[0], color=ER_col, marker='*')    
    ax[2].scatter(CW_ER_zs[0], CW_ER_zl[0], color=ER_col, marker='*')

    ax[0].axvline(CW_ER_sg[0]-CW_ER_sg[2], alpha=0.25, color=ER_col)
    ax[0].axvline(CW_ER_sg[0]+CW_ER_sg[1], alpha=0.25, color=ER_col)
    ax[1].axvline(CW_ER_sg[0]-CW_ER_sg[2], alpha=0.25, color=ER_col)
    ax[1].axvline(CW_ER_sg[0]+CW_ER_sg[1], alpha=0.25, color=ER_col)
    ax[2].axvline(CW_ER_zs[0]-CW_ER_zs[2], alpha=0.25, color=ER_col)
    ax[2].axvline(CW_ER_zs[0]+CW_ER_zs[1], alpha=0.25, color=ER_col)

    ax[0].axhline(CW_ER_zl[0]-CW_ER_zl[2], alpha=0.25, color=ER_col)
    ax[0].axhline(CW_ER_zl[0]+CW_ER_zl[1], alpha=0.25, color=ER_col)
    ax[1].axhline(CW_ER_zs[0]-CW_ER_zs[2], alpha=0.25, color=ER_col)
    ax[1].axhline(CW_ER_zs[0]+CW_ER_zs[1], alpha=0.25, color=ER_col)
    ax[2].axhline(CW_ER_zl[0]-CW_ER_zl[2], alpha=0.25, color=ER_col)
    ax[2].axhline(CW_ER_zl[0]+CW_ER_zl[1], alpha=0.25, color=ER_col)
    
    
    _dzl, _dzs, _dsg = np.diff(zl_array)[0] / 2, np.diff(zs_array)[0] / 2, np.diff(sigma_array)[0] / 2
    for _zl_, _zs_, _sg_ in zip(zl_param_space, zs_param_space, sg_param_space):
        ax[0].add_patch(Rectangle((_sg_ - _dsg, _zl_ - _dzl), 2 * _dsg, 2 * _dzl, facecolor = ER_col, alpha = 0.25))
        ax[1].add_patch(Rectangle((_sg_ - _dsg, _zs_ - _dzs), 2 * _dsg, 2 * _dzs, facecolor = ER_col, alpha = 0.25))
        ax[2].add_patch(Rectangle((_zs_ - _dzs, _zl_ - _dzl), 2 * _dzs, 2 * _dzl, facecolor = ER_col, alpha = 0.25))
    plt.show()



def compare_COSMOS_HST_Faure(zl_array, zs_array, sigma_array, M_array_UV, mag_cut, ONLY_FULL_SAMPLE = 0, LENS_LIGHT = 1, __MAG_OVER_ARCSEC_SQ__ = 0, PLOT_FOR_KEYNOTE = 1):
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

    FAURE_A_data  = pd.read_csv('../galess/data/FAURE_2008/FAURE_A.csv')
    FAURE_A_names = FAURE_A_data['Cosmos Name'].to_numpy()
    FAURE_A_zl    = FAURE_A_data['z_l'].to_numpy()
    FAURE_A_Rarc  = FAURE_A_data['R_arc'].to_numpy()
    FAURE_A_Reff  = FAURE_A_data['Reff'].to_numpy()
    FAURE_A_m_Ib  = FAURE_A_data['mag_814W'].to_numpy()
    FAURE_A_ell   = FAURE_A_data['ell'].to_numpy()
    FAURE_A_m_src = FAURE_A_data['mag_814W_src'].to_numpy()

    FAURE_B_data  = pd.read_csv('../galess/data/FAURE_2008/FAURE_B.csv')
    FAURE_B_names = FAURE_B_data['Cosmos Name'].to_numpy()
    FAURE_B_zl    = FAURE_B_data['z_l'].to_numpy()
    FAURE_B_Rarc  = FAURE_B_data['R_arc'].to_numpy()
    FAURE_B_Reff  = FAURE_B_data['Reff'].to_numpy()
    FAURE_B_m_Ib  = FAURE_B_data['mag_814W'].to_numpy()
    FAURE_B_ell   = FAURE_B_data['ell'].to_numpy()
    FAURE_B_m_src = FAURE_B_data['mag_814W_src'].to_numpy()
    ### PLOT DATA #################################################################################
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.23, hspace=.2)

    ccc = 'w' if PLOT_FOR_KEYNOTE else 'k'
    cc2 = 'r'


    _nbins_zl = np.arange(0.0, 1.6, 0.2 )
    _nbins_zs = np.arange(0.0, 5  , 0.5 )
    _nbins_sg = np.arange(100, 400, 25  )
    _nbins_Re = np.arange(0  , 4  , 0.25)
    m_obs = np.linspace(15, 30, 31)
    line_thick = 3
    if PLOT_FOR_KEYNOTE:
        ER_col1, ER_col2  = 'darkorange', 'lime'
        _ALPHA_ = 1
    else:
        ER_col1, ER_col2  = 'forestgreen', 'firebrick'
        _ALPHA_ = 1

    ### FULL SAMPLE ###
    F_zl_hist, F_zl_bins = ax[0].hist( np.append(FAURE_A_zl,  FAURE_B_zl)          , bins=_nbins_zl, density=True, histtype='step', lw=line_thick, color=ER_col1, alpha = _ALPHA_, label=f'Faure 2008 - Full Sample ({len(np.append(FAURE_A_zl,  FAURE_B_zl))})')
    F_Ra_hist, F_Ra_bins = ax[1].hist( np.append(FAURE_A_Rarc/1.5,FAURE_B_Rarc/1.5), bins=_nbins_Re, density=True, histtype='step', lw=line_thick, color=ER_col1, alpha = _ALPHA_)
    F_mI_hist, F_mI_bins = ax[2].hist( np.append(FAURE_A_m_Ib, FAURE_B_m_Ib)       , bins=m_obs  , density=True, histtype='step', lw=line_thick, color=ER_col1, alpha = _ALPHA_)

    title = 'COSMOS HST i band FAURE'
    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
    _ , __  , ___, P_zs_LL  , P_zl_LL  , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL, sigma_array, zl_array, zs_array, SMOOTH=1)
    _ , __  , ___, P_zs_noLL  , P_zl_noLL  , P_sg_noLL   = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH=1)
    if LENS_LIGHT: matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_LL, Theta_E_LL, prob_LL, P_zs_LL, P_zl_LL, P_sg_LL
    else: matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_noLL, Theta_E_noLL, prob_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL
    ax[0].plot(zl_array, P_zl, c=ccc, ls='-', label=title)
    ax[0].plot(zs_array, P_zs, c=ccc, ls='--')
    ax[0].set_xlim((0,5.2))
    ax[0].set_xlabel(r'$z$', fontsize=20) 
    ax[0].set_ylabel(r'$dP/dz$', fontsize=20)
    T_hist_Mason, T_bins_Mason = ax[1].hist(np.ravel(Theta_E), weights=np.ravel(matrix), bins = np.arange(0, 4, 0.2), 
                range=(0, 3), density=True, histtype='step', color=ccc, ls = '-', label=title)
    ax[1].set_xlabel(r'$\Theta_E$ ["]', fontsize=20)
    ax[1].set_ylabel(r'$dP/d\Theta_E$', fontsize=20)
    ax[1].set_xlim((0,4))
    m_lens = ls.get_len_magnitude_distr(m_obs, zl_array, sigma_array, matrix)
    norm = integrate.simps(m_lens, m_obs)
    ax[2].plot(m_obs, m_lens/norm, color=ccc)
    ax[2].set_xlabel(r'$m_\text{I814W}^\text{len}$ [mag]', fontsize=20)
    ax[2].set_ylabel(r'$dP/dm$', fontsize=20)
    ax[2].set_xlim((15,25))
    ax[2].set_ylim((0,0.5))

    pval_zl_Mason = kstest(F_zl_hist, P_zl)[1]
    pval_Ra_Mason = kstest(F_Ra_hist, T_hist_Mason)[1]
    pval_mI_Mason = kstest(F_mI_hist, m_lens/norm)[1]

    title = 'COSMOS HST i band FAURE Geng'
    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
    _ , __  , ___, P_zs_LL  , P_zl_LL  , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL, sigma_array, zl_array, zs_array, SMOOTH=1)
    _ , __  , ___, P_zs_noLL  , P_zl_noLL  , P_sg_noLL   = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH=1)
    if LENS_LIGHT: matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_LL, Theta_E_LL, prob_LL, P_zs_LL, P_zl_LL, P_sg_LL
    else: matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_noLL, Theta_E_noLL, prob_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL
    ax[0].plot(zl_array, P_zl, c=cc2, ls='-', label=title)
    ax[0].plot(zs_array, P_zs, c=cc2, ls='--')
    ax[0].set_xlim((0,5.2))
    ax[0].set_xlabel(r'$z$', fontsize=20) 
    ax[0].set_ylabel(r'$dP/dz$', fontsize=20)
    ax[1].hist(np.ravel(Theta_E), weights=np.ravel(matrix), bins = np.arange(0, 4, 0.2), 
                range=(0, 3), density=True, histtype='step', color=cc2, ls = '-', label=title)
    ax[1].set_xlabel(r'$\Theta_E$ ["]', fontsize=20)
    ax[1].set_ylabel(r'$dP/d\Theta_E$', fontsize=20)
    ax[1].set_xlim((0,4))
    m_lens = ls.get_len_magnitude_distr(m_obs, zl_array, sigma_array, matrix)
    norm = integrate.simps(m_lens, m_obs)
    ax[2].plot(m_obs, m_lens/norm, color=cc2)
    ax[2].set_xlabel(r'$m_\text{I814W}^\text{len}$ [mag]', fontsize=20)
    ax[2].set_ylabel(r'$dP/dm$', fontsize=20)
    ax[2].set_xlim((15,25))
    ax[2].set_ylim((0,0.5))
    
    ax[0].legend(fontsize=10)
    plt.show()


def compare_SL2S(zl_array, zs_array, sigma_array, LENS_LIGHT = 1, PLOT_FOR_KEYNOTE = 1):
    SL2S_data       = pd.read_csv('../galess/data/SL2S_Sonnenfeld/redshifts_sigma.csv')
    SL2S_data_names = SL2S_data['Name'].to_numpy()
    SL2S_data_zl    = SL2S_data['z_l'].to_numpy()
    SL2S_data_zs    = SL2S_data['z_s'].to_numpy()
    SL2S_data_sigma = SL2S_data['sigma'].to_numpy()
    id = np.where(SL2S_data['Name'].notnull())
    SL2S_data_sigma = SL2S_data_sigma[id[0]]
    SL2S_dataB      = pd.read_csv('../galess/data/SL2S_Sonnenfeld/data.csv')
    SL2S_data_nameB = SL2S_dataB['Name'].to_numpy()
    SL2S_data_Rein  = SL2S_dataB['R_Ein'].to_numpy()
    SL2S_data_msrc  = SL2S_dataB['mag_src'].to_numpy()

    ### PLOT DATA #################################################################################
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    ccc = 'w' if PLOT_FOR_KEYNOTE else 'k'
    cc2 = 'r'
    if PLOT_FOR_KEYNOTE: 
        ER_col1, ER_col2, _ALPHA_  = 'darkorange', 'lime', 1
    else: 
        ER_col1, ER_col2, _ALPHA_  = 'forestgreen', 'firebrick', 1

    _nbins_zl = np.arange(0.0, 1.2, 0.3 )
    _nbins_zs = np.arange(0.5, 4  , 0.5 )
    _nbins_sg = np.arange(100, 400, 25  )
    _nbins_Re = np.arange(0  , 4  , 0.25)
    

    title = 'CFHTLS i band'
    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
    _ , __  , ___, P_zs_LL   , P_zl_LL   , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH=1)
    _ , __  , ___, P_zs_noLL , P_zl_noLL , P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH=1)
    if LENS_LIGHT: matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_LL, Theta_E_LL, prob_LL, P_zs_LL, P_zl_LL, P_sg_LL
    else: matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_noLL, Theta_E_noLL, prob_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.23, hspace=.2)

    line_thick = 3
    ### BEST SAMPLE ###
    ax[0].hist( SL2S_data_zl    , bins=_nbins_zl, density=True , histtype='step' , lw=line_thick, color=ER_col1, alpha = _ALPHA_, label='Sonnenfeld et al. 2013 - Full Sample (53)')
    ax[0].hist( SL2S_data_zs    , bins=_nbins_zs, density=True , histtype='step' , lw=line_thick, color=ER_col1, alpha = _ALPHA_, ls='--')
    ax[1].hist( SL2S_data_sigma , bins=_nbins_sg, density=True , histtype='step' , lw=line_thick, color=ER_col1, alpha = _ALPHA_)
    ax[2].hist( SL2S_data_Rein  , bins=_nbins_Re, density=True , histtype='step' , lw=line_thick, color=ER_col1, alpha = _ALPHA_)

    ax[0].plot(zl_array, P_zl, c=ccc, ls='-', label=title)
    ax[0].plot(zs_array, P_zs, c=ccc, ls='--')
    ax[0].set_xlim((0,5.2))
    ax[0].set_xlabel(r'$z$', fontsize=20) 
    ax[0].set_ylabel(r'$dP/dz$', fontsize=20)
    ax[1].plot(sigma_array, P_sg_noLL, c=ccc, ls = '-', label=title)
    ax[1].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
    ax[1].set_ylabel(r'$dP/d\sigma$', fontsize=20)
    ax[2].hist(np.ravel(Theta_E), weights=np.ravel(matrix), bins = np.arange(0, 4, 0.2), 
                range=(0, 3), density=True, histtype='step', color=ccc, ls = '-', label=title)
    ax[2].set_xlabel(r'$\Theta_E$ ["]', fontsize=20)
    ax[2].set_ylabel(r'$dP/d\Theta_E$', fontsize=20)
    ax[2].set_xlim((0,4))

    title = 'CFHTLS i band Geng'
    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
    _ , __  , ___, P_zs_LL   , P_zl_LL   , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH=1)
    _ , __  , ___, P_zs_noLL , P_zl_noLL , P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH=1)
    if LENS_LIGHT: matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_LL, Theta_E_LL, prob_LL, P_zs_LL, P_zl_LL, P_sg_LL
    else: matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_noLL, Theta_E_noLL, prob_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL
    ax[0].plot(zl_array, P_zl, c=cc2, ls='-', label=title)
    ax[0].plot(zs_array, P_zs, c=cc2, ls='--')
    ax[0].set_xlim((0,5.2))
    ax[0].set_xlabel(r'$z$', fontsize=20) 
    ax[0].set_ylabel(r'$dP/dz$', fontsize=20)
    ax[1].plot(sigma_array, P_sg_noLL, c=cc2, ls = '-', label=title)
    ax[1].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
    ax[1].set_ylabel(r'$dP/d\sigma$', fontsize=20)
    ax[2].hist(np.ravel(Theta_E), weights=np.ravel(matrix), bins = np.arange(0, 4, 0.2), 
                range=(0, 3), density=True, histtype='step', color=cc2, ls = '-', label=title)
    ax[2].set_xlabel(r'$\Theta_E$ ["]', fontsize=20)
    ax[2].set_ylabel(r'$dP/d\Theta_E$', fontsize=20)
    ax[2].set_xlim((0,4))
    ax[0].legend(fontsize=10)
    plt.show()


def compare_CASSOWARY(zl_array, zs_array, sigma_array, LENS_LIGHT = 1, PLOT_FOR_KEYNOTE = 1):
    CASS_data       = pd.read_csv('../galess/data/CASSOWARY/cassowary.csv')
    CASS_data_names = CASS_data['Name'].to_numpy()
    CASS_data_zl    = CASS_data['z_l'].to_numpy()
    CASS_data_zs    = CASS_data['z_s'].to_numpy()
    CASS_data_sigma = CASS_data['sigma'].to_numpy()
    return 0



def compare_JACOBS_CNN_DES(zl_array, zs_array, sigma_array, LENS_LIGHT = 1, PLOT_FOR_KEYNOTE = 1):
    JAC_DES_data       = pd.read_csv('../galess/data/Jacobs_CNN/JACOBS_2019_DES_CNN.tsv', sep=';')
    JAC_DES_data_zl    = JAC_DES_data['z'].to_numpy()
    JAC_DES_data_imag  = JAC_DES_data['imag'].to_numpy()

    ### PLOT DATA #################################################################################
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    ccc = 'w' if PLOT_FOR_KEYNOTE else 'k'
    cc2 = 'r'
    if PLOT_FOR_KEYNOTE: 
        ER_col1, ER_col2, _ALPHA_  = 'darkorange', 'lime', 1
    else: 
        ER_col1, ER_col2, _ALPHA_  = 'forestgreen', 'firebrick', 1
    title = 'DES i band'
    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
    _ , __  , ___, P_zs_LL   , P_zl_LL   , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH=1)
    _ , __  , ___, P_zs_noLL , P_zl_noLL , P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH=1)
    if LENS_LIGHT: matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_LL, Theta_E_LL, prob_LL, P_zs_LL, P_zl_LL, P_sg_LL
    else: matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_noLL, Theta_E_noLL, prob_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL
    m_obs = np.linspace(15, 30, 31)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.23, hspace=.2)
    line_thick = 3
    _nbins_zl = np.arange(0.0, 2.2, 0.1 )
    ### BEST SAMPLE ###
    ax[0].hist( JAC_DES_data_zl    , bins=_nbins_zl, density=True , histtype='step' , lw = line_thick, color=ER_col1, alpha = _ALPHA_, label='Jacobs et al. 2019 (511)')
    ax[1].hist( JAC_DES_data_imag, bins=m_obs  , density=True, histtype='step', lw = line_thick, color=ER_col1, alpha = _ALPHA_)
    
    ax[0].plot(zl_array, P_zl, c=ccc, ls='-', label=title)
    ax[0].plot(zs_array, P_zs, c=ccc, ls='--')
    ax[0].set_xlim((0,3.2))
    ax[0].set_xlabel(r'$z$', fontsize=20) 
    ax[0].set_ylabel(r'$dP/dz$', fontsize=20)
    m_lens = ls.get_len_magnitude_distr(m_obs, zl_array, sigma_array, matrix, obs_band = 'sdss_i0')
    norm = integrate.simps(m_lens, m_obs)
    ax[1].plot(m_obs, m_lens/norm, color=ccc)
    ax[1].set_xlabel(r'$m_\text{I814W}^\text{len}$ [mag]', fontsize=20)
    ax[1].set_ylabel(r'$dP/dm$', fontsize=20)
    ax[1].set_xlim((15,25))
    ax[1].set_ylim((0,0.5))

    title = 'DES i band Geng'
    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
    _ , __  , ___, P_zs_LL   , P_zl_LL   , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH=1)
    _ , __  , ___, P_zs_noLL , P_zl_noLL , P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH=1)
    if LENS_LIGHT: matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_LL, Theta_E_LL, prob_LL, P_zs_LL, P_zl_LL, P_sg_LL
    else: matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_noLL, Theta_E_noLL, prob_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL    
    ax[0].plot(zl_array, P_zl, c=cc2, ls='-', label=title)
    ax[0].plot(zs_array, P_zs, c=cc2, ls='--')
    ax[0].set_xlim((0,3.2))
    ax[0].set_xlabel(r'$z$', fontsize=20) 
    ax[0].set_ylabel(r'$dP/dz$', fontsize=20)
    m_lens = ls.get_len_magnitude_distr(m_obs, zl_array, sigma_array, matrix, obs_band = 'sdss_i0')
    norm = integrate.simps(m_lens, m_obs)
    ax[1].plot(m_obs, m_lens/norm, color=cc2)
    ax[1].set_xlabel(r'$m_\text{I814W}^\text{len}$ [mag]', fontsize=20)
    ax[1].set_ylabel(r'$dP/dm$', fontsize=20)
    ax[1].set_xlim((15,25))
    ax[1].set_ylim((0,0.5))
    ax[0].legend(fontsize=10)
    plt.show()



def compare_GRILLO_SLACS(zl_array, zs_array, sigma_array, LENS_LIGHT = 1, PLOT_FOR_KEYNOTE = 1):
    ### from Grillo+(2009) --- selection of grade A lenses from Bolton+2008
    SLACS_title=['SDSS Name' , 'zl' , 'zs' ,'REi','t_e', 'u_mag', 'g_mag','r_mag' ,'i_mag' ,'z_mag']
    SLACS_data=[['J0008-0004', 0.440, 1.192, 1.16, 1.71, 22.632 , 20.655 , 19.294 , 18.654 , 18.082],
                ['J0029-0055', 0.227, 0.931, 0.96, 2.16, 20.479 , 18.792 , 17.546 ,	17.072 , 16.728],
                ['J0037-0942', 0.196, 0.632, 1.53, 2.19, 19.780 , 18.038 , 16.807 ,	16.340 , 15.993],
                ['J0044+0113', 0.120, 0.197, 0.79, 2.61, 18.701 , 17.120 , 16.195 ,	15.771 , 15.461],
                ['J0109+1500', 0.294, 0.525, 0.69, 1.38, 22.867 , 19.753 , 18.204 ,	17.641 , 17.261],
                ['J0157-0056', 0.513, 0.924, 0.79, 1.06, 22.722 , 21.258 , 19.670 ,	18.702 , 18.280],
                ['J0216-0813', 0.332, 0.523, 1.16, 2.67, 21.075 , 19.124 , 17.456 ,	16.860 , 16.573],
                ['J0252+0039', 0.280, 0.982, 1.04, 1.39, 21.331 , 20.061 , 18.813 ,	18.237 , 17.916],
                ['J0330-0020', 0.351, 1.071, 1.10, 1.20, 20.714 , 19.947 , 18.462 ,	17.919 , 17.564],
                ['J0405-0455', 0.075, 0.810, 0.80, 1.36, 19.504 , 17.618 , 16.771 ,	16.365 , 16.060],
                ['J0728+3835', 0.206, 0.688, 1.25, 1.78, 20.404 , 18.623 , 17.356 ,	16.887 , 16.583],
                ['J0737+3216', 0.322, 0.581, 1.00, 2.82, 21.239 , 19.400 , 17.834 ,	17.214 , 16.872],
                ['J0822+2652', 0.241, 0.594, 1.17, 1.82, 20.548 , 18.899 , 17.501 ,	16.986 , 16.627],
                ['J0903+4116', 0.430, 1.065, 1.29, 1.78, 21.401 , 20.302 , 18.646 ,	17.967 , 17.540],
                ['J0912+0029', 0.164, 0.324, 1.63, 3.87, 19.327 , 17.410 , 16.229 ,	15.746 , 15.379],
                ['J0935-0003', 0.347, 0.467, 0.87, 4.24, 21.331 , 19.193 , 17.515 ,	16.915 , 16.554],
                ['J0936+0913', 0.190, 0.588, 1.09, 2.11, 20.073 , 18.218 , 17.002 ,	16.555 , 16.226],
                ['J0946+1006', 0.222, 0.609, 1.38, 2.35, 20.465 , 18.899 , 17.578 ,	17.102 , 16.744],
                ['J0956+5100', 0.240, 0.470, 1.33, 2.19, 20.175 , 18.475 , 17.129 ,	16.633 , 16.247],
                ['J0959+4416', 0.237, 0.531, 0.96, 1.98, 20.452 , 18.850 , 17.510 ,	17.016 , 16.698],
                ['J0959+0410', 0.126, 0.535, 0.99, 1.39, 20.403 , 18.697 , 17.639 ,	17.168 , 16.763],
                ['J1016+3859', 0.168, 0.439, 1.09, 1.46, 20.388 , 18.434 , 17.265 ,	16.827 , 16.502],
                ['J1020+1122', 0.282, 0.553, 1.20, 1.59, 21.470 , 19.508 , 18.003 ,	17.465 , 17.172],
                ['J1023+4230', 0.191, 0.696, 1.41, 1.77, 20.390 , 18.664 , 17.366 ,	16.871 , 16.548],
                ['J1029+0420', 0.104, 0.615, 1.01, 1.56, 19.340 , 17.556 , 16.629 ,	16.224 , 15.884],
                ['J1100+5329', 0.317, 0.858, 1.52, 2.24, 21.003 , 19.161 , 17.704 ,	17.103 , 16.869],
                ['J1106+5228', 0.096, 0.407, 1.23, 1.68, 18.841 , 16.940 , 16.001 ,	15.612 , 15.287],
                ['J1112+0826', 0.273, 0.629, 1.49, 1.50, 21.775 , 19.375 , 17.809 ,	17.243 , 16.906],
                ['J1134+6027', 0.153, 0.474, 1.10, 2.02, 19.936 , 18.085 , 17.024 ,	16.538 , 16.192],
                ['J1142+1001', 0.222, 0.504, 0.98, 1.91, 20.711 , 18.836 , 17.495 ,	17.002 , 16.723],
                ['J1143-0144', 0.106, 0.402, 1.68, 4.80, 18.635 , 16.845 , 15.827 ,	15.405 , 15.062],
                ['J1153+4612', 0.180, 0.875, 1.05, 1.16, 20.575 , 18.793 , 17.680 ,	17.218 , 16.895],
                ['J1204+0358', 0.164, 0.631, 1.31, 1.47, 20.330 , 18.544 , 17.399 ,	16.936 , 16.661],
                ['J1205+4910', 0.215, 0.481, 1.22, 2.59, 20.744 , 18.541 , 17.234 ,	16.718 , 16.343],
                ['J1213+6708', 0.123, 0.640, 1.42, 3.23, 19.054 , 17.176 , 16.159 ,	15.731 , 15.377],
                ['J1218+0830', 0.135, 0.717, 1.45, 3.18, 19.223 , 17.394 , 16.343 ,	15.893 , 15.543],
                ['J1250+0523', 0.232, 0.795, 1.13, 1.81, 19.983 , 18.500 , 17.256 ,	16.733 , 16.464],
                ['J1402+6321', 0.205, 0.481, 1.35, 2.70, 20.393 , 18.293 , 16.952 ,	16.444 , 16.077],
                ['J1403+0006', 0.189, 0.473, 0.83, 1.46, 20.371 , 18.687 , 17.506 ,	17.028 , 16.721],
                ['J1416+5136', 0.299, 0.811, 1.37, 1.43, 21.225 , 19.592 , 18.080 ,	17.521 , 17.171],
                ['J1420+6019', 0.063, 0.535, 1.04, 2.06, 18.196 , 16.386 , 15.541 ,	15.153 , 14.869],
                ['J1430+4105', 0.285, 0.575, 1.52, 2.55, 20.331 , 18.974 , 17.706 ,	17.196 , 16.851],
                ['J1436-0000', 0.285, 0.805, 1.12, 2.24, 21.250 , 19.216 , 17.801 ,	17.229 , 16.876],
                ['J1443+0304', 0.134, 0.419, 0.81, 0.94, 20.607 , 18.524 , 17.483 ,	17.034 , 16.696],
                ['J1451-0239', 0.125, 0.520, 1.04, 2.48, 19.240 , 17.538 , 16.480 ,	16.129 , 15.673],
                ['J1525+3327', 0.358, 0.717, 1.31, 2.90, 20.936 , 19.446 , 17.877 ,	17.247 , 16.935],
                ['J1531-0105', 0.160, 0.744, 1.71, 2.50, 19.475 , 17.519 , 16.362 ,	15.915 , 15.591],
                ['J1538+5817', 0.143, 0.531, 1.00, 1.58, 19.495 , 18.174 , 17.173 ,	16.741 , 16.429],
                ['J1621+3931', 0.245, 0.602, 1.29, 2.14, 20.502 , 18.780 , 17.380 ,	16.859 , 16.553],
                ['J1627-0053', 0.208, 0.524, 1.23, 1.98, 20.623 , 18.588 , 17.286 ,	16.805 , 16.490],
                ['J1630+4520', 0.248, 0.793, 1.78, 1.96, 20.594 , 18.877 , 17.396 ,	16.862 , 16.541],
                ['J1636+4707', 0.228, 0.674, 1.09, 1.68, 20.980 , 19.005 , 17.665 ,	17.174 , 16.877],
                ['J2238-0754', 0.137, 0.713, 1.27, 2.33, 19.716 , 17.803 , 16.766 ,	16.309 , 15.963],
                ['J2300+0022', 0.229, 0.464, 1.24, 1.83, 20.517 , 19.007 , 17.647 ,	17.127 , 16.783],
                ['J2303+1422', 0.155, 0.517, 1.62, 3.28, 19.467 , 17.562 , 16.385 ,	15.907 , 15.585],
                ['J2321-0939', 0.082, 0.532, 1.60, 4.11, 18.085 , 16.145 , 15.200 ,	14.771 , 14.458],
                ['J2341+0000', 0.186, 0.807, 1.44, 3.15, 19.513 , 18.121 , 16.921 ,	16.381 , 16.010]]
    SLACS_data  = np.asarray(SLACS_data)
    SLACS_names = SLACS_data[:,0]
    SLACS_zl, SLACS_zs   = np.asarray(SLACS_data[:,1], dtype='float'), np.asarray(SLACS_data[:,2], dtype='float')
    SLACS_REin, SLACS_Te = np.asarray(SLACS_data[:,3], dtype='float'), np.asarray(SLACS_data[:,4], dtype='float')
    SLACS_umag, SLACS_gmag = np.asarray(SLACS_data[:,5], dtype='float'), np.asarray(SLACS_data[:,6], dtype='float')
    SLACS_rmag, SLACS_imag, SLACS_zmag = np.asarray(SLACS_data[:,7], dtype='float'), np.asarray(SLACS_data[:,8], dtype='float'), np.asarray(SLACS_data[:,9], dtype='float')

    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    ccc = 'w' if PLOT_FOR_KEYNOTE else 'k'
    if PLOT_FOR_KEYNOTE: 
        ER_col1, ER_col2, _ALPHA_  = 'darkorange', 'lime', 1
    else: 
        ER_col1, ER_col2, _ALPHA_  = 'forestgreen', 'firebrick', 1

    title = 'COSMOS HST i band'
    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
    _ , __  , ___, P_zs_LL   , P_zl_LL   , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH=1)
    _ , __  , ___, P_zs_noLL , P_zl_noLL , P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH=1)

    if LENS_LIGHT:
        matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_LL, Theta_E_LL, prob_LL, P_zs_LL, P_zl_LL, P_sg_LL
    else:
        matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_noLL, Theta_E_noLL, prob_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL
    m_obs = np.linspace(15, 30, 31)
    
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.23, hspace=.2)
    ax[0].plot(zl_array, P_zl, c=ccc, ls='-', label=title)
    ax[0].plot(zs_array, P_zs, c=ccc, ls='--')
    ax[0].set_xlim((0,3.2))
    ax[0].set_xlabel(r'$z$', fontsize=20) 
    ax[0].set_ylabel(r'$dP/dz$', fontsize=20)

    m_lens_i = ls.get_len_magnitude_distr(m_obs, zl_array, sigma_array, matrix, obs_band = 'sdss_i0')
    m_lens_g = ls.get_len_magnitude_distr(m_obs, zl_array, sigma_array, matrix, obs_band = 'sdss_g0')
    norm_i = integrate.simps(m_lens_i, m_obs)
    norm_g = integrate.simps(m_lens_g, m_obs)
    # ax[1].plot(m_obs, m_lens_i/norm_i, color=ccc)
    ax[1].plot(m_obs, m_lens_g/norm_g, color=ccc)
    ax[1].set_xlabel(r'$m_{g\text{-band}}^\text{len}$ [mag]', fontsize=20)
    ax[1].set_ylabel(r'$dP/dm$', fontsize=20)
    ax[1].set_xlim((15,25))
    ax[1].set_ylim((0,0.75))

    ax[2].hist(np.ravel(Theta_E), weights=np.ravel(matrix), bins = np.arange(0, 4, 0.2), 
                range=(0, 3), density=True, histtype='step', color=ccc, ls = '-', label=title)
    ax[2].set_xlabel(r'$\Theta_E$ ["]', fontsize=20)
    ax[2].set_ylabel(r'$dP/d\Theta_E$', fontsize=20)
    ax[2].set_xlim((0,4))

    _nbins_zl = np.arange(0.0, 1.2, 0.1 )
    _nbins_zs = np.arange(0.0, 4  , 0.2 )
    _nbins_sg = np.arange(100, 400, 25  )
    _nbins_Re = np.arange(0  , 4  , 0.25)

    ### BEST SAMPLE ###
    ax[0].hist( SLACS_zl    , bins=_nbins_zl, density=True , histtype='step' , color=ER_col1, alpha = _ALPHA_, label=f'Grillo et al. ({len(SLACS_zl)})')
    ax[0].hist( SLACS_zs    , bins=_nbins_zs, density=True , histtype='step' , color=ER_col1, alpha = _ALPHA_, ls = '--')
    ax[1].hist( SLACS_gmag  , bins=m_obs    , density=True , histtype='step' , color=ER_col1, alpha = _ALPHA_)
    ax[2].hist( SLACS_REin  , bins=_nbins_Re, density=True , histtype='step' , color=ER_col1, alpha = _ALPHA_)
    ax[0].legend(fontsize=10)
    plt.show()


def compare_SUGOHI(zl_array, zs_array, sigma_array, LENS_LIGHT = 1, PLOT_FOR_KEYNOTE = 1):
    SUGOHI_upto_2020      = pd.read_csv('../galess/data/SUGOHI/SUGOHI_SONNENFELD_2020.tsv', sep=';')
    SUGOHI_SONN_2018      = SUGOHI_upto_2020[SUGOHI_upto_2020['Ref'] == 'c']
    SUGOHI_SONN_2020      = SUGOHI_upto_2020[SUGOHI_upto_2020['Ref'] == 'i']
    SUGOHI_W_data_AB      = pd.read_csv('../galess/data/SUGOHI/SUGOHI_Wong_A_B.tsv', sep=';')
    SUGOHI_W_data_C       = pd.read_csv('../galess/data/SUGOHI/SUGOHI_Wong_C.tsv', sep=';')
    
    SUGOHI_SONN_2018_zl    = SUGOHI_SONN_2018['zl'].to_numpy().astype('float')
    # SUGOHI_SONN_2018_zs    = SUGOHI_SONN_2018['zs'].to_numpy().astype('float')
    # SUGOHI_SONN_2020_zl    = SUGOHI_SONN_2020['zl'].to_numpy().astype('float')
    # SUGOHI_SONN_2020_zs    = SUGOHI_SONN_2020['zs'].to_numpy().astype('float')
    SUGOHI_W_data_AB_zl    = SUGOHI_W_data_AB['zL'].to_numpy().astype('float')
    # SUGOHI_W_data_AB_zs    = SUGOHI_W_data_AB['zS'].to_numpy().astype('float')
    SUGOHI_W_data_C_zl     = SUGOHI_W_data_C['zL'].to_numpy().astype('float')

    ### PLOT DATA #################################################################################
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    ccc = 'w' if PLOT_FOR_KEYNOTE else 'k'
    if PLOT_FOR_KEYNOTE: 
        ER_col1, ER_col2, ER_col3, ER_col4, _ALPHA_  = 'darkorange', 'lime', 'magenta', 'cyan', 1
    else: 
        ER_col1, ER_col2, ER_col3, ER_col4, _ALPHA_  = 'forestgreen', 'firebrick', 'orange', 'teal', 1

    title = 'SUBARU HSC SuGOHI i band'
    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
    _ , __  , ___, P_zs_LL   , P_zl_LL   , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH=1)
    _ , __  , ___, P_zs_noLL , P_zl_noLL , P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH=1)

    if LENS_LIGHT:
        matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_LL, Theta_E_LL, prob_LL, P_zs_LL, P_zl_LL, P_sg_LL
    else:
        matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_noLL, Theta_E_noLL, prob_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL
    
    _nbins_zl = np.arange(0.0, 1.5, 0.1 )
    _nbins_zs = np.arange(0.0, 5  , 0.2 )

    fig, ax = plt.subplots(1, 1, figsize=(6, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.23, hspace=.2)
    ax.plot(zl_array, P_zl, c=ccc, ls='-', label=title)
    ax.hist( SUGOHI_SONN_2018_zl, bins=_nbins_zl, density=True, histtype='step', color=ER_col3, alpha = _ALPHA_, label=f'Sonnenfeld et al. 2018 ({len(SUGOHI_SONN_2018_zl)})')
    # ax[0].hist( SUGOHI_SONN_2020_zl, bins=_nbins_zl, density=True, histtype='step', color=ER_col4, alpha = _ALPHA_, label=f'Sonnenfeld et al. 2020 ({len(SUGOHI_SONN_2020_zl)})')
    ax.hist( SUGOHI_W_data_AB_zl, bins=_nbins_zl, density=True, histtype='step', color=ER_col1, alpha = _ALPHA_, label=f'Wong et al. 2022 - Grade A+B ({len(SUGOHI_W_data_AB_zl)})')
    ax.hist( SUGOHI_W_data_C_zl , bins=_nbins_zl, density=True, histtype='step', color=ER_col2, alpha = _ALPHA_, label=f'Wong et al. 2022 - Grade C ({len(SUGOHI_W_data_C_zl)})')
    ax.set_xlim((0,2.0))
    ax.set_ylim((0,6.5))
    ax.set_xlabel(r'$z_l$', fontsize=20) 
    ax.set_ylabel(r'$dP/dz$', fontsize=20)
    ax.legend(fontsize=10, loc=1)
    if(0):
        ax[1].plot(zs_array, P_zs, c=ccc, ls='-')   
        # ax[1].hist(SUGOHI_SONN_2018_zs, bins=_nbins_zs, density=True, histtype='step', color=ER_col3, alpha = _ALPHA_)
        # ax[1].hist(SUGOHI_SONN_2020_zs, bins=_nbins_zs, density=True, histtype='step', color=ER_col4, alpha = _ALPHA_)
        # ax[1].hist(SUGOHI_W_data_AB_zs, bins=_nbins_zs, density=True, histtype='step', color=ER_col1, alpha = _ALPHA_)
        ax[1].set_xlabel(r'$z_s$', fontsize=20)
        ax[1].set_ylabel(r'$dP/dz$', fontsize=20)
        ax[1].set_xlim((0,5.2))
    plt.show()

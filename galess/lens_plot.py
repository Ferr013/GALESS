import os.path
import numpy as np
import pandas as pd
from scipy import integrate

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

import lens_stat as ls
import ls_utils as utils



def set_plt_param(PLOT_FOR_KEYNOTE = 1):
    params_paper = {
        'axes.labelsize': 14,
        'legend.fontsize': 7,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False,
        'figure.figsize': [6, 4]
        }
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
        "savefig.edgecolor": 'lightgray'
        }
    plt.rcParams.update(plt.rcParamsDefault)
    if(PLOT_FOR_KEYNOTE): 
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
    else:
        plt.rcParams.update(params_paper)   
        line_c = 'k'
        cmap_c = cm.inferno
        _col_  = None
        col_A  = 'r'
        col_B  = 'k'
        col_C  = 'k'
        col_D  = 'k'
        fn_prefix = ''
        return line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix

def plot_z_sigma_distributions(fig, ax, title, zl_array, zs_array, sigma_array,
                               Theta_E_LL, matrix_LL, Theta_E_noLL, matrix_noLL,
                               PLOT_FOR_KEYNOTE = 1, CONTOUR = 1, LOG = 0, SMOOTH = 0, SAVE = 0):

    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
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
        ax[0,0].set_ylabel(r'$P$', fontsize=20)
        ax[0,0].set_xlabel(r'$z$', fontsize=20)
        ax[0,0].set_xlim((0,5.2))
        


    ax[1,1].plot(sigma_array, P_sg_LL, c=col_C, ls = ':')
    ax[1,1].plot(sigma_array, P_sg_noLL, c=col_C, ls = '-')
    ax[1,1].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
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



def compare_z_distributions_surveys(ax, title, color,
                                        zl_array, zs_array, sigma_array, matrix_LL, matrix_noLL,
                                        PLOT_FOR_KEYNOTE = 1, SMOOTH = 0):
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    _n, __n, ___n, P_zs_noLL, P_zl_noLL, P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH)
    _ , __  , ___, P_zs_LL  , P_zl_LL  , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH)
    ax[0].plot(zl_array, P_zl_noLL, c=color, ls='-' , label=title)
    ax[0].plot(zs_array, P_zs_noLL, c=color, ls=':')
    ax[1].plot(zl_array, P_zl_LL, c=color, ls='--', label=title)
    ax[1].plot(zs_array, P_zs_LL, c=color, ls=':')
    ax[0].set_ylabel(r'$P$', fontsize=20)
    ax[0].set_xlabel(r'$z$', fontsize=20) 
    ax[1].set_xlabel(r'$z$', fontsize=20)

def single_compare_z_distributions_surveys(ax, title, color,
                                        zl_array, zs_array, sigma_array, matrix_LL, matrix_noLL,
                                        PLOT_FOR_KEYNOTE = 1, SMOOTH = 1):
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    _n, __n, ___n, P_zs_noLL, P_zl_noLL, P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH)
    _ , __  , ___, P_zs_LL  , P_zl_LL  , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH)
    ax[0].plot(zl_array, P_zl_noLL, c=color, ls='-' , label=title)
    ax[0].plot(zs_array, P_zs_noLL, c=color, ls=':')
    ax[1].plot(zl_array, P_zl_LL, c=color, ls='--', label=title)
    ax[1].plot(zs_array, P_zs_LL, c=color, ls=':')
    ax[0].set_xlabel(r'$z$', fontsize=20) 
    ax[1].set_xlabel(r'$z$', fontsize=20) 
    ax[0].set_ylabel(r'$dP/dz$', fontsize=20)

def single_compare_sigma_distributions_surveys(ax, title, color,
                                        zl_array, zs_array, sigma_array, matrix_LL, matrix_noLL,
                                        PLOT_FOR_KEYNOTE = 1, SMOOTH = 1):
    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    _n, __n, ___n, P_zs_noLL, P_zl_noLL, P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH)
    _ , __  , ___, P_zs_LL  , P_zl_LL  , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH)
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





    
def plot_angular_separation(survey_title, zs_array, omega, cmap_c = cm.cool, 
                            SPLIT_REDSHIFTS = 0, PLOT_ACF = 0, PLOT_FOR_KEYNOTE = 1, 
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

    rad_to_arcsec = 1/206265
    _theta_arcsec = np.logspace(-1,3.333334,14)
    omega[_theta_arcsec<1]  = 0
    omega[_theta_arcsec>10] = 0
    dPdt = 2 * np.pi * _theta_arcsec * np.diff(_theta_arcsec)[0] * (omega + 1)
    cPdt = np.cumsum(dPdt)/np.cumsum(dPdt)[6]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.15, hspace=.2)   
    ax[0].set_ylabel(r'$P(<\theta$)', fontsize=20)
    ax[0].set_xlabel(r'$\theta$ [arcsec]', fontsize=20)
    ax[1].set_xlabel(r'$\theta$ [arcsec]', fontsize=20)
    ax[0].set_xlim((0,10))
    ax[1].set_xlim((0,10))
    ax[0].set_ylim((0,1.1))
    ax[1].set_ylim((0,1.1))
    if(SPLIT_REDSHIFTS):
        iterabel_zs_array = np.asarray((1, 3, 5, 7, 9))
        color = iter(cmap_c(np.linspace(0, 1, len(iterabel_zs_array)+1)))
        for zs in iterabel_zs_array:
            ccc = next(color)
            izs = np.argmin(np.abs(zs_array-zs))
            lw = 3 if (zs==1 or zs==9) else 1
            ax[0].hist(np.ravel(Theta_pos_noLL[izs][:][:]), weights=np.ravel(matrix_noLL[izs][:][:]), bins=200, range=(0, 12), 
            density=True, histtype='step', color=ccc, label=str(zs), lw=lw, cumulative=True)
            ax[1].hist(np.ravel(Theta_pos_LL[izs][:][:]), weights=np.ravel(matrix_LL[izs][:][:]), bins=200, range=(0, 12), 
            density=True, histtype='step', color=ccc, label=str(zs), lw=lw, cumulative=True)
    else:
        color = iter(cmap_c(np.linspace(0, 1, 2)))
        ax[0].hist(np.ravel(Theta_pos_noLL), weights=np.ravel(matrix_noLL), bins=200, range=(0, 12), 
            density=True, histtype='step', color=next(color), label='no LL', cumulative=True)
        ax[1].hist(np.ravel(Theta_pos_LL), weights=np.ravel(matrix_LL), bins=200, range=(0, 12), 
            density=True, histtype='step', color=next(color), label='w/ LL', cumulative=True)
    if(PLOT_ACF): 
        ax[0].plot(th_r_array, P_Rgalpos, 'g:')
        ax[1].plot(th_r_array, P_Rgalpos, 'g:')
        ax[0].plot(th_r_array, P_rnd_rnd, 'y:')
        ax[1].plot(th_r_array, P_rnd_rnd, 'y:')
    ax[0].plot(_theta_arcsec, cPdt, 'r:')
    ax[1].plot(_theta_arcsec, cPdt, 'r:')

    plt.legend(fontsize=15, loc='lower right')
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
    title = 'COSMOS HST i band FAURE'
    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
    _ , __  , ___, P_zs_LL  , P_zl_LL  , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL, sigma_array, zl_array, zs_array, SMOOTH=1)
    _ , __  , ___, P_zs_noLL  , P_zl_noLL  , P_sg_noLL   = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH=1)

    if LENS_LIGHT:
        matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_LL, Theta_E_LL, prob_LL, P_zs_LL, P_zl_LL, P_sg_LL
    else:
        matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_noLL, Theta_E_noLL, prob_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL
    
    ax[0].plot(zl_array, P_zl, c=ccc, ls='-', label=title)
    ax[0].plot(zs_array, P_zs, c=ccc, ls='--')
    ax[0].set_xlim((0,5.2))
    ax[0].set_xlabel(r'$z$', fontsize=20) 
    ax[0].set_ylabel(r'$dP/dz$', fontsize=20)
    
    ax[1].hist(np.ravel(Theta_E), weights=np.ravel(matrix), bins = np.arange(0, 4, 0.2), 
                range=(0, 3), density=True, histtype='step', color=ccc, ls = '-', label=title)
    ax[1].set_xlabel(r'$\Theta_E$ ["]', fontsize=20)
    ax[1].set_ylabel(r'$dP/d\Theta_E$', fontsize=20)
    ax[1].set_xlim((0,4))


    m_obs = np.linspace(15, 30, 31)
    #ax[2].plot(m_obs, m_num, color=ccc)
    if __MAG_OVER_ARCSEC_SQ__:         
        ax[2].set_xlabel(r'$m_\text{I814W}^\text{src}$ [mag arcsec$^{-2}$]', fontsize=20)
        ax[2].set_ylabel(r'$N_\text{gal}$', fontsize=20)
        ax[2].set_xlim((19,mag_cut-1))
        ax[2].set_ylim((0,50))
    else:
        # ax[2].axvline(mag_cut, color=ccc, ls = '--', alpha=0.75)
        # ax[2].set_xlabel(r'$m_\text{I814W}^\text{src}$ [mag]', fontsize=20)
        # ax[2].set_ylabel(r'$N_\text{gal}$', fontsize=20)
        # ax[2].set_xlim((19,mag_cut+1))
        m_lens = ls.get_len_magnitude_distr(m_obs, zl_array, sigma_array, matrix)
        norm = integrate.simps(m_lens, m_obs)
        ax[2].plot(m_obs, m_lens/norm, color=ccc)
        ax[2].set_xlabel(r'$m_\text{I814W}^\text{len}$ [mag]', fontsize=20)
        ax[2].set_ylabel(r'$dP/dm$', fontsize=20)
        ax[2].set_xlim((15,25))
        ax[2].set_ylim((0,0.5))
    

    _nbins_zl = np.arange(0.0, 1.6, 0.2 )
    _nbins_zs = np.arange(0.0, 5  , 0.5 )
    _nbins_sg = np.arange(100, 400, 25  )
    _nbins_Re = np.arange(0  , 4  , 0.25)
    if PLOT_FOR_KEYNOTE:
        ER_col1, ER_col2  = 'darkorange', 'lime'
        _ALPHA_ = 1
    else:
        ER_col1, ER_col2  = 'forestgreen', 'firebrick'
        _ALPHA_ = 1
    ### FULL SAMPLE ###
    ax[0].hist( np.append(FAURE_A_zl,  FAURE_B_zl)          , bins=_nbins_zl, density=True, histtype='step', color=ER_col1, alpha = _ALPHA_, label='Faure 2008 - Full Sample (53)')
    ax[1].hist( np.append(FAURE_A_Rarc/1.5,FAURE_B_Rarc/1.5), bins=_nbins_Re, density=True, histtype='step', color=ER_col1, alpha = _ALPHA_)
    if __MAG_OVER_ARCSEC_SQ__:        
        ax[2].hist( np.append(FAURE_A_m_src, FAURE_B_m_src)       , bins=m_obs  , density=False, histtype='step', color=ER_col1, alpha = _ALPHA_)
    else:
        ax[2].hist( np.append(FAURE_A_m_Ib, FAURE_B_m_Ib)       , bins=m_obs  , density=True, histtype='step', color=ER_col1, alpha = _ALPHA_)
    if not ONLY_FULL_SAMPLE:
        ### BEST SAMPLE ###
        ax[0].hist( FAURE_zl      , bins=_nbins_zl, density=True, histtype='step', color=ER_col2, alpha = _ALPHA_, label='Faure 2008 - Best Sample (16)')
        ax[0].hist( FAURE_zs      , bins=_nbins_zs, density=True, histtype='step', color=ER_col2, alpha = _ALPHA_, ls='--')
        ax[1].hist( FAURE_Rein    , bins=_nbins_Re, density=True, histtype='step', color=ER_col2, alpha = _ALPHA_)
        if __MAG_OVER_ARCSEC_SQ__:        
            ax[2].hist( FAURE_m_src   , bins=m_obs    , density=False, histtype='step', color=ER_col2, alpha = _ALPHA_)
        else:
            ax[2].hist( FAURE_m_Ib, bins=m_obs  , density=True, histtype='step', color=ER_col2, alpha = _ALPHA_)

    ax[0].legend(fontsize=10)
    plt.show()


def compare_Sl2S(zl_array, zs_array, sigma_array, LENS_LIGHT = 1, PLOT_FOR_KEYNOTE = 1):
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
    if PLOT_FOR_KEYNOTE: 
        ER_col1, ER_col2, _ALPHA_  = 'darkorange', 'lime', 1
    else: 
        ER_col1, ER_col2, _ALPHA_  = 'forestgreen', 'firebrick', 1

    title = 'CFHTLS i band'
    matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
    _ , __  , ___, P_zs_LL   , P_zl_LL   , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH=1)
    _ , __  , ___, P_zs_noLL , P_zl_noLL , P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH=1)

    if LENS_LIGHT:
        matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_LL, Theta_E_LL, prob_LL, P_zs_LL, P_zl_LL, P_sg_LL
    else:
        matrix, Theta_E, prob, P_zs, P_zl, P_sg = matrix_noLL, Theta_E_noLL, prob_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL
    
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.23, hspace=.2)
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

    _nbins_zl = np.arange(0.0, 1.2, 0.3 )
    _nbins_zs = np.arange(0.5, 4  , 0.5 )
    _nbins_sg = np.arange(100, 400, 25  )
    _nbins_Re = np.arange(0  , 4  , 0.25)

    ### BEST SAMPLE ###
    ax[0].hist( SL2S_data_zl    , bins=_nbins_zl, density=True , histtype='step' , color=ER_col1, alpha = _ALPHA_, label='Sonnenfeld 2013 - Full Sample (53)')
    ax[0].hist( SL2S_data_zs    , bins=_nbins_zs, density=True , histtype='step' , color=ER_col1, alpha = _ALPHA_, ls='--')
    ax[1].hist( SL2S_data_sigma , bins=_nbins_sg, density=True , histtype='step' , color=ER_col1, alpha = _ALPHA_)
    ax[2].hist( SL2S_data_Rein  , bins=_nbins_Re, density=True , histtype='step' , color=ER_col1, alpha = _ALPHA_)

    # ax[0].hist( SL2S_data_zl    , bins=5, density=True , histtype='step' , color=ER_col1, alpha = _ALPHA_, label='Sonnenfeld 2013')
    # ax[0].hist( SL2S_data_zs    , bins=5, density=True , histtype='step' , color=ER_col1, alpha = _ALPHA_, ls='--')
    # ax[1].hist( SL2S_data_sigma , bins=5, density=True , histtype='step' , color=ER_col1, alpha = _ALPHA_)
    # ax[2].hist( SL2S_data_Rein  , bins=5, density=True , histtype='step' , color=ER_col1, alpha = _ALPHA_)

    ax[0].legend(fontsize=10)
    plt.show()


def compare_CASSOWARY(zl_array, zs_array, sigma_array, LENS_LIGHT = 1, PLOT_FOR_KEYNOTE = 1):
    CASS_data       = pd.read_csv('../galess/data/CASSOWARY/cassowary.csv')
    CASS_data_names = CASS_data['Name'].to_numpy()
    CASS_data_zl    = CASS_data['z_l'].to_numpy()
    CASS_data_zs    = CASS_data['z_s'].to_numpy()
    CASS_data_sigma = CASS_data['sigma'].to_numpy()

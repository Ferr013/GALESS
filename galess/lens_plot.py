import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os.path

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
                               PLOT_FOR_KEYNOTE = 1, LOG = 0, SMOOTH = 0, SAVE = 0):

    line_c, cmap_c, _col_, col_A, col_B, col_C, col_D, fn_prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    Ngal_zl_sigma_noLL, Ngal_zs_sigma_noLL, Ngal_zs_zl_noLL, P_zs_noLL, P_zl_noLL, P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH)
    Ngal_zl_sigma_LL, Ngal_zs_sigma_LL, Ngal_zs_zl_LL, P_zs_LL, P_zl_LL, P_sg_LL = ls.get_N_and_P_projections(matrix_LL, sigma_array, zl_array, zs_array, SMOOTH)
    _nbins_Re = np.arange(0  , 4  , 0.25)
    fig.suptitle(title, fontsize=25)
    ax[0,0].plot(zl_array, P_zl_LL, c=col_A, ls=':')
    ax[0,0].plot(zs_array, P_zs_LL, c=col_B, ls=':' , label='w/ lens light')
    ax[0,0].plot(zl_array, P_zl_noLL, c=col_A, ls='-')
    ax[0,0].plot(zs_array, P_zs_noLL, c=col_B, ls='-', label='No lens light')
    ax[0,0].set_ylabel(r'$P$', fontsize=20)
    ax[0,0].set_xlabel(r'$z$', fontsize=20)
    ax[1,1].plot(sigma_array, P_sg_LL, c=col_C, ls = ':')
    ax[1,1].plot(sigma_array, P_sg_noLL, c=col_C, ls = '-')
    ax[1,1].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
    ax[0,1].hist(np.ravel(Theta_E_LL), weights=np.ravel(matrix_LL), bins=_nbins_Re, range=(0, 3), density=True, histtype='step', color=col_D, ls = ':')
    ax[0,1].hist(np.ravel(Theta_E_noLL), weights=np.ravel(matrix_noLL), bins=_nbins_Re, range=(0, 3), density=True, histtype='step', color=col_D, ls = '-')
    ax[0,1].set_xlabel(r'$\Theta_E$ [arcsec]', fontsize=20)
    _sigma, _zl = np.meshgrid(sigma_array, zl_array)
    ax[0,0].legend(fontsize=20)
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
    if(np.sum(matrix_noLL))>10_000:
        ax[1,0].legend([f'#Lenses w/ LL: {np.sum(matrix_LL):.1e}', f'#Lenses no LL: {np.sum(matrix_noLL):.1e}'], fontsize=20)
    else:
        ax[1,0].legend([f'#Lenses w/ LL: {np.sum(matrix_LL):.0f}', f'#Lenses no LL: {np.sum(matrix_noLL):.0f}'], fontsize=20)
    ax[1,0].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
    ax[1,0].set_ylabel(r'$z_l$', fontsize=20)
    ax[0,0].set_xlim((0,5.2))
    ax[1,1].set_xlim((100,400))
    ax[1,0].set_xlim((100,400))
    ax[1,0].set_ylim((0,2.5))
    if(LOG):
        ax[0,0].set_yscale('log')
        ax[0,1].set_yscale('log')
        ax[0,0].set_ylim((1e-3,2))
        ax[0,1].set_ylim((1e-3,2))    
    plt.tight_layout()
    if (SAVE):
        folderpath = 'img/'+utils.remove_spaces_from_string(title)
        if not os.path.exists(folderpath): os.makedirs(folderpath)
        plt.savefig(folderpath+'/'+fn_prefix+'corner_plts.jpg', dpi=200)



def compare_z_distributions_surveys(ax, title, color,
                                        zl_array, zs_array, sigma_array, matrix_LL, matrix_noLL,
                                        PLOT_FOR_KEYNOTE = 1, SMOOTH = 0):
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
    _n, __n, ___n, P_zs_noLL, P_zl_noLL, P_sg_noLL = ls.get_N_and_P_projections(matrix_noLL, sigma_array, zl_array, zs_array, SMOOTH)
    _ , __  , ___, P_zs_LL  , P_zl_LL  , P_sg_LL   = ls.get_N_and_P_projections(matrix_LL  , sigma_array, zl_array, zs_array, SMOOTH)
    ax[0].plot(sigma_array, P_sg_noLL, c=color, ls = '-', label=title)
    ax[1].plot(sigma_array, P_sg_LL  , c=color, ls = ':')
    ax[0].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
    ax[1].set_xlabel(r'$\sigma$ [km/s]', fontsize=20)
    ax[0].set_ylabel(r'$dP/d\sigma$', fontsize=20)

def compare_z_distributions_surveys(surveys_selection, sigma_array, zl_array, zs_array, cmap_c = cm.cool):
    _col_  = iter(cmap_c(np.linspace(0, 1, len(surveys_selection))))
    _PLOT_FOR_KEYNOTE = 1
    set_plt_param(PLOT_FOR_KEYNOTE = _PLOT_FOR_KEYNOTE)
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
                                                        PLOT_FOR_KEYNOTE = 1, SMOOTH = 1)
    ax[0].set_xlim((0,5.2))
    ax[1].set_xlim((0,5.2))
    ax[0].legend()
    plt.show()

def compare_sigma_distributions_surveys(surveys_selection, sigma_array, zl_array, zs_array, cmap_c = cm.cool):
    _col_  = iter(cmap_c(np.linspace(0, 1, len(surveys_selection))))
    _PLOT_FOR_KEYNOTE = 1
    set_plt_param(PLOT_FOR_KEYNOTE = _PLOT_FOR_KEYNOTE)
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
                                                        PLOT_FOR_KEYNOTE = 1, SMOOTH = 1)
    ax[0].set_xlim((100,400))
    ax[1].set_xlim((100,400))
    ax[0].legend()
    plt.show()

    
def plot_angular_separation(survey_title, zs_array, cmap_c = cm.cool, SPLIT_REDSHIFTS = 0, RANDOM_FIELD = 0, PLOT_ACF = 0):
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

    if(RANDOM_FIELD): 
        th_r_array   = np.linspace(0,10,100) #arcsec
        RR = (np.power(th_r_array+np.diff(th_r_array)[0],2)-np.power(th_r_array,2))/(10**2)
        P_rnd_galpos = np.cumsum(RR) 
    if(PLOT_ACF):
        #Barone_Nugent angualr two point correlation function https://iopscience.iop.org/article/10.1088/0004-637X/793/1/17/pdf
        th_r_array   = np.linspace(0,10,100) #arcsec
        A_w, beta    = 0.4, 0.6
        omega_galpos = (A_w*np.power(th_r_array+np.diff(th_r_array)[0], -beta))
        RR = (np.power(th_r_array+np.diff(th_r_array)[0],2)-np.power(th_r_array,2))/(10**2)
        #TODO: it should come from \omega = (DD-2DR-RR)/RR -- how to find DR?
        #TODO: or should I look at the integral definition? See [Eq.88] https://ned.ipac.caltech.edu/level5/March01/Strauss/Strauss5.html
        P_rnd_galpos = np.cumsum((1+omega_galpos)*RR) 
    
    _PLOT_FOR_KEYNOTE = 1
    set_plt_param(PLOT_FOR_KEYNOTE = _PLOT_FOR_KEYNOTE)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.15, hspace=.2)   
    ax[0].set_ylabel(r'$P(<\theta$)', fontsize=20)
    ax[0].set_xlabel(r'$\theta$ [arcsec]', fontsize=20)
    ax[1].set_xlabel(r'$\theta$ [arcsec]', fontsize=20)
    ax[0].set_xlim((0,10))
    ax[1].set_xlim((0,10))
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
    if(RANDOM_FIELD or PLOT_ACF): 
        ax[0].plot(th_r_array, P_rnd_galpos, 'g:')
        ax[1].plot(th_r_array, P_rnd_galpos, 'g:')
    plt.legend(fontsize=15, loc='lower right')
    plt.tight_layout()
    plt.show()
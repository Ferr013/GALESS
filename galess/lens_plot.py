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

def get_Harikane_ACF_small_scale(z=3):
    if(z==1):
        return [[1.9963236274483052, 0.4783824964943629 ],
                [2.896634936533222 , 0.37413962607747775],
                [4.202972824736602 , 0.32034502042509283],
                [5.970098880561779 , 0.24414084840368558],
                [8.5708797932537   , 0.18367284109722465],
                [12.304650542654327, 0.11380983017100621],
                [17.85385904050169 , 0.08025911785892756],
                [26.182665465840035, 0.07236866510323885]]
    if(z==3):
        return np.array([[1.9716086603186689, 2.7345961868755966],
                        [2.8456616883636414 , 1.5292060917860244],
                        [4.107199672835568  , 0.8177515184528737],
                        [6.037769010610303  , 0.47820228886038607],
                        [8.714431166023571  , 0.29903922282111883],
                        [12.34904764254925  , 0.22867742907131222],
                        [17.823624166116165 , 0.20449354873797368],
                        [25.72518850120658  , 0.1462340619871049],
                        [37.12967224032399  , 0.11958312623948672]])
    return 0

def plot_Harikane_ACF_small_scale():
    data = get_Harikane_ACF_small_scale()
    x, y = data[:,0], data[:,1]
    fit  = np.polyfit(x, y, 3)
    pol  = np.poly1d(fit)
    xp   = np.linspace(1, 20, 100)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex=False, sharey=False)
    ax.scatter(x, y)
    ax.plot(xp, pol(xp), c='g')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim((1,20))
    plt.show()
    
def plot_angular_separation(survey_title, zs_array, cmap_c = cm.cool, SPLIT_REDSHIFTS = 0, PLOT_ACF = 0, A_w = 0.4, beta = 0.6):
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
    th_r_array    = np.linspace(0,12,100) #arcsec
    th_r_radar    = th_r_array*rad_to_arcsec

    data = get_Harikane_ACF_small_scale()
    x, y = data[:,0], data[:,1]
    fit  = np.polyfit(x, y, 3)
    pol  = np.poly1d(fit) # A_w theta^-beta
        
    if(PLOT_ACF):        
        #ACF Integral definition - See [Eq.1] https://articles.adsabs.harvard.edu/pdf/1977ApJ...217..385G
        RR = np.pi*(np.power(th_r_array+np.diff(th_r_array)[0],2)-np.power(th_r_array,2))#/(np.pi*10**2)
        ''' THIS WORKS ONLY AT LARGE SCALE >100"
        #Barone_Nugent angualr two point correlation function 
        #https://iopscience.iop.org/article/10.1088/0004-637X/793/1/17/pdf
        CR = 2*np.pi*A_w*(np.power(th_r_radar+np.diff(th_r_radar)[0], 2-beta)-np.power(th_r_radar, 2-beta))/(2-beta)
        '''
        dth_r = np.diff(th_r_array)[0]
        intgr = np.zeros(0)
        for th_r in th_r_array:
            xint  = np.array([th_r, th_r+dth_r])
            yint  = np.array([pol(xint[0])*xint[0], pol(xint[1])*(xint[1])])
            intgr = np.append(intgr, np.trapz(yint, xint))
        CR = 2*np.pi*intgr
        P_Rgalpos = np.cumsum(RR+CR)/np.sum(RR+CR) 
        P_rnd_rnd = np.cumsum(RR)/np.sum(RR) 
    
    _PLOT_FOR_KEYNOTE = 1
    set_plt_param(PLOT_FOR_KEYNOTE = _PLOT_FOR_KEYNOTE)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=.15, hspace=.2)   
    ax[0].set_ylabel(r'$P(<\theta$)', fontsize=20)
    ax[0].set_xlabel(r'$\theta$ [arcsec]', fontsize=20)
    ax[1].set_xlabel(r'$\theta$ [arcsec]', fontsize=20)
    ax[0].set_xlim((0,12))
    ax[1].set_xlim((0,12))
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

    plt.legend(fontsize=15, loc='lower right')
    plt.tight_layout()
    plt.show()
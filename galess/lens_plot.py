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
        'legend.fontsize': 10,
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

    ax[0,0].set_xlim((0,4.2))
    ax[1,1].set_xlim((100,400))
    ax[1,0].set_xlim((100,400))
    ax[1,0].set_ylim((0,2.5))
    if(LOG):
        ax[0,0].set_yscale('log')
        ax[0,2].set_yscale('log')
        ax[0,0].set_ylim((1e-3,2))
        ax[0,2].set_ylim((1e-3,2))    
    plt.tight_layout()
    if (SAVE):
        folderpath = 'img/'+utils.remove_spaces_from_string(title)
        if not os.path.exists(folderpath): os.makedirs(folderpath)
        plt.savefig(folderpath+'/'+fn_prefix+'corner_plts.jpg', dpi=200)

"""Survey selections and figures for the lens populations in Ferrami & Wyithe (2024)."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from galess.LensStat import lens_stat as ls
from galess.RareLensStat import rare_lens_stat as rls
from galess.Utils import ls_utils as utils
from galess.Plots.lens_plot import (
    _survey_results, compare_ALL_distributions_surveys, model_response_m_cut,
    plot_ALL_distributions, plot_effect_vel_disp_function, plot_R_distribution_in_ax,
    plot_s_distribution_in_ax, plot_z_distribution_in_ax, set_plt_param
)


def _finish_plot(name=None):
    if name:
        Path('img').mkdir(exist_ok=True)
        plt.savefig(Path('img') / name, dpi=200, bbox_inches='tight')
    plt.show()


def plot_Lens_Fraction(m_lim=28.5, mu_arc_SIE=3, M_array=0, zs_array_plot=0,
    schechter_plot=ls.schechter_LF, PLOT_FOR_KEYNOTE=0, SAVE=0):
    """Plot the lensed, arc, counterimage, cusp and quad fractions versus source magnitude."""
    if np.isscalar(M_array) and M_array == 0:
        M_array = np.linspace(-14, -26, 37)
    if np.isscalar(zs_array_plot) and zs_array_plot == 0:
        zs_array_plot = np.arange(1, 10)
    _, cmap, *_, prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
    for zs, color in zip(zs_array_plot, cmap(np.linspace(0, 1, len(zs_array_plot)))):
        limit = m_lim - 5 * np.log10(ls.cosmo_luminosity_distance(zs) * 1e5)
        fraction = ls.Fraction_lensed_SIE(1, M_array, schechter_plot, ls.Phi_vel_disp_Mason, zs)
        arc = ls.Fraction_1st_image_arc_SIE(mu_arc_SIE, M_array, schechter_plot, zs)
        axes[0, 0].plot(M_array, fraction, color=color)
        axes[0, 1].plot(M_array, fraction * arc, color=color)
        for image, ax in zip((2, 3, 4), (axes[1, 0], axes[1, 1], axes[1, 1])):
            visible = ls.Fraction_Nth_image_above_Mlim_SIE(
                image, M_array, limit, schechter_plot, zs)
            ax.plot(M_array, fraction * visible, color=color, ls=':' if image == 3 else '-',
                label=f'$z={zs}$' if image == 4 else None)
    for ax, label in zip(axes.flat, ('Lensed fraction', 'Arc fraction', 'Second image',
        'Cusp (dotted) and quad')):
        ax.set(xlabel=r'$M_{AB,1}$', ylabel=label, xlim=(-24, -16), ylim=(7e-4, 1.2), yscale='log')
    axes[1, 1].legend()
    fig.tight_layout()
    _finish_plot(prefix + 'frac_lens_SIE.png' if SAVE else None)


def _saved_title(title):
    # The plot-only SIE pair is written when there are no existing survey results.
    if Path(utils._survey_result_path(title, 'matrix_LL')).is_file():
        return title
    return title + ' SIE lens light'


def get_distributions_from_title(title, sigma_array, zl_array, zs_array, LENS_LIGHT=1,
    SMOOTH=1, PRINT=0, zl_lim=(0., 99), zs_lim=(0., 99), sg_lim=(0, 1000), Rein_lim=0,
    results=None):
    """Apply open redshift/dispersion cuts and a minimum Einstein radius in arcsec.

    Return the selected count grid, original radii and magnitude tensor, and the
    normalized source-redshift, lens-redshift and velocity-dispersion projections.
    """
    if results is None:
        results = utils.load_pickled_files(_saved_title(title))
    selected = ((zs_array > zs_lim[0]) & (zs_array < zs_lim[1]))[:, None, None]
    selected = selected & ((sigma_array > sg_lim[0]) & (sigma_array < sg_lim[1]))[None, :, None]
    selected = selected & ((zl_array > zl_lim[0]) & (zl_array < zl_lim[1]))
    matrix, theta, prob = results[:3] if LENS_LIGHT else results[3:]
    if PRINT:
        for counts, radius, light in ((results[0], results[1], True),
            (results[3], results[4], False)):
            number = counts[selected & (radius >= Rein_lim)].sum()
            print(f'{title} (lens light {light})  {number:g}')
    matrix = np.where(selected & (theta >= Rein_lim), matrix, 0)
    *_, P_zs, P_zl, P_sg = ls.get_N_and_P_projections(
        matrix, sigma_array, zl_array, zs_array, SMOOTH)
    return matrix, theta, prob, P_zs, P_zl, P_sg


def plot_survey_ALL_distributions(surveys, READ_FILES=1, _PLOT_FOR_KEYNOTE=0):
    """Plot each survey on the standard population grid."""
    from galess.ComputeSurveys.compute_surveys import sigma_array, zl_array, zs_array
    for title in surveys:
        LL, theta_LL, _, noLL, theta_noLL, _ = _survey_results(
            title, sigma_array, zl_array, zs_array, read=READ_FILES)
        plot_ALL_distributions(title, zl_array, zs_array, sigma_array, theta_LL, LL,
            theta_noLL, noLL, PLOT_FOR_KEYNOTE=_PLOT_FOR_KEYNOTE)


def plot_z_sigma_distributions_double_lenses(title, zl_array, zs_array, sigma_array,
    matrix_LL, matrix_noLL, PLOT_FOR_KEYNOTE=0, SMOOTH=1, SAVE=0):
    """Show both source-redshift distributions of a double-source-plane forecast."""
    _, _, _, c1, c2, c3, _, prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for matrix, style, label in ((matrix_LL, ':', 'With lens light'),
        (matrix_noLL, '-', 'Subtracted lens light')):
        *_, near, far, lens, velocity = rls.get_N_and_P_projections_double_lens(
            matrix, sigma_array, zl_array, zs_array, SMOOTH)
        ax[0].plot(zl_array, lens, c=c1, ls=style)
        ax[0].plot(zs_array, near, c=c2, ls=style, label=label)
        ax[0].plot(zs_array, far, c=c3, ls=style)
        ax[1].plot(sigma_array, velocity, c=c1, ls=style)
    ax[0].set(xlabel='$z$', ylabel='$dP/dz$')
    ax[1].set(xlabel=r'$\sigma$ [km/s]', ylabel=r'$dP/d\sigma$')
    ax[0].legend()
    fig.suptitle(title)
    _finish_plot(prefix + 'DBL_corner_plts.png' if SAVE else None)


def plot_z_sigma_distributions(fig, ax, title, zl_array, zs_array, sigma_array,
    Theta_E_LL, matrix_LL, Theta_E_noLL, matrix_noLL, PLOT_FOR_KEYNOTE=0, CONTOUR=1,
    LOG=0, SMOOTH=0, SAVE=0, LEGEND=1, DOUBLE_LENS=0):
    """Draw the joint redshift and velocity distributions on a supplied 2-by-2 grid."""
    _, cmap, _, c1, c2, c3, c4, prefix = set_plt_param(PLOT_FOR_KEYNOTE)
    for matrix, radii, style in ((matrix_LL, Theta_E_LL, ':'), (matrix_noLL, Theta_E_noLL, '-')):
        if DOUBLE_LENS:
            proj = rls.get_N_and_P_projections_double_lens(matrix, sigma_array, zl_array,
                zs_array, SMOOTH)
            vel_lens, source_lens = proj[0], proj[2]
            near, far, lens, velocity = proj[-4:]
            # Sum over the other source plane before weighting each radius histogram.
            radius_weights = ((radii[0], matrix.sum(axis=1), c3),
                (radii[1], matrix.sum(axis=0), c4))
        else:
            vel_lens, _, source_lens, near, lens, velocity = ls.get_N_and_P_projections(
                matrix, sigma_array, zl_array, zs_array, SMOOTH)
            radius_weights = ((radii, matrix, c4),)
        if CONTOUR and not LOG:
            ax[0, 0].contour(zl_array, zs_array, source_lens, cmap=cmap, linestyles=style)
            ax[0, 0].set(xlabel='$z_l$', ylabel='$z_s$')
        else:
            ax[0, 0].plot(zl_array, lens, c=c1, ls=style)
            ax[0, 0].plot(zs_array, near, c=c2, ls=style)
            if DOUBLE_LENS:
                ax[0, 0].plot(zs_array, far, c=c3, ls=style)
            ax[0, 0].set(xlabel='$z$', ylabel='$dP/dz$')
        for theta, weight, color in radius_weights:
            ax[0, 1].hist(theta.ravel(), np.arange(0, 4, .25), weights=weight.ravel(),
                histtype='step', density=True, ls=style, color=color)
        ax[1, 0].contour(sigma_array, zl_array, vel_lens.T, cmap=cmap, linestyles=style)
        label = 'With lens light' if style == ':' else 'Subtracted lens light'
        ax[1, 1].plot(sigma_array, velocity, color=c3, ls=style,
            label=f'{label} ({matrix.sum():.3g})')
    if LOG:
        ax[0, 0].set_yscale('log')
        ax[0, 1].set_yscale('log')
    ax[0, 1].set(xlabel=r'$\Theta_E$ [arcsec]', ylabel=r'$dP/d\Theta_E$')
    ax[1, 0].set(xlabel=r'$\sigma$ [km/s]', ylabel='$z_l$')
    ax[1, 1].set(xlabel=r'$\sigma$ [km/s]', ylabel=r'$dP/d\sigma$')
    if LEGEND:
        ax[1, 1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    _finish_plot(prefix + ('DBL_' if DOUBLE_LENS else '') + 'corner_plts.png' if SAVE else None)


def single_compare_z_distributions_surveys(ax, title, color, zl_array, zs_array, sigma_array,
    matrix_LL, matrix_noLL, PLOT_FOR_KEYNOTE=0, SMOOTH=1, PLOT_ALL=0):
    """Overlay lens and source redshifts, with separate axes for the two light treatments."""
    for a, m in ([(ax, matrix_noLL)] if PLOT_ALL else zip(ax, (matrix_noLL, matrix_LL))):
        plot_z_distribution_in_ax(a, title, color, zl_array, zs_array, sigma_array, m, SMOOTH)
        a.set(xlabel='$z$', ylabel='$dP/dz$')


def single_compare_sigma_distributions_surveys(ax, title, color, zl_array, zs_array, sigma_array,
    matrix_LL, matrix_noLL, PLOT_FOR_KEYNOTE=0, SMOOTH=1, PLOT_ALL=0):
    """Overlay the lens velocity distributions of one survey."""
    for a, m in ([(ax, matrix_noLL)] if PLOT_ALL else zip(ax, (matrix_noLL, matrix_LL))):
        plot_s_distribution_in_ax(a, title, color, zl_array, zs_array, sigma_array, m, SMOOTH)
        a.set(xlabel=r'$\sigma$ [km/s]', ylabel=r'$dP/d\sigma$')


def single_compare_EinRing_distributions_surveys(ax, title, color, zl_array, zs_array,
    sigma_array, matrix_LL, matrix_noLL, PLOT_FOR_KEYNOTE=0, SMOOTH=1, PLOT_ALL=0):
    """Weight each lens Einstein radius by the corresponding survey count."""
    theta = np.zeros_like(matrix_LL)
    for izs, zs in enumerate(zs_array):
        for izl in np.flatnonzero((zl_array > 0) & (zl_array < zs)):
            theta[izs, :, izl] = ls.Theta_E(sigma_array, zl_array[izl], zs)
    for a, m in ([(ax, matrix_noLL)] if PLOT_ALL else zip(ax, (matrix_noLL, matrix_LL))):
        plot_R_distribution_in_ax(a, title, color, np.arange(0, 4, .25), m, theta)
        a.set(xlabel=r'$\Theta_E$ [arcsec]', ylabel=r'$dP/d\Theta_E$')


def _compare_marginals(plot, titles, sigma, zl, zs, keynote):
    _, cmap, *_ = set_plt_param(keynote)
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(11, 5))
    for title, color in zip(titles, cmap(np.linspace(0, 1, len(titles) + 1))):
        key = _saved_title(title)
        noLL = utils._load_survey_result(key, 'matrix_noLL')
        LL = utils._load_survey_result(key, 'matrix_LL')
        plot(ax, title, color, zl, zs, sigma, LL, noLL)
    ax[0].set_title('Subtracted lens light')
    ax[1].set_title('With lens light')
    ax[0].legend()
    fig.tight_layout()
    plt.show()


def compare_z_distributions_surveys(surveys_selection, sigma_array, zl_array, zs_array,
    PLOT_FOR_KEYNOTE=0):
    _compare_marginals(single_compare_z_distributions_surveys, surveys_selection,
        sigma_array, zl_array, zs_array, PLOT_FOR_KEYNOTE)


def compare_sigma_distributions_surveys(surveys_selection, sigma_array, zl_array, zs_array,
    PLOT_FOR_KEYNOTE=0):
    _compare_marginals(single_compare_sigma_distributions_surveys, surveys_selection,
        sigma_array, zl_array, zs_array, PLOT_FOR_KEYNOTE)


def _search_catalog(name):
    path = Path(utils.BASEPATH) / 'LENS_SEARCHES' / name
    return pd.read_csv(path, sep=';' if path.suffix == '.tsv' else ',')


def _faure_catalogs():
    best = _search_catalog('FAURE_2008/FAURE_A.csv')
    full = pd.concat((best, _search_catalog('FAURE_2008/FAURE_B.csv')), ignore_index=True)
    return full, best


def _des_catalogs():
    phot = _search_catalog('JACOBS_CNN/JACOBS_2019_DES_CNN.tsv')
    spec = _search_catalog('AGEL/AGEL.csv')
    # The spectroscopic catalogue uses a large sentinel for unknown redshifts.
    return phot, spec.where(spec[['z_l', 'z_s']] < 100)


def _catalog_hist(ax, values, color, density, bins='fd', limits=None, label=None, **kwargs):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    ax.hist(values, bins=bins, range=limits, density=density, histtype='step', lw=3,
        color=color, label=label, **kwargs)


def _search_models(title, sigma, zl, zs, light, smooth, keynote, simple=False,
    threshold=(3, 3), **selection):
    models = [('Mason', ls.Phi_vel_disp_Mason, 'w' if keynote else 'k', threshold[0])]
    if not simple:
        models.append(('Geng', ls.Phi_vel_disp_Geng, 'r', threshold[1]))
    for name, vdf, color, mu in models:
        key = title if name == 'Mason' else title + ' VDF Geng'
        if mu == 4:
            key += ' mu4'
        elif mu == 5:
            key += '_muarc5'
        grids = _survey_results(key, sigma, zl, zs, survey=title, VDF=vdf,
            arc_mu_threshold=mu)
        selected = get_distributions_from_title(key, sigma, zl, zs, LENS_LIGHT=light,
            SMOOTH=smooth, PRINT=True, results=grids, **selection)
        yield f'{title}\n{name} VDF', color, selected


def _plot_lens_magnitudes(ax, m_obs, zl, sigma, matrix, color, density):
    from scipy.integrate import simpson

    counts = ls.get_len_magnitude_distr(m_obs, zl, sigma, matrix, obs_band='sdss_i0')
    if density:
        area = simpson(counts, x=m_obs)
        if area:
            counts = counts / area
    ax.plot(m_obs, counts, color=color)
    ax.set(xlabel=r'$m_i^{\mathrm{lens}}$ [mag]',
        ylabel=r'$dP/dm$' if density else 'Lenses per magnitude bin', xlim=(15, 25))


def compare_COSMOS_HST_Faure(zl_array, zs_array, sigma_array, M_array_UV, mag_cut,
    ONLY_FULL_SAMPLE=1, LENS_LIGHT=1, __MAG_OVER_ARCSEC_SQ__=0, PLOT_FOR_KEYNOTE=0,
    SMOOTH=1, SAVE=0, SIMPLE=0):
    """Compare the COSMOS lens redshifts, arc radii and F814W magnitudes to Faure et al."""
    set_plt_param(PLOT_FOR_KEYNOTE)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    fig.subplots_adjust(wspace=.23)
    colors = ('darkorange', 'lime') if PLOT_FOR_KEYNOTE else ('forestgreen', 'orange')
    m_obs = np.linspace(15, 30, 31)
    for i, (data, name, color) in enumerate(zip(_faure_catalogs(), ('Full', 'Best'), colors)):
        if SIMPLE and i:
            break
        label = f'Faure et al. 2008\n{name} sample ({len(data)})'
        _catalog_hist(ax[0], data['z_l'], color, True, limits=(0, 1.5), label=label)
        # Convert the measured arc separation to the Einstein radius as in the paper.
        _catalog_hist(ax[1], data['R_arc'] / 1.5, color, True, limits=(0, 4))
        _catalog_hist(ax[2], data['mag_814W'], color, True, bins=m_obs)
    for label, color, (matrix, radius, _, pzs, pzl, _) in _search_models(
        'COSMOS HST i band', sigma_array, zl_array, zs_array, LENS_LIGHT, SMOOTH,
        PLOT_FOR_KEYNOTE, simple=SIMPLE, zl_lim=(.2, 1), sg_lim=(160, 1000)):
        ax[0].plot(zl_array, pzl, color=color, label=label)
        ax[0].plot(zs_array, pzs, color=color, ls='--')
        ax[1].hist(radius.ravel(), weights=matrix.ravel(), bins=np.arange(0, 4, .2),
            density=True, histtype='step', color=color)
        _plot_lens_magnitudes(ax[2], m_obs, zl_array, sigma_array, matrix, color, True)
    ax[0].set(xlabel='$z$', ylabel='$dP/dz$', xlim=(0, 4.2))
    ax[1].set(xlabel=r'$\Theta_E$ [arcsec]', ylabel=r'$dP/d\Theta_E$', xlim=(0, 4))
    ax[2].set(xlabel=r'$m_{\mathrm{F814W}}^{\mathrm{lens}}$ [mag]', ylim=(0, .4))
    ax[0].legend(fontsize=13)
    _finish_plot('COSMOS_HST_Faure.png' if SAVE else None)


def compare_JACOBS_CNN_DES(zl_array, zs_array, sigma_array, LENS_LIGHT=1,
    PLOT_FOR_KEYNOTE=0, SAVE=0, DENSITY=1, MU5=0, legend_size=18):
    """Compare DES forecasts with the Jacobs photometric and AGEL spectroscopic samples."""
    set_plt_param(PLOT_FOR_KEYNOTE)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(wspace=.23)
    cphot, cspec = ('darkorange', 'lime') if PLOT_FOR_KEYNOTE else ('forestgreen', 'orange')
    phot, spec = _des_catalogs()
    m_obs = np.linspace(15, 30, 31)
    _catalog_hist(ax[0], phot['z'], cphot, DENSITY, limits=(0, 1.5),
        bins='fd' if DENSITY else zl_array, label=f'Jacobs et al. 2019 ({len(phot)})')
    _catalog_hist(ax[0], spec['z_l'], cspec, DENSITY, limits=(0, 1.5),
        bins='fd' if DENSITY else zl_array, label=f'Tran et al. 2022 ({len(spec)})')
    _catalog_hist(ax[0], spec['z_s'], cspec, DENSITY, limits=(.5, 3), ls='--',
        bins='fd' if DENSITY else zs_array)
    _catalog_hist(ax[1], phot['imag'], cphot, DENSITY, bins=m_obs)
    for label, color, (matrix, _, _, pzs, pzl, _) in _search_models(
        'DES i band', sigma_array, zl_array, zs_array, LENS_LIGHT, True, PLOT_FOR_KEYNOTE,
        threshold=(5, 5) if MU5 else (3, 3), Rein_lim=2 if MU5 else 0):
        scale_l = 1 if DENSITY else matrix.sum() * abs(zl_array[1] - zl_array[0])
        scale_s = 1 if DENSITY else matrix.sum() * abs(zs_array[1] - zs_array[0])
        ax[0].plot(zl_array, pzl * scale_l, color=color, label=label)
        ax[0].plot(zs_array, pzs * scale_s, color=color, ls='--')
        _plot_lens_magnitudes(ax[1], m_obs, zl_array, sigma_array, matrix, color, DENSITY)
    ax[0].set(xlabel='$z$', ylabel='$dP/dz$' if DENSITY else 'Lenses per bin', xlim=(0, 3.6))
    if DENSITY:
        ax[0].set_ylim(0, 3.3)
        ax[1].set_ylim(0, .5)
    ax[0].legend(fontsize=legend_size)
    _finish_plot('JACOBS_CNN_DES.png' if SAVE else None)


def compare_SUGOHI(zl_array, zs_array, sigma_array, LENS_LIGHT=1, PLOT_FOR_KEYNOTE=0,
    SAVE=0, DENSITY=1, SMOOTH=1, legend_size=18,
    survey_title='SUBARU HSC SuGOHI i band 1400deg2'):
    """Compare SuGOHI candidates of each grade to the Subaru HSC i-band forecast."""
    set_plt_param(PLOT_FOR_KEYNOTE)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(wspace=.23)
    colors = ('darkorange', 'magenta', 'lime') if PLOT_FOR_KEYNOTE else (
        'forestgreen', 'orange', 'teal')
    m_obs = np.linspace(15, 30, 31)
    for grade, color in zip('ABC', colors):
        data = _search_catalog(f'SUGOHI/SuGOHI_grade{grade}.csv')
        zl, mag = data['z_lens'], data['lens_mag_i']
        zl, mag = zl[zl > 0], mag[mag > 0]
        alpha = .5 if grade == 'C' else 1
        _catalog_hist(ax[0], zl, color, DENSITY, bins='fd' if DENSITY else zl_array,
            limits=(0, 1.5) if grade == 'A' else None, alpha=alpha,
            label=f'SuGOHI grade {grade} ({len(zl)})')
        _catalog_hist(ax[1], mag, color, DENSITY, bins=m_obs, alpha=alpha)
    for label, color, (matrix, _, _, _, pzl, _) in _search_models(
        survey_title, sigma_array, zl_array, zs_array, LENS_LIGHT, SMOOTH,
        PLOT_FOR_KEYNOTE, zl_lim=(.2, 1), zs_lim=(0, 15), sg_lim=(230, 1000), Rein_lim=.2):
        scale = 1 if DENSITY else matrix.sum() * abs(zl_array[1] - zl_array[0])
        ax[0].plot(zl_array, pzl * scale, color=color, label=label)
        _plot_lens_magnitudes(ax[1], m_obs, zl_array, sigma_array, matrix, color, DENSITY)
    ax[0].set(xlabel='$z_l$', ylabel='$dP/dz$' if DENSITY else 'Lenses per bin', xlim=(0, 3.2))
    if DENSITY:
        ax[0].set_ylim(0, 5.5)
        ax[1].set_ylim(0, .55)
    ax[0].legend(fontsize=legend_size)
    _finish_plot('SUGOHI.png' if SAVE else None)


def compare_SL2S(zl_array, zs_array, sigma_array, LENS_LIGHT=1, PLOT_FOR_KEYNOTE=0,
    SMOOTH=1, SAVE=0):
    """Compare the full and grade A SL2S samples with the CFHTLS i-band forecast."""
    set_plt_param(PLOT_FOR_KEYNOTE)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    fig.subplots_adjust(wspace=.25)
    colors = ('darkorange', 'lime') if PLOT_FOR_KEYNOTE else ('forestgreen', 'orange')
    data = _search_catalog('SL2S_Sonnenfeld/redshifts_sigma.csv')
    radii = _search_catalog('SL2S_Sonnenfeld/data.csv')
    for sample, name, color in zip((data, data[data['Grade'] == 'A']), ('Full', 'Best'), colors):
        _catalog_hist(ax[0], sample['z_l'], color, True,
            label=f'Sonnenfeld et al. 2013\n{name} sample ({len(sample)})')
        _catalog_hist(ax[0], sample['z_s'], color, True, bins=np.arange(.5, 4, .5), ls='--')
        _catalog_hist(ax[1], sample['sigma'], color, True, bins=np.arange(100, 400, 25))
        matched = radii[radii['Name'].isin(sample['Name'])]
        _catalog_hist(ax[2], matched['R_Ein'], color, True, bins=np.arange(0, 4, .25))
    for label, color, (matrix, radius, _, pzs, pzl, psg) in _search_models(
        'CFHTLS i band', sigma_array, zl_array, zs_array, LENS_LIGHT, SMOOTH, PLOT_FOR_KEYNOTE,
        threshold=(4, 3), zl_lim=(.1, .8), sg_lim=(180, 1000)):
        ax[0].plot(zl_array, pzl, color=color, label=label)
        ax[0].plot(zs_array, pzs, color=color, ls='--')
        ax[1].plot(sigma_array, psg, color=color)
        ax[2].hist(radius.ravel(), weights=matrix.ravel(), bins=np.arange(0, 4, .2),
            density=True, histtype='step', color=color)
    ax[0].set(xlabel='$z$', ylabel='$dP/dz$', xlim=(0, 3.6))
    ax[1].set(xlabel=r'$\sigma$ [km/s]', ylabel=r'$dP/d\sigma$')
    ax[2].set(xlabel=r'$\Theta_E$ [arcsec]', ylabel=r'$dP/d\Theta_E$', xlim=(0, 4))
    ax[0].legend(fontsize=13)
    _finish_plot('SL2S_Sonnenfeld.png' if SAVE else None)


def compare_all_searches_z(zl_array, zs_array, sigma_array, LENS_LIGHT=1, PLOT_FOR_KEYNOTE=0,
    SAVE=0, DENSITY=1, SMOOTH=1):
    """Compare the lens redshift distributions of COSMOS, SL2S and DES side by side."""
    set_plt_param(PLOT_FOR_KEYNOTE)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=.13)
    primary, secondary = ('darkorange', 'lime') if PLOT_FOR_KEYNOTE else ('forestgreen', 'teal')
    full, best = _faure_catalogs()
    sl2s = _search_catalog('SL2S_Sonnenfeld/redshifts_sigma.csv')
    phot, spec = _des_catalogs()
    # The number comparison uses the search completeness adopted for each sample.
    panels = (
        ('COSMOS HST i band', ((full['z_l'], 'Faure 2008, full'),
            (best['z_l'], 'Faure 2008, best')), (.2, 1), 160, 1),
        ('CFHTLS i band', ((sl2s['z_l'], 'SL2S, full'),
            (sl2s.loc[sl2s['Grade'] == 'A', 'z_l'], 'SL2S, grade A')), (.1, .8), 180, .3 * .5),
        ('DES i band', ((phot['z'], 'Jacobs et al. 2019'),
            (spec['z_l'], 'Tran et al. 2022')), (0, 99), 0, .88 * .15))
    for i, (ax, (title, observations, zlim, sig_min, completeness)) in enumerate(zip(axes, panels)):
        for j, ((values, name), color) in enumerate(zip(observations, (primary, secondary))):
            if i == 2 and j == 0 and not DENSITY:
                continue  # Only the spectroscopically identified AGEL systems enter the count.
            _catalog_hist(ax, values, color, DENSITY, bins='fd' if DENSITY else zl_array,
                limits=None if i == 1 else (0, 1.5), label=f'{name} ({values.count()})')
        for label, color, (matrix, _, _, _, pzl, _) in _search_models(
            title, sigma_array, zl_array, zs_array, LENS_LIGHT, SMOOTH, PLOT_FOR_KEYNOTE,
            zl_lim=zlim, sg_lim=(sig_min, 1000)):
            scale = 1 if DENSITY else matrix.sum() * abs(zl_array[1] - zl_array[0]) * completeness
            ax.plot(zl_array, pzl * scale, color=color, label=label)
        ax.set(xlabel='$z_l$', xlim=(0, 3 if i == 0 else 2))
        if DENSITY:
            ax.set_ylim(0, (2, 2.5, 3)[i])
        ax.legend(fontsize=13)
    axes[0].set_ylabel('$dP/dz$' if DENSITY else 'Lenses per bin')
    suffix = 'DENS' if DENSITY else 'NUMBER'
    _finish_plot(f'Searches_z_{suffix}.png' if SAVE else None)

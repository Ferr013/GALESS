"""Module providing a set of utility functions."""

import os
from importlib.resources import files
import pickle
import numpy as np
import galess.LensStat.lens_stat as ls

BASEPATH = os.path.dirname(os.path.abspath(__file__)) + '/../data/'

def remove_spaces_from_string(string):
    '''
    Returns a string with no spaces.

            Parameters:
                    input: (string)
                        String with some spaces
            Returns:
                    output: (string)
                        String with no spaces
    '''
    return "".join(string.split())

def read_survey_params(title, VERBOSE = 0):
    '''
    Returns a dictionary with the input parameters for a given combination of survey and
    photometric band from a .param file in galess/data/surveys_params/.

            Parameters:
                    title: (string)
                        Title of the survey + photo band
                    VERBOSE: (boolean)
                        Flag to print the items in params
            Returns:
                    params: (dictionary)
                        Parameters of the requested survey and photometric band
    '''
    TITLE = remove_spaces_from_string(title)
    FPATH = BASEPATH + 'surveys_params/'+TITLE+'.param'
    params = {}
    with open(FPATH, "r") as file:
        for line in file:
            key, value = line.strip().split(": ")
            if key in ['limit', 'cut', 'area', 'seeing', 'exp_time_sec', 'pixel_arcsec', 'zero_point_m', 'sky_bckgnd_m']:
                value = float(value)
            params[key] = value
    if(VERBOSE):
        print("Survey params:")
        for key, value in params.items():
            print(f"{key}: {value}")
    return params

def _survey_result_path(title, field):
    name = remove_spaces_from_string(title)
    return os.path.join(BASEPATH, 'surveys_results', name, name + '_' + field + '.pkl')


def _load_survey_result(title, field):
    """Read one saved distribution without loading the magnitude tensors."""
    with open(_survey_result_path(title, field), 'rb') as pickle_file:
        return pickle.load(pickle_file)


def load_pickled_files(title, DOUBLE_LENS = 0):
    '''
    Returns the resulting distributions for a given combination of survey and photometric band.
    The results has the same data structure as lens_stat.calculate_num_lenses_and_prob().

            Parameters:
                    title: (string)
                        Title of the survey + photo band
                    DOUBLE_LENS: (boolean)
                        When true, each model returns separate radii for the two source planes.
            Returns:
                    Ngal_matrix_LL: ndarray(dtype=float, ndim=3)
                        (Model with Lens Light) Number of lensed galaxies per bin
                        of velocity dispersion, redshift of the lens and the source, integrated
                        over the source magnitude up to the magnitude limit of the survey.
                    Theta_E_mat_LL: ndarray(dtype=float, ndim=3)
                        (Model with Lens Light) Value of the Einstein Radius of a
                        SIS lens per bin of velocity dispersion, redshift of the
                        lens and of the source.
                    Ngal_tensor_LL: ndarray(dtype=float, ndim=4)
                        (Model with Lens Light) Number of lensed galaxies per bin
                        of velocity dispersion, redshift of the lens and the source,
                        and abs magnitude of the source.
                    Ngal_matrix_noLL: ndarray(dtype=float, ndim=3)
                        (Model with NO Lens Light) Number of lensed galaxies per bin
                        of velocity dispersion, redshift of the lens and the source, integrated
                        over the source magnitude up to the magnitude limit of the survey.
                    Theta_E_mat_noLL: ndarray(dtype=float, ndim=3)
                        (Model with NO Lens Light) Value of the Einstein Radius of a
                        SIS lens per bin of velocity dispersion, redshift of the
                        lens and of the source.
                    Ngal_tensor_noLL: ndarray(dtype=float, ndim=4)
                        (Model with NO Lens Light) Number of lensed galaxies per bin
                        of velocity dispersion, redshift of the lens and the source,
                        and abs magnitude of the source.
    '''
    if not os.path.isfile(_survey_result_path(title, 'matrix_LL')):
        raise ValueError('Files do not exist. Run the model on this survey.')
    result = []
    for model in ('LL', 'noLL'):
        result.append(_load_survey_result(title, 'matrix_' + model))
        radii = ('_z1', '_z2') if DOUBLE_LENS else ('',)
        for plane in radii:
            result.append(_load_survey_result(title, 'Theta_E_' + model + plane))
        result.append(_load_survey_result(title, 'prob_' + model))
    return tuple(result)


def save_pickled_files(
                    title,
                    temp_LL, Theta_E_LL, prob_LL,
                    temp_noLL, Theta_E_noLL, prob_noLL,
                    DOUBLE_LENS = 0):
    '''
    Saves the resulting distributions for a given combination of survey and photometric band
    in a picke file inside galess/data/surveys_results/.

            Parameters:
                    title: (string)
                        Title of the survey + photo band
                    Ngal_matrix_LL: ndarray(dtype=float, ndim=3)
                        (Model with Lens Light) Number of lensed galaxies per bin
                        of velocity dispersion, redshift of the lens and the source, integrated
                        over the source magnitude up to the magnitude limit of the survey.
                    Theta_E_mat_LL: ndarray(dtype=float, ndim=3)
                        (Model with Lens Light) Value of the Einstein Radius of a
                        SIS lens per bin of velocity dispersion, redshift of the
                        lens and of the source.
                    Ngal_tensor_LL: ndarray(dtype=float, ndim=4)
                        (Model with Lens Light) Number of lensed galaxies per bin
                        of velocity dispersion, redshift of the lens and the source,
                        and abs magnitude of the source.
                    Ngal_matrix_noLL: ndarray(dtype=float, ndim=3)
                        (Model with NO Lens Light) Number of lensed galaxies per bin
                        of velocity dispersion, redshift of the lens and the source, integrated
                        over the source magnitude up to the magnitude limit of the survey.
                    Theta_E_mat_noLL: ndarray(dtype=float, ndim=3)
                        (Model with NO Lens Light) Value of the Einstein Radius of a
                        SIS lens per bin of velocity dispersion, redshift of the
                        lens and of the source.
                    Ngal_tensor_noLL: ndarray(dtype=float, ndim=4)
                        (Model with NO Lens Light) Number of lensed galaxies per bin
                        of velocity dispersion, redshift of the lens and the source,
                        and abs magnitude of the source.
                    DOUBLE_LENS: (boolean)
                        When true, pass each model's radii as a (near, far) pair.
            Returns:
                    None
    '''
    os.makedirs(os.path.dirname(_survey_result_path(title, 'matrix_LL')), exist_ok=True)
    for model, matrix, radii, prob in (('LL', temp_LL, Theta_E_LL, prob_LL),
        ('noLL', temp_noLL, Theta_E_noLL, prob_noLL)):
        fields = [('matrix_' + model, matrix), ('prob_' + model, prob)]
        if DOUBLE_LENS:
            fields.extend([('Theta_E_' + model + '_z1', radii[0]),
                ('Theta_E_' + model + '_z2', radii[1])])
        else:
            fields.append(('Theta_E_' + model, radii))
        for name, value in fields:
            with open(_survey_result_path(title, name), 'wb') as pickle_file:
                pickle.dump(value, pickle_file)


def print_summary_surveys(surveys_selection):
    '''
    Prints a summary of some of the survey parameters and the total number of identifiable lenses
    for a list of surveys + photo bands.

            Parameters:
                    title: (list of strings)
                        Title of the survey + photo band
            Returns:
                    None
    '''
    print(f'|     Survey - Filter     | PSF/Seeing ["] | Area [deg^2] | m_cut [mag] | m_lim [mag] |   N_lenses (LL)   |   N_lenses (LL)   | N [deg^-1] |')
    print(f'|                         |                |              |             |             | VDF: Mason + 2015 |  VDF: Geng + 2021 |            |')
    print()
    if hasattr(surveys_selection, '__len__') is False:
        surveys_selection = [surveys_selection]
    for title in surveys_selection:
        survey_params = read_survey_params(title, VERBOSE = 0)
        limit    = survey_params['limit']
        cut      = survey_params['cut']
        area     = survey_params['area']
        seeing   = survey_params['seeing']
        exp_time_sec = survey_params['exp_time_sec']
        pixel_arcsec = survey_params['pixel_arcsec']
        zero_point_m = survey_params['zero_point_m']
        sky_bckgnd_m = survey_params['sky_bckgnd_m']
        photo_band   = survey_params['photo_band']
        try:
            matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = load_pickled_files(title)
            matrix_LG, Theta_E_LG, prob_LG, matrix_noLG, Theta_E_noLG, prob_noLG = load_pickled_files(title + ' VDF Geng')
        except ValueError:
            print('FILE do NOT exist - RUNNING MODEL')
            M_array     = np.linspace(-13 , -25 , 25)
            sigma_array = np.linspace(100 , 400 , 31)
            zl_array    = np.arange(0.  , 2.5 , 0.1)
            zs_array    = np.arange(0.  , 5.4 , 0.2)
            min_SNR     = 20
            arc_mu_thr  = 3
            VDF = ls.Phi_vel_disp_Mason
            matrix_noLL, Theta_E_noLL, prob_noLL = ls.calculate_num_lenses_and_prob(
                                                    sigma_array, zl_array, zs_array, M_array, limit, area,
                                                    seeing, min_SNR, exp_time_sec, sky_bckgnd_m, zero_point_m,
                                                    photo_band = photo_band, mag_cut=cut, arc_mu_threshold = arc_mu_thr,
                                                    Phi_vel_disp = VDF, LENS_LIGHT_FLAG = False)
            matrix_LL, Theta_E_LL, prob_LL = ls.calculate_num_lenses_and_prob(
                                                    sigma_array, zl_array, zs_array, M_array, limit, area,
                                                    seeing, min_SNR, exp_time_sec, sky_bckgnd_m, zero_point_m,
                                                    photo_band = photo_band, mag_cut=cut, arc_mu_threshold = arc_mu_thr,
                                                    Phi_vel_disp = VDF, LENS_LIGHT_FLAG = True)
            save_pickled_files(title,  matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL)

            VDF = ls.Phi_vel_disp_Geng
            matrix_noLG, Theta_E_noLG, prob_noLG = ls.calculate_num_lenses_and_prob(
                                                    sigma_array, zl_array, zs_array, M_array, limit, area,
                                                    seeing, min_SNR, exp_time_sec, sky_bckgnd_m, zero_point_m,
                                                    photo_band = photo_band, mag_cut=cut, arc_mu_threshold = arc_mu_thr,
                                                    Phi_vel_disp = VDF, LENS_LIGHT_FLAG = False)
            matrix_LG, Theta_E_LG, prob_LG = ls.calculate_num_lenses_and_prob(
                                                    sigma_array, zl_array, zs_array, M_array, limit, area,
                                                    seeing, min_SNR, exp_time_sec, sky_bckgnd_m, zero_point_m,
                                                    photo_band = photo_band, mag_cut=cut, arc_mu_threshold = arc_mu_thr,
                                                    Phi_vel_disp = VDF, LENS_LIGHT_FLAG = True)
            save_pickled_files(title + ' VDF Geng',  matrix_LG, Theta_E_LG, prob_LG, matrix_noLG, Theta_E_noLG, prob_noLG)
        N_LL, N_noLL = f'{np.sum(matrix_LL):.0f}', f'{np.sum(matrix_noLL):.0f}'
        N_LG, N_noLG = f'{np.sum(matrix_LG):.0f}', f'{np.sum(matrix_noLG):.0f}'
        if np.sum(matrix_noLL)>10_000:
            N_LL, N_noLL = f'{np.sum(matrix_LL):.1e}', f'{np.sum(matrix_noLL):.1e}'
            N_LG, N_noLG = f'{np.sum(matrix_LG):.1e}', f'{np.sum(matrix_noLG):.1e}'
        print(f'|{title:^25}|{seeing:16.3f}|{area:14.3f}|{cut:13.1f}|{limit:13.1f}|{N_noLL:>9} ({N_LL:^7})|{N_noLG:>9} ({N_LG:^7})| {(np.sum(matrix_LL)/area):>4.0f} - {(np.sum(matrix_LG)/area):<4.0f}|')

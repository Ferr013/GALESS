"""Module providing a set of utility functions."""

import os
from importlib.resources import files
import pickle
import numpy as np

# import galess.LensStat.lens_stat as ls
import sys
path_root = os.path.split(os.path.abspath(''))[0]
sys.path.append(str(path_root) + '/galess/LensStat/')
import lens_stat as ls


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

def load_pickled_files(title, DOUBLE_LENS = 0):
    '''
    Returns the resulting distributions for a given combination of survey and photometric band.
    The results has the same data structure as lens_stat.calculate_num_lenses_and_prob().

            Parameters:
                    title: (string)
                        Title of the survey + photo band
                    DOUBLE_LENS: (boolean)
                        Flag to extract a double lens file structure
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
    TITLE = remove_spaces_from_string(title)
    FOLDERPATH = BASEPATH + 'surveys_results/'+TITLE+'/'
    if os.path.exists(FOLDERPATH):
        FPATH = FOLDERPATH+TITLE+'_matrix_LL.pkl'
        if os.path.isfile(FPATH):
            with open(FPATH, 'rb') as pickle_file: _temp_LL = pickle.load(pickle_file)
            FPATH = FOLDERPATH+TITLE+'_prob_LL.pkl'
            with open(FPATH, 'rb') as pickle_file: _prob_LL = pickle.load(pickle_file)
            FPATH = FOLDERPATH+TITLE+'_matrix_noLL.pkl'
            with open(FPATH, 'rb') as pickle_file: _temp_noLL = pickle.load(pickle_file)
            FPATH = FOLDERPATH+TITLE+'_prob_noLL.pkl'
            with open(FPATH, 'rb') as pickle_file: _prob_noLL = pickle.load(pickle_file)
            if DOUBLE_LENS:
                FPATH = FOLDERPATH+TITLE+'_Theta_E_LL_z1.pkl'
                with open(FPATH, 'rb') as pickle_file: _Theta_E_LL_z1 = pickle.load(pickle_file)
                FPATH = FOLDERPATH+TITLE+'_Theta_E_noLL_z1.pkl'
                with open(FPATH, 'rb') as pickle_file: _Theta_E_noLL_z1 = pickle.load(pickle_file)
                FPATH = FOLDERPATH+TITLE+'_Theta_E_LL_z2.pkl'
                with open(FPATH, 'rb') as pickle_file: _Theta_E_LL_z2 = pickle.load(pickle_file)
                FPATH = FOLDERPATH+TITLE+'_Theta_E_noLL_z2.pkl'
                with open(FPATH, 'rb') as pickle_file: _Theta_E_noLL_z2 = pickle.load(pickle_file)
                return _temp_LL, _Theta_E_LL_z1, _Theta_E_LL_z2, _prob_LL, _temp_noLL, _Theta_E_noLL_z1, _Theta_E_noLL_z2, _prob_noLL
            FPATH = FOLDERPATH+TITLE+'_Theta_E_LL.pkl'
            with open(FPATH, 'rb') as pickle_file: _Theta_E_LL = pickle.load(pickle_file)
            FPATH = FOLDERPATH+TITLE+'_Theta_E_noLL.pkl'
            with open(FPATH, 'rb') as pickle_file: _Theta_E_noLL = pickle.load(pickle_file)
            return _temp_LL, _Theta_E_LL, _prob_LL, _temp_noLL, _Theta_E_noLL, _prob_noLL
    raise ValueError('Files do not exist. Run the model on this survey.')

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
                        Flag to extract a double lens file structure
            Returns:
                    None
    '''
    TITLE = remove_spaces_from_string(title)
    FOLDERPATH = BASEPATH + 'surveys_results/'+TITLE+'/'
    if not os.path.exists(FOLDERPATH):
        os.makedirs(FOLDERPATH)
    FPATH = FOLDERPATH+TITLE+'_matrix_LL.pkl'
    with open(FPATH, 'wb') as pickle_file:
        pickle.dump(temp_LL, pickle_file)
    FPATH = FOLDERPATH+TITLE+'_prob_LL.pkl'
    with open(FPATH, 'wb') as pickle_file:
        pickle.dump(prob_LL, pickle_file)
    FPATH = FOLDERPATH+TITLE+'_matrix_noLL.pkl'
    with open(FPATH, 'wb') as pickle_file:
        pickle.dump(temp_noLL, pickle_file)
    FPATH = FOLDERPATH+TITLE+'_prob_noLL.pkl'
    with open(FPATH, 'wb') as pickle_file:
        pickle.dump(prob_noLL, pickle_file)
    if DOUBLE_LENS:
        FPATH = FOLDERPATH+TITLE+'_Theta_E_LL_z1.pkl'
        with open(FPATH, 'wb') as pickle_file:
            pickle.dump(Theta_E_LL[0], pickle_file)
        FPATH = FOLDERPATH+TITLE+'_Theta_E_noLL_z1.pkl'
        with open(FPATH, 'wb') as pickle_file:
            pickle.dump(Theta_E_noLL[0], pickle_file)
        FPATH = FOLDERPATH+TITLE+'_Theta_E_LL_z2.pkl'
        with open(FPATH, 'wb') as pickle_file:
            pickle.dump(Theta_E_LL[1], pickle_file)
        FPATH = FOLDERPATH+TITLE+'_Theta_E_noLL_z2.pkl'
        with open(FPATH, 'wb') as pickle_file:
            pickle.dump(Theta_E_noLL[1], pickle_file)
    else:
        FPATH = FOLDERPATH+TITLE+'_Theta_E_LL.pkl'
        with open(FPATH, 'wb') as pickle_file:
            pickle.dump(Theta_E_LL, pickle_file)
        FPATH = FOLDERPATH+TITLE+'_Theta_E_noLL.pkl'
        with open(FPATH, 'wb') as pickle_file:
            pickle.dump(Theta_E_noLL, pickle_file)
    pass

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
    print(f'|     Survey - Filter     | PSF/Seeing ["] | Area [deg^2] | m_cut [mag] | m_lim [mag] | N [deg^-1] | N_lenses (LL)       |')
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
            matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = load_pickled_files(title)
        N_LL, N_noLL = f'{np.sum(matrix_LL):.0f}', f'{np.sum(matrix_noLL):.0f}'
        if np.sum(matrix_noLL)>10_000:
            N_LL, N_noLL = f'{np.sum(matrix_LL):.1e}', f'{np.sum(matrix_noLL):.1e}'
        print(f'|{title:^25}|{seeing:16.3f}|{area:14.3f}|{cut:13.1f}|{limit:13.1f}|{(np.sum(matrix_noLL)/area):12.0f}|{N_noLL:>9} ({N_LL:>9})|')
        print()

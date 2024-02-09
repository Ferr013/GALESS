import os
from importlib.resources import files
import pickle
import numpy as np

# BASEPATH = os.path.dirname(os.path.abspath(''))+'/GALESS/galess/'
# BASEPATH = '/../../data/'
# BASEPATH = files('data').joinpath('').read_text()
# BASEPATH = './../data/'
BASEPATH = os.path.dirname(os.path.abspath(__file__)) + '/../data/'


def remove_spaces_from_string(string):
    return "".join(string.split())

def read_survey_params(title, VERBOSE = 0):
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
            else:
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
    print(f'|     Survey - Filter     | PSF/Seeing ["] | Area [deg^2] | m_cut [mag] | m_lim [mag] | N [deg^-1] | N_lenses (LL)       |')
    print()
    if hasattr(surveys_selection, '__len__') == False: surveys_selection = [surveys_selection]
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
        matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = load_pickled_files(title)
        if(np.sum(matrix_noLL))>10_000:
            N_LL, N_noLL = f'{np.sum(matrix_LL):.1e}', f'{np.sum(matrix_noLL):.1e}'
        else:
            N_LL, N_noLL = f'{np.sum(matrix_LL):.0f}', f'{np.sum(matrix_noLL):.0f}'
        print(f'|{title:^25}|{seeing:16.3f}|{area:14.3f}|{cut:13.1f}|{limit:13.1f}|{(np.sum(matrix_noLL)/area):12.0f}|{N_noLL:>9} ({N_LL:>9})|')
        print()

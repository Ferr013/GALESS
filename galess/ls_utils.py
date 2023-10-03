import os
import pickle

def remove_spaces_from_string(string):
    return "".join(string.split())

def read_survey_params(title, VERBOSE = 0):
    TITLE = remove_spaces_from_string(title)
    FPATH = os.path.split(os.path.dirname(os.path.abspath('')))[0]+'/GALESS/galess/data/surveys_params/'+TITLE+'.param'
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

def load_pickled_files(title):
    TITLE = remove_spaces_from_string(title)
    FOLDERPATH = os.path.split(os.path.dirname(os.path.abspath('')))[0]+'/GALESS/galess/data/surveys_results/'+TITLE+'/'
    if os.path.exists(FOLDERPATH):
        FNAME = TITLE+'_matrix_LL.pkl'
        FPATH = FOLDERPATH+FNAME        
        if os.path.isfile(FPATH):
            with open(FPATH, 'rb') as pickle_file: _temp_LL = pickle.load(pickle_file)
            FPATH = FOLDERPATH+TITLE+'_Theta_E_LL.pkl'
            with open(FPATH, 'rb') as pickle_file: _Theta_E_LL = pickle.load(pickle_file)
            FPATH = FOLDERPATH+TITLE+'_prob_LL.pkl'
            with open(FPATH, 'rb') as pickle_file: _prob_LL = pickle.load(pickle_file)
            FPATH = FOLDERPATH+TITLE+'_matrix_noLL.pkl'
            with open(FPATH, 'rb') as pickle_file: _temp_noLL = pickle.load(pickle_file)
            FPATH = FOLDERPATH+TITLE+'_Theta_E_noLL.pkl'
            with open(FPATH, 'rb') as pickle_file: _Theta_E_noLL = pickle.load(pickle_file)
            FPATH = FOLDERPATH+TITLE+'_prob_noLL.pkl'
            with open(FPATH, 'rb') as pickle_file: _prob_noLL = pickle.load(pickle_file)
            return _temp_LL, _Theta_E_LL, _prob_LL, _temp_noLL, _Theta_E_noLL, _prob_noLL
    raise ValueError('Files do not exists. Run the model on this survey.')

def save_pickled_files(
                    title, 
                    temp_LL, Theta_E_LL, prob_LL,
                    temp_noLL, Theta_E_noLL, prob_noLL):
    TITLE = remove_spaces_from_string(title)
    FOLDERPATH = os.path.split(os.path.dirname(os.path.abspath('')))[0]+'/GALESS/galess/data/surveys_results/'+TITLE+'/'
    if not os.path.exists(FOLDERPATH):
        os.makedirs(FOLDERPATH)
    FPATH = FOLDERPATH+TITLE+'_matrix_LL.pkl'
    with open(FPATH, 'wb') as pickle_file:  
        pickle.dump(temp_LL, pickle_file)
    FPATH = FOLDERPATH+TITLE+'_Theta_E_LL.pkl'
    with open(FPATH, 'wb') as pickle_file:  
        pickle.dump(Theta_E_LL, pickle_file)
    FPATH = FOLDERPATH+TITLE+'_prob_LL.pkl'
    with open(FPATH, 'wb') as pickle_file:  
        pickle.dump(prob_LL, pickle_file)
    FPATH = FOLDERPATH+TITLE+'_matrix_noLL.pkl'
    with open(FPATH, 'wb') as pickle_file:  
        pickle.dump(temp_noLL, pickle_file)
    FPATH = FOLDERPATH+TITLE+'_Theta_E_noLL.pkl'
    with open(FPATH, 'wb') as pickle_file:  
        pickle.dump(Theta_E_noLL, pickle_file)
    FPATH = FOLDERPATH+TITLE+'_prob_noLL.pkl'
    with open(FPATH, 'wb') as pickle_file:  
        pickle.dump(prob_noLL, pickle_file)
    pass    
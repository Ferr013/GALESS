import os
import sys
import numpy as np
from tqdm import tqdm

import galess.LensStat.lens_stat as ls
import galess.Utils.ls_utils as utils

M_array     = np.linspace(-13 , -25 , 25)
sigma_array = np.linspace(100 , 400 , 31)
zl_array    = np.arange(0.  , 2.5 , 0.1)
zs_array    = np.arange(0.  , 5.4 , 0.2)
min_SNR     = 20
arc_mu_thr  = 3
surveys_titles = [
     'COSMOS Web F115W', 'COSMOS Web F150W', 'COSMOS Web F277W',
     'PEARLS NEP F115W', 'PEARLS NEP F150W', 'PEARLS NEP F277W',
     'JADES Deep F115W', 'JADES Deep F150W', 'JADES Deep F277W',
     'COSMOS HST i band',
     'EUCLID Wide VIS',
     'Roman HLWA J',
     'DES i band',
     'LSST i band',
     'SUBARU HSC SuGOHI i band',]

def Compute_SL_distributions(surveys = surveys_titles,
                              sigma_array = sigma_array, zl_array = zl_array,
                              zs_array = zs_array, M_array = M_array,
                              VDF = ls.Phi_vel_disp_Mason):
     for title in tqdm(surveys):
          print('Computing strong lensing statistics for ', title)
          survey_params = utils.read_survey_params(title, VERBOSE = 0)

          limit    = survey_params['limit']
          cut      = survey_params['cut']
          area     = survey_params['area']
          seeing   = survey_params['seeing']
          exp_time_sec = survey_params['exp_time_sec']
          zero_point_m = survey_params['zero_point_m']
          sky_bckgnd_m = survey_params['sky_bckgnd_m']
          photo_band   = survey_params['photo_band']

          if VDF == ls.Phi_vel_disp_Geng:
               title = title + ' VDF Geng'
          if VDF == ls.Phi_vel_disp_SDSS:
               title = title + ' VDF Choi'

          try:
               matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL = utils.load_pickled_files(title)
          except ValueError:
               print('FILE do NOT exist - RUNNING MODEL')
               matrix_noLL, Theta_E_noLL, prob_noLL = ls.calculate_num_lenses_and_prob(
                                                                           sigma_array, zl_array, zs_array, M_array, limit, area,
                                                                           seeing, min_SNR, exp_time_sec, sky_bckgnd_m, zero_point_m,
                                                                           photo_band = photo_band, mag_cut=cut, arc_mu_threshold = arc_mu_thr,
                                                                           Phi_vel_disp = VDF, LENS_LIGHT_FLAG = False, SIE_FLAG = True)

               matrix_LL, Theta_E_LL, prob_LL = ls.calculate_num_lenses_and_prob(
                                                                           sigma_array, zl_array, zs_array, M_array, limit, area,
                                                                           seeing, min_SNR, exp_time_sec, sky_bckgnd_m, zero_point_m,
                                                                           photo_band = photo_band, mag_cut=cut, arc_mu_threshold = arc_mu_thr,
                                                                           Phi_vel_disp = VDF, LENS_LIGHT_FLAG = True, SIE_FLAG = False)

               utils.save_pickled_files(title,  matrix_LL, Theta_E_LL, prob_LL, matrix_noLL, Theta_E_noLL, prob_noLL)

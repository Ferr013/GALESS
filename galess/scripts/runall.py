#!/usr/bin/python3
import sys
import galess.LensStat.lens_stat as ls
import galess.ComputeSurveys.compute_surveys as cs

# surveys_titles = [
#      'COSMOS Web F115W', 'COSMOS Web F150W', 'COSMOS Web F277W',
#      'PEARLS NEP F115W', 'PEARLS NEP F150W', 'PEARLS NEP F277W',
#      'JADES Deep F115W', 'JADES Deep F150W', 'JADES Deep F277W',
#      'COSMOS HST i band',
#      'DES i band',
#      'SUBARU HSC SuGOHI i band',
#      'EUCLID Wide VIS',
#      'Roman HLWA J',
#      'LSST i band', 'LSSTsinglevisit i band',
#      'CFHTLS i band']

surveys_titles = [
     'COSMOS Web F115W',
     'DES i band',
     'SUBARU HSC SuGOHI i band',]

cs.Compute_SL_distributions(surveys = [surveys_titles], VDF = ls.Phi_vel_disp_Mason)

#!/usr/bin/python3
import sys
import galess.LensStat.lens_stat as ls
import galess.ComputeSurveys.compute_surveys as cs

surveys_titles = [
    'COSMOS HST i band',
    'CFHTLS i band',
    'SUBARU HSC SuGOHI i band',
    'DES i band',
    'COSMOS Web F150W', 'COSMOS Web F115W', 'COSMOS Web F277W',
    'PEARLS NEP F115W', 'PEARLS NEP F150W', 'PEARLS NEP F277W',
    'EUCLID Wide VIS', 'EUCLID Wide Y', 'EUCLID Wide J', 'EUCLID Wide H',
    'Roman HLWA J',
    'LSST i band', 'LSSTsinglevisit i band']

cs.Compute_SL_distributions(surveys = surveys_titles, VDF = ls.Phi_vel_disp_Mason)
cs.Compute_SL_distributions(surveys = surveys_titles, VDF = ls.Phi_vel_disp_Geng)

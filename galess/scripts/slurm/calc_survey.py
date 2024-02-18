#!/usr/bin/python3
import sys
import galess.ComputeSurveys.compute_surveys as cs

survey = sys.argv[1]
print('Computing strong lensing statistics for ', survey)
cs.Compute_SL_distributions(surveys = [survey])

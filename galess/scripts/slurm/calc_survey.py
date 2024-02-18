#!/usr/bin/python3
import sys
import galess.ComputeSurveys.compute_surveys as cs

cs.Compute_SL_distributions(surveys = sys.argv[1])

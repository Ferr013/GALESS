#!/bin/bash
#
#SBATCH --job-name=SL_surveys
#
#SBATCH --ntasks=1
#SBATCH --time=20:00
#SBATCH --mem-per-cpu=200
#
#SBATCH --array=1-2

A=(COSMOS Web F115W, 'EUCLID Wide VIS')

srun ./calc_survey.py ${FILES[$SLURM_ARRAY_TASK_ID]}
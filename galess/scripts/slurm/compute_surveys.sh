#!/bin/bash
#
#SBATCH --job-name=SL_surveys
#SBATCH --output=outputs/slurm_%x_%a.out
#SBATCH --error=outputs/slurm_%x_%a.err
#SBATCH --export=ALL
#
#SBATCH --ntasks=1
#SBATCH --time=20:00
#SBATCH --mem-per-cpu=200
#
#SBATCH --array=1-2

SURVEYS=(COSMOS Web F115W, 'EUCLID Wide VIS')

srun calc_survey.py ${SURVEYS[$SLURM_ARRAY_TASK_ID]}
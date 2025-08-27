#!/bin/bash

#SBATCH -J homography_geometry_double_head
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=16         # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=a100-galvani   # Which partition will run your job
#SBATCH --time=2-23:59             # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:2               # (optional) Requesting type and number of GPUs
#SBATCH --mem=120G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=/mnt/lustre/work/bamler/zxiao29/haolong/glue-factory/log/homography_geometry_double_head-%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/lustre/work/bamler/zxiao29/haolong/glue-factory/log/homography_geometry_double_head-%j.err        # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ENTER_YOUR_EMAIL   # Email to which notifications will be sent

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus
ls $WORK # not necessary just here to illustrate that $WORK is available here

# Go to working directory, add git commit
cd ${WORK}/haolong/glue-factory
git add -u
git commit -m "homography_geometry_double_head"

source ~/.bashrc
conda activate $WORK/.miniconda3/envs/haolong
srun python -m gluefactory.train homography_geometry_double_head --conf gluefactory/configs/superpoint+simpleglue_homography.yaml  \
            --mp bfloat16 --distributed model.matcher.checkpointed=True
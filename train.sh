#!/bin/sh
#SBATCH -o /home/nttung/Challenge/MediaevalSport/Decompose_MultiLabel_13_LSTMPose/slurm_out/%j.out # STDOUT
python train.py
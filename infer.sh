#!/bin/sh
#SBATCH -o /home/nttung/Challenge/MediaevalSport/Decompose_MultiLabel_13_LSTMPose/slurm_out_infer/%j.out # STDOUT
python infer_json.py
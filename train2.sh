#!/bin/sh
#SBATCH -o /home/nttung/Challenge/MediaevalSport/Decompose_MultiLabel_13_LSTMPose/slurm_out/%j.out # STDOUT
python train.py --frame_interval 30 --img_size 30 120 120 --img_hand_size 30 120 240 --refined_kp True
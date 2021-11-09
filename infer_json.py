import os 
import cv2
import pdb
import datetime
import numpy as np
import torch
import time
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
import pdb
import os.path as osp
import sys 
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns

from dataset import Keypoint3DCNNSport, name_component_to_id_1, name_component_to_id_2, name_component_to_id_3
from model import LSTM3DCNNPose
from sklearn.metrics import confusion_matrix
from utils.utils_model import find_best_combination_hard, find_best_combination_soft


from dataset import name_stroke_to_id_2020, name_stroke_to_id_2021

def get_args_parser():
    # default path

    # Save models' checkpoints
    # path_ckpt_run_save = os.path.join(path_ckpt_save, 'LSTMPose_MediaEval21_%s' % (datetime.datetime.now().strftime('%d-%m-%Y_%H-%M')))
    # os.mkdir(path_ckpt_run_save)

    # 2020 data
    # root_video_frame = '/home/nttung/Challenge/MediaevalSport/baseline/SportTaskME21/data2020'
    # root_json_pose = '/home/nttung/Challenge/MediaevalSport/team_baseline/auxilary-data/out_pose_json_2020_refined'
    # root_xml_file_val = '/home/nttung/Challenge/MediaevalSport/2020_data/data/classificationTask/split_data/valid'
    # path_pretrained_model = '/home/nttung/Challenge/MediaevalSport/Decompose_MultiLabel/checkpoint_2020/LSTM3DCNNPose_MediaEval20_17-10-2021_07-32/LSTM3DCNNPose_184.pth'
    

    # 2021 data
    root_video_frame = '/home/nttung/Challenge/MediaevalSport/baseline/SportTaskME21/data'
    root_json_pose = '/home/nttung/Challenge/MediaevalSport/team_baseline/auxilary-data/out_pose_json_refined'
    root_xml_file_test = '/home/nttung/Challenge/MediaevalSport/2021_data/data/classificationTask/test'
    path_pretrained_model = '/home/nttung/Challenge/MediaevalSport/Decompose_MultiLabel_13_LSTMPose/best_checkpoint_2021/LSTM3DCNNPose_8.pth'

    parser = argparse.ArgumentParser('LSTMPose val arguments', add_help=False)
    parser.add_argument('--frame_interval', default=30, type=int)
    parser.add_argument('--img_size', default=[30, 120, 120], type=list)
    parser.add_argument('--path_pretrained_model', default=path_pretrained_model, type=str)
    parser.add_argument('--img_hand_size', nargs='+', default=[30, 120, 240], type=int)

    parser.add_argument('--root_video_frame', default=root_video_frame, type=str)
    parser.add_argument('--root_json_pose', default=root_json_pose, type=str)
    parser.add_argument('--root_xml_file_test', default=root_xml_file_test, type=str)
    parser.add_argument('--human_box_only', default=True , type=bool)

    return parser

def test(model, test_loader, args, device, name_stroke_to_id):
    
    print("Perform test")
    
    if args.root_video_frame.find('2020') != -1:
        is_2020 = True
        name_img = "confusion_matrix_2020"
    elif args.root_video_frame.find('21') != -1:
        is_2020 = False
        name_img = "confusion_matrix_2021"


    softmax_func = nn.Softmax(dim=0)

    json_path_result = "./submission_decompose_13.json"
    json_res = {}

    for index, data in enumerate(test_loader, 0):
        pose_embed_seqs, rgb_seqs, hand_seqs = data["pose_embedding_video"], data["rgb_sequence"], data["hand_sequence"]

        name_instances = data["name_instance"]             
        # Move to gpu
        pose_embed_seqs, rgb_seqs, hand_seqs = pose_embed_seqs.to(device),\
                                    rgb_seqs.to(device), hand_seqs.to(device)


        pred1, pred2, pred3 = model(rgb_seqs, hand_seqs, pose_embed_seqs)
        for p1, p2, p3, name_ins in zip(pred1, pred2, pred3, name_instances):
                               
            p1 = softmax_func(p1)
            p2 = softmax_func(p2)
            p3 = softmax_func(p3)

            
            best_score, best_comb_label = find_best_combination_hard(p1, p2, p3, is_2020)
            
            l1 = best_comb_label[0][0]
            l3 = best_comb_label[0][1]
            l2 = best_comb_label[1]

            stroke_l1 = name_component_to_id_1[l1]
            stroke_l2 = name_component_to_id_2[l2]
            stroke_l3 = name_component_to_id_3[l3]

            stroke = ' '.join([stroke_l1, stroke_l2, stroke_l3])
            
            json_res[name_ins] = {}
            json_res[name_ins]["result"] = [stroke, best_score]

    
    with open(json_path_result, "w") as fp:
        json.dump(json_res, fp, indent = 4)


def print_args(args):
    print("RUNNING AT SETTINGS \n")
    print("--root_video_frame {} \n".format(args.root_video_frame))
    print("--root_json_pose {} \n".format(args.root_json_pose))
    print("--root_xml_file_test {} \n".format(args.root_xml_file_test))
    print("--path_pretrained_model {} \n".format(args.path_pretrained_model))
    print("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LSTM3DCNNPose Test script', parents=[get_args_parser()])
    args = parser.parse_args()


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # dataset
    test_data = Keypoint3DCNNSport(args.root_video_frame, args.root_json_pose, 
                                args.root_xml_file_test, args.frame_interval, args.img_size, args.img_hand_size, 
                                type_data="test", human_box_only=args.human_box_only)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, 
                                                shuffle=True, num_workers=8)


    if args.root_video_frame.find('2020') != -1:
        name_stroke_to_id = name_stroke_to_id_2020
    elif args.root_video_frame.find('21') != -1:
        name_stroke_to_id = name_stroke_to_id_2021

    # model build
    num_key_frame = args.frame_interval
    embed_dim_LSTM = 2*17 # size of pose embed input to LSTM
    hidden_dim_LSTM = 128
    
    num_class_1 = len(name_component_to_id_1) # number of class
    num_class_2 = len(name_component_to_id_2) # number of class
    num_class_3 = len(name_component_to_id_3) # number of class
    num_class = [num_class_1, num_class_2, num_class_3]

    size_data = np.array(args.img_size)
    size_data_hand = np.array(args.img_hand_size)
    model = LSTM3DCNNPose(embed_dim_LSTM, hidden_dim_LSTM, num_class, size_data, size_data_hand)
    model.to(device)

    # load model
    model.load_state_dict(torch.load(args.path_pretrained_model))

    # set model to eval mode
    model.eval()

    print_args(args)

    test(model, test_loader, args, device, name_stroke_to_id)

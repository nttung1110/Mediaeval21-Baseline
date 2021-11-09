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
import pandas as pd
import seaborn as sns

from dataset import Keypoint3DCNNSport
from model import LSTM3DCNNPose
from sklearn.metrics import confusion_matrix

from dataset import name_stroke_to_id_2020, name_stroke_to_id_2021

def get_args_parser():
    # default path 
    # 2021 data
    # root_video_frame = '/home/nttung/Challenge/MediaevalSport/baseline/SportTaskME21/data'
    # root_json_pose = '/home/nttung/Challenge/MediaevalSport/team_baseline/auxilary-data/out_pose_json_refined'
    # root_xml_file_train = '/home/nttung/Challenge/MediaevalSport/2021_data/data/classificationTask/train'
    # root_xml_file_val = '/home/nttung/Challenge/MediaevalSport/2021_data/data/classificationTask/valid'
    # path_ckpt_save = '/home/nttung/Challenge/MediaevalSport/team_baseline/checkpoint'

    # Save models' checkpoints
    # path_ckpt_run_save = os.path.join(path_ckpt_save, 'LSTMPose_MediaEval21_%s' % (datetime.datetime.now().strftime('%d-%m-%Y_%H-%M')))
    # os.mkdir(path_ckpt_run_save)

    # 2020 data
    root_video_frame = '/home/nttung/Challenge/MediaevalSport/baseline/SportTaskME21/data2020'
    root_json_pose = '/home/nttung/Challenge/MediaevalSport/team_baseline/auxilary-data/out_pose_json_2020_refined'
    root_xml_file_val = '/home/nttung/Challenge/MediaevalSport/2020_data/data/classificationTask/split_data/valid'
    path_pretrained_model = '/home/nttung/Challenge/MediaevalSport/LSTM_CNN3D_Pose/checkpoint_2020/LSTM3DCNNPose_MediaEval20_12-10-2021_02-41/LSTM3DCNNPose_186.pth'
    

    parser = argparse.ArgumentParser('LSTMPose val arguments', add_help=False)
    parser.add_argument('--frame_interval', default=30, type=int)
    parser.add_argument('--img_size', default=[30, 120, 120], type=list)
    parser.add_argument('--path_pretrained_model', default=path_pretrained_model, type=str)

    parser.add_argument('--root_video_frame', default=root_video_frame, type=str)
    parser.add_argument('--root_json_pose', default=root_json_pose, type=str)
    parser.add_argument('--root_xml_file_val', default=root_xml_file_val, type=str)
    

    return parser

def val_on_epoch(model, val_loader, device, name_stroke_to_id):
    print("Evaluating on epoch...")
    since = time.time()

    model.eval() # set to eval mode

    # accs = torch.zeros(len(name_stroke_to_id), device=device)
    acc = 0
    tp = torch.zeros(len(name_stroke_to_id), device=device)
    fp = torch.zeros(len(name_stroke_to_id), device=device)
    total_samples = torch.zeros(len(name_stroke_to_id), device=device)

    for index, data in enumerate(val_loader, 0):
        pose_embed_seqs, rgb_seqs, gt_label_ids = data["pose_embedding_video"],\
                                                    data["rgb_sequence"],\
                                                    data["label_id"]

        pose_embed_seqs, rgb_seqs, gt_label_ids = pose_embed_seqs.to(device),\
                                                rgb_seqs.to(device),\
                                                gt_label_ids.to(device)

        outputs = model(rgb_seqs, pose_embed_seqs)


        bs = outputs.shape[0]
        for idx_batch in range(bs):
            prob_vec = outputs[idx_batch]
            
            gt_label = int(gt_label_ids[idx_batch])
            
            predicted_label = torch.argmax(prob_vec).item()
            if gt_label == predicted_label:
                tp[gt_label] += 1
                acc += 1
            else:
                fp[predicted_label] += 1

            total_samples[gt_label] += 1 

    acc = acc / sum(total_samples).item()
    
    return acc

def val_with_confusion_matrix(model, val_loader, args, device, name_stroke_to_id):

    pred_label = torch.zeros(0,dtype=torch.long, device='cpu')
    gt_label = torch.zeros(0,dtype=torch.long, device='cpu')


    confusion_matrix = np.zeros((len(name_stroke_to_id), len(name_stroke_to_id)))
    for index, data in enumerate(val_loader, 0):
        pose_embed_seqs, rgb_seqs, label_ids = data["pose_embedding_video"],\
                                                data["rgb_sequence"],\
                                                data["label_id"]

        pose_embed_seqs, rgb_seqs, label_ids = pose_embed_seqs.to(device),\
                                                rgb_seqs.to(device),\
                                                label_ids.to(device)

        outputs = model(rgb_seqs, pose_embed_seqs)
        pdb.set_trace()
        _, preds = torch.max(outputs, 1)

        for t, p in zip(label_ids.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    plt.figure(figsize=(15,10))

    class_names = name_stroke_to_id
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def print_args(args):
    print("RUNNING AT SETTINGS \n")
    print("--root_video_frame {} \n".format(args.root_video_frame))
    print("--root_json_pose {} \n".format(args.root_json_pose))
    print("--root_xml_file_val {} \n".format(args.root_xml_file_val))
    print("--path_pretrained_model {} \n".format(args.path_pretrained_model))
    print("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LSTM3DCNNPose Evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # dataset
    val_data = Keypoint3DCNNSport(args.root_video_frame, args.root_json_pose, 
                                args.root_xml_file_val, args.frame_interval, args.img_size)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, 
                                                shuffle=True, num_workers=8)


    if args.root_video_frame.find('2020'):
        name_stroke_to_id = name_stroke_to_id_2020
    elif args.root_video_frame.find('2021'):
        name_stroke_to_id = name_stroke_to_id_2021

    # model build
    num_key_frame = args.frame_interval
    embed_dim_LSTM = 2*17 # size of pose embed input to LSTM
    hidden_dim_LSTM = 128
    num_class = len(name_stroke_to_id) # number of class
    size_data = np.array(args.img_size)
    model = LSTM3DCNNPose(embed_dim_LSTM, hidden_dim_LSTM, num_class, size_data)
    model.to(device)

    # load model
    model.load_state_dict(torch.load(args.path_pretrained_model))

    # set model to eval mode
    model.eval()

    print_args(args)

    val_with_confusion_matrix(model, val_loader, 
                    args, device, name_stroke_to_id)

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

from dataset import Keypoint3DCNNSport, name_component_to_id_1, name_component_to_id_2, name_component_to_id_3
from model import LSTM3DCNNPose
from utils.utils_model import find_best_combination_hard, find_best_combination_soft

from dataset import name_stroke_to_id_2020, name_stroke_to_id_2021

def get_args_parser():
    # default path 
    # 2021 data
    root_video_frame = '/home/nttung/Challenge/MediaevalSport/baseline/SportTaskME21/data'
    root_json_pose = '/home/nttung/Challenge/MediaevalSport/team_baseline/auxilary-data/out_pose_json_refined'
    root_xml_file_train = '/home/nttung/Challenge/MediaevalSport/2021_data/data/classificationTask/train'
    root_xml_file_val = '/home/nttung/Challenge/MediaevalSport/2021_data/data/classificationTask/valid'
    path_ckpt_save = '/home/nttung/Challenge/MediaevalSport/Decompose_MultiLabel_13_LSTMPose/checkpoint_2021'

    # Save models' checkpoints
    # path_ckpt_run_save = os.path.join(path_ckpt_save, 'LSTMPose_MediaEval21_%s' % (datetime.datetime.now().strftime('%d-%m-%Y_%H-%M')))
    # os.mkdir(path_ckpt_run_save)

    # 2020 data
    # root_video_frame = '/home/nttung/Challenge/MediaevalSport/baseline/SportTaskME21/data2020'
    # root_json_pose = '/home/nttung/Challenge/MediaevalSport/team_baseline/auxilary-data/out_pose_json_2020_refined'
    # root_xml_file_train = '/home/nttung/Challenge/MediaevalSport/2020_data/data/classificationTask/split_data/train'
    # root_xml_file_val = '/home/nttung/Challenge/MediaevalSport/2020_data/data/classificationTask/split_data/valid'
    # path_ckpt_save = '/home/nttung/Challenge/MediaevalSport/Decompose_MultiLabel/checkpoint_2020'
    
    
    # Save models' checkpoints
    path_ckpt_run_save = os.path.join(path_ckpt_save, 'LSTM3DCNNPose_MediaEval21_%s' % (datetime.datetime.now().strftime('%d-%m-%Y_%H-%M')))
    if not os.path.isdir(path_ckpt_run_save):
        os.mkdir(path_ckpt_run_save)

    parser = argparse.ArgumentParser('LSTMPose arguments', add_help=False)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=4, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--optimizer', default='adagrad', type=str)
    parser.add_argument('--frame_interval', default=30, type=int)
    parser.add_argument('--img_size', nargs='+', default=[30, 120, 120], type=int)
    parser.add_argument('--img_hand_size', nargs='+', default=[30, 120, 240], type=int)
    parser.add_argument('--path_pretrained_model', default=None, type=str)

    parser.add_argument('--root_video_frame', default=root_video_frame, type=str)
    parser.add_argument('--root_json_pose', default=root_json_pose, type=str)
    parser.add_argument('--root_xml_file_train', default=root_xml_file_train, type=str)
    parser.add_argument('--root_xml_file_val', default=root_xml_file_val, type=str)
    
    parser.add_argument('--path_ckpt_run_save', default=path_ckpt_run_save, type=str)

    # Additional methods
    parser.add_argument('--human_box_only', default=True , type=bool)
    parser.add_argument('--refined_kp', default=False , type=bool) ### Get keypoints of table tennis player only


    return parser

def val_on_epoch(model, val_loader, device, name_stroke_to_id, args):
    print("Evaluating on epoch...")
    since = time.time()

    model.eval() # set to eval mode

    # accs = torch.zeros(len(name_stroke_to_id), device=device)
    acc = 0
    # tp = torch.zeros(len(name_stroke_to_id), device=device)
    # fp = torch.zeros(len(name_stroke_to_id), device=device)
    # total_samples = torch.zeros(len(name_stroke_to_id), device=device)
    total_samples = 0

    acc_1 = 0
    acc_2 = 0
    acc_3 = 0

    independent_acc_1 = 0
    independent_acc_2 = 0
    independent_acc_3 = 0

    softmax_func = nn.Softmax(dim=0)

    if args.root_video_frame.find('2020') != -1:
        is_2020 = True

    elif args.root_video_frame.find('21') != -1:
        is_2020 = False

    print("Is 2020 data: ", is_2020)

    for index, data in enumerate(val_loader, 0):
        pose_embed_seqs, rgb_seqs, hand_seqs = data["pose_embedding_video"], data["rgb_sequence"], data["hand_sequence"]

        gt_label_ids_1, gt_label_ids_2, gt_label_ids_3 = data["label_id_1"], \
                                                data["label_id_2"], \
                                                data["label_id_3"]
                                        
        # Move to gpu
        pose_embed_seqs, rgb_seqs, hand_seqs = pose_embed_seqs.to(device),\
                                    rgb_seqs.to(device), hand_seqs.to(device)
    
        gt_label_ids_1, gt_label_ids_2, gt_label_ids_3 = gt_label_ids_1.to(device),\
                                                    gt_label_ids_2.to(device),\
                                                    gt_label_ids_3.to(device)

        x1, x2, x3 = model(rgb_seqs, hand_seqs, pose_embed_seqs)

        bs = x1.shape[0] # equal batch size 
        for idx_batch in range(bs):
            prob_vec_1 = x1[idx_batch]
            prob_vec_2 = x2[idx_batch]
            prob_vec_3 = x3[idx_batch]

            # apply softmax first for three prob_vec
            norm_prob_vec_1 = softmax_func(prob_vec_1)
            norm_prob_vec_2 = softmax_func(prob_vec_2)
            norm_prob_vec_3 = softmax_func(prob_vec_3)

            
            gt_label_1 = int(gt_label_ids_1[idx_batch])
            gt_label_2 = int(gt_label_ids_2[idx_batch])
            gt_label_3 = int(gt_label_ids_3[idx_batch])

            # dependable accuracy - hard assignment
            best_score, best_comb_label = find_best_combination_hard(norm_prob_vec_1, norm_prob_vec_2, norm_prob_vec_3, is_2020)
            # return result in format ((label1, label3), label2)


            # soft assignment
            # best_score, best_comb_label = find_best_combination_soft(norm_prob_vec_1, norm_prob_vec_2, norm_prob_vec_3, is_2020)
            # Only positive if all three label match with gt
            if best_comb_label[0][0] == gt_label_1:
                acc_1 += 1
            
            if best_comb_label[0][1] == gt_label_3:
                acc_3 += 1

            if best_comb_label[1] == gt_label_2:
                acc_2 += 1

            if best_comb_label[0][0] == gt_label_1 and best_comb_label[0][1] == gt_label_3 and best_comb_label[1] == gt_label_2:
                acc += 1

            # independent accuracy
            predicted_label_1 = torch.argmax(norm_prob_vec_1).item()
            predicted_label_2 = torch.argmax(norm_prob_vec_2).item()
            predicted_label_3 = torch.argmax(norm_prob_vec_3).item()
            # Independent acc
            if predicted_label_1 == gt_label_1:
                independent_acc_1 += 1

            if predicted_label_2 == gt_label_2:
                independent_acc_2 += 1

            if predicted_label_3 == gt_label_3:
                independent_acc_3 += 1

            total_samples += 1 

    acc = acc / total_samples
    acc1 = acc_1 / total_samples
    acc2 = acc_2 / total_samples
    acc3 = acc_3 / total_samples

    independent_acc_1 = independent_acc_1 / total_samples
    independent_acc_2 = independent_acc_2 / total_samples
    independent_acc_3 = independent_acc_3 / total_samples
    
    return acc1, acc2, acc3, independent_acc_1, independent_acc_2, independent_acc_3, acc

def train_and_val(model, train_data, val_data, criterion, optimizer, args, device, name_stroke_to_id):
    # Training
    if args.path_pretrained_model is None:
        start_epoch = 0
    else:
        # get resume previous epoch
        # format: /../LSTMPose_{num epoch}.pth
        start_epoch = int(args.path_pretrained_model.split('/')[-1].split('_')[-1][:-4]) + 1
        model.to(device)
        model.load_state_dict(torch.load(args.path_ckpt))
        # model, optimizer, start_epoch = train_from_ckpt(path_ckpt, model, optimizer)

    running_loss = 0.0
    for epoch in range(start_epoch+1, args.epochs):
        model.train() # set to train mode
        for index, data in enumerate(train_data, 0):
            pose_embed_seqs, rgb_seqs, hand_seqs = data["pose_embedding_video"], data["rgb_sequence"], data["hand_sequence"]

            label_ids_1, label_ids_2, label_ids_3 = data["label_id_1"], \
                                                    data["label_id_2"], \
                                                    data["label_id_3"]
                                            
            # Move to gpu
            pose_embed_seqs, rgb_seqs, hand_seqs = pose_embed_seqs.to(device),\
                                        rgb_seqs.to(device), hand_seqs.to(device)
        
            label_ids_1, label_ids_2, label_ids_3 = label_ids_1.to(device),\
                                                     label_ids_2.to(device),\
                                                     label_ids_3.to(device)

            optimizer.zero_grad()

            pred1, pred2, pred3 = model(rgb_seqs, hand_seqs, pose_embed_seqs)
            # pdb.set_trace()
            
            # Calculating loss for three prediction heads
            loss_1 = criterion(pred1, label_ids_1)
            loss_2 = criterion(pred2, label_ids_2)
            loss_3 = criterion(pred3, label_ids_3)

            # Summing loss
            loss = loss_1 + loss_2 + loss_3
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if index % 2 == 0:
                # print every 50 mini batches
                print("[epoch_%d, batch_%d] loss: %.10f" % (epoch, index, running_loss / 50))
                wandb.log({'loss': running_loss/50})
                running_loss = 0.0

            # debug
            # acc1, acc2, acc3, independent_acc_1, independent_acc_2, independent_acc_3, acc = val_on_epoch(model, val_data, device, name_stroke_to_id, args)

        # save checkpoint
        if epoch % 2 == 0:
            write_path_ckpt = osp.join(args.path_ckpt_run_save, 'LSTM3DCNNPose_{}.pth'.format(epoch))
            torch.save(model.state_dict(), write_path_ckpt)

        # eval on each epoch
        acc1, acc2, acc3, independent_acc_1, independent_acc_2, independent_acc_3, acc = val_on_epoch(model, val_data, device, name_stroke_to_id, args)
        print("Accuracy of sub-category 1 on epoch %d is %.3f"%(epoch, acc1))
        print("Accuracy of sub-category 2 on epoch %d is %.3f"%(epoch, acc2))
        print("Accuracy of sub-category 3 on epoch %d is %.3f"%(epoch, acc3))
        print("Accuracy of combined category on epoch %d is %.3f"%(epoch, acc))

        print("Independent Accuracy of sub-category 1 on epoch %d is %.3f"%(epoch, independent_acc_1))
        print("Independent Accuracy of sub-category 2 on epoch %d is %.3f"%(epoch, independent_acc_2))
        print("Independent Accuracy of sub-category 3 on epoch %d is %.3f"%(epoch, independent_acc_3))

def print_args(args):
    print("RUNNING AT SETTINGS \n")
    print("--lr {} --batch_size {} --weight_decay {} --epochs {} --optimizer {} --frame_interval {} --img_size {} --human_box_only {} --refined_kp {}".format(args.lr, args.batch_size, args.weight_decay, args.epochs, args.optimizer, args.frame_interval, args.img_size, args.human_box_only, args.refined_kp))
    print("--root_video_frame {} \n".format(args.root_video_frame))
    print("--root_json_pose {} \n".format(args.root_json_pose))
    print("--root_xml_file_train {} \n".format(args.root_xml_file_train))
    print("--root_xml_file_val {} \n".format(args.root_xml_file_val))
    print("--path_ckpt_run_save {} \n".format(args.path_ckpt_run_save))
    print("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LSTM3DCNNPose Training script', parents=[get_args_parser()])
    args = parser.parse_args()


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # dataset 
    train_data = Keypoint3DCNNSport(args.root_video_frame, args.root_json_pose, 
                                args.root_xml_file_train, args.frame_interval, args.img_size, args.img_hand_size
                                , human_box_only=args.human_box_only, refined_kp = args.refined_kp)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.num_workers)
    
    val_data = Keypoint3DCNNSport(args.root_video_frame, args.root_json_pose, 
                                args.root_xml_file_val, args.frame_interval, args.img_size, args.img_hand_size
                                , human_box_only=args.human_box_only, refined_kp = args.refined_kp)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, 
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

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    
    # init wandb
    wandb.init(project='MediaevalSport', entity='nttung1110')
    wandb.watch(model, log_freq=100)

    print_args(args)

    train_and_val(model, train_loader, val_loader, 
                    criterion, optimizer, args, device, name_stroke_to_id)

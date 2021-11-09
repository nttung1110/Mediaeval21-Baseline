import os
import numpy 
import torch
import cv2
import os.path as osp 
import os
import math
import pdb
import numpy as np
import json
import copy

from tqdm import tqdm
from xml.etree import ElementTree
from utils import refine_keypoints


name_stroke_to_id_2021 = ['Serve Forehand Backspin',
                'Serve Forehand Loop',
                'Serve Forehand Sidespin',
                'Serve Forehand Topspin',

                'Serve Backhand Backspin',
                #'Serve Backhand Loop',
                #'Serve Backhand Sidespin',
                'Serve Backhand Topspin',

                'Offensive Forehand Hit',
                'Offensive Forehand Loop',
                'Offensive Forehand Flip',

                'Offensive Backhand Hit',
                'Offensive Backhand Loop',
                'Offensive Backhand Flip',

                'Defensive Forehand Push',
                'Defensive Forehand Block',
                'Defensive Forehand Backspin',

                'Defensive Backhand Push',
                'Defensive Backhand Block',
                'Defensive Backhand Backspin']

name_stroke_to_id_2020 = ['Serve Forehand Backspin',
                'Serve Forehand Loop',
                'Serve Forehand Sidespin',
                'Serve Forehand Topspin',

                'Serve Backhand Backspin',
                'Serve Backhand Loop',
                'Serve Backhand Sidespin',
                'Serve Backhand Topspin',

                'Offensive Forehand Hit',
                'Offensive Forehand Loop',
                'Offensive Forehand Flip',

                'Offensive Backhand Hit',
                'Offensive Backhand Loop',
                'Offensive Backhand Flip',

                'Defensive Forehand Push',
                'Defensive Forehand Block',
                'Defensive Forehand Backspin',

                'Defensive Backhand Push',
                'Defensive Backhand Block',
                'Defensive Backhand Backspin']

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

KEYPOINT_CONNECTION_RULES = [
    # face
    ("left_ear", "left_eye", (102, 204, 255)),
    ("right_ear", "right_eye", (51, 153, 255)),
    ("left_eye", "nose", (102, 0, 204)),
    ("nose", "right_eye", (51, 102, 255)),
    # upper-body
    ("left_shoulder", "right_shoulder", (255, 128, 0)),
    ("left_shoulder", "left_elbow", (153, 255, 204)),
    ("right_shoulder", "right_elbow", (128, 229, 255)),
    ("left_elbow", "left_wrist", (153, 255, 153)),
    ("right_elbow", "right_wrist", (102, 255, 224)),
    # lower-body
    ("left_hip", "right_hip", (255, 102, 0)),
    ("left_hip", "left_knee", (255, 255, 77)),
    ("right_hip", "right_knee", (153, 255, 204)),
    ("left_knee", "left_ankle", (191, 255, 128)),
    ("right_knee", "right_ankle", (255, 195, 77)),
]


# Be careful when changing the orders of name_component_to_id
name_component_to_id_1 = ['Serve', 'Offensive', 'Defensive']
name_component_to_id_2 = ['Forehand', 'Backhand']
name_component_to_id_3 = ['Backspin', 'Loop', 'Sidespin', 'Topspin', 'Hit', 'Flip', 'Push', 'Block']

def encode_onehot(cat_id, num_cat):
    oh_encoding = np.zeros((1, num_cat))
    oh_encoding[:, cat_id] = 1
    return oh_encoding

def euclidean_wrist2elbow(pos_twist, pos_elbow):
    dis = ((pos_twist[0]-pos_elbow[0])**2 + (pos_twist[1]-pos_elbow[1])**2)**0.5

    return dis

class Keypoint3DCNNSport(torch.utils.data.Dataset):
    def __init__(self, root_video_frame, root_json_pose, root_xml_file, frame_interval, 
                img_size, img_hand_size, type_data = "trainval",
                human_box_only=False, refined_kp = False):
        self.root_video_frame = root_video_frame
        self.root_json_pose = root_json_pose
        self.root_xml_file = root_xml_file
        self.frame_interval = frame_interval
        self.img_size = img_size
        self.img_hand_size = img_hand_size
        self.human_box_only = human_box_only
        self.refined_kp = refined_kp
        self.type_data = type_data
        self.data = []

        if self.human_box_only:
            print("Image with human box only")
        else: 
            print("Full image")

        if self.refined_kp:
            print("Refined main kp only")
        else: 
            print("Get kp by using mmpose confidence")

        if self.root_video_frame.find('2020') != -1:
            self.name_stroke_to_id = name_stroke_to_id_2020
            self.dataset_name = "2020"
        elif self.root_video_frame.find('21') != -1:
            self.name_stroke_to_id = name_stroke_to_id_2021
            self.dataset_name = "2021"

        print("Dataset name:", self.dataset_name)

        self.name_component_to_id_1 = name_component_to_id_1
        self.name_component_to_id_2 = name_component_to_id_2
        self.name_component_to_id_3 = name_component_to_id_3
        
        # build data list
        for xml_file in os.listdir(self.root_xml_file):
            if xml_file.endswith('.xml') is False:
                continue
            xml_path = osp.join(self.root_xml_file, xml_file)
            root = ElementTree.parse(xml_path).getroot()
            for instance in root:
                if self.type_data == "trainval":
                    label_join = instance.get('move')
                    label = instance.get('move').split(' ')
                sequence = [int(instance.get('begin')), int(instance.get('end'))]
                video_id = xml_path.split('/')[-1][:-4]
                name_instance = video_id + "_fr_"+str(sequence[0])+"_"+str(sequence[1])

                if self.type_data == "trainval":
                    self.data.append({'label_1': label[0], 'label_2': label[1], 'label_3': label[2], 'label': label_join,
                                    'sequence': sequence, 'video_id':video_id, 'name_instance': name_instance})
                elif self.type_data == "test":
                    self.data.append({'sequence': sequence, 'video_id':video_id, 'name_instance': name_instance})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get frame data 
        sample = self.data[index]

        if self.type_data == "trainval":
            label_1, label_2, label_3, label_join = sample['label_1'], sample['label_2'],\
                                                    sample['label_3'], sample['label']
        sequence = sample['sequence']
        video_id = sample['video_id']
        name_instance = sample["name_instance"]

        total_frame = sequence[1] - sequence[0] + 1
        normalized_num_frame = int(math.floor(total_frame / self.frame_interval))

        frame_idx_list = []

        start_frame, end_frame = sequence[0], sequence[1]
        for i in range(self.frame_interval):
            next_frame_id = start_frame + i*normalized_num_frame
            frame_idx_list.append(next_frame_id)

        # get pose embedding vector for each_frame
        pose_path = osp.join(self.root_json_pose, video_id+'.json')    

        # read pose_data and rgb data

        '''
        Rule for extracting pose:
            1. More than two pose existed => get kp with highest confident score
            2. No pose existed => get the pose of nearest frame  
        '''
        with open(pose_path,'r') as fp:
            pose_data = json.load(fp)


        pose_embedding_video = []
        rgb_sequence = []
        hand_sequence = []
        
        for frame_idx in frame_idx_list:
            padding_frame_idx = str(frame_idx).zfill(8)
            path_frame = osp.join(self.root_video_frame, video_id, padding_frame_idx+'.png')

            # get rgb data
            rgb = cv2.imread(path_frame)
            h, w, _ = rgb.shape

            # get pose data
            kp_list = pose_data[path_frame]

            if len(kp_list) >= 1:
                # get the pose with highest
                idx_chosen = max(range(len(kp_list)), key=lambda index: kp_list[index]['score'])
                main_kp_frame = kp_list[idx_chosen]

                if self.refined_kp and len(kp_list) > 1: 
                    main_kp_frame = refine_keypoints.find_main_kp_frame(kp_list, w, h, video_id, frame_idx)


            elif len(kp_list) == 0:
                run_nearest = 1

                # find pose of nearest frame
                lr_list = [-run_nearest, run_nearest]
                while True:
                    for addi in lr_list:
                        neighbor_frame_idx = frame_idx + addi 

                        if neighbor_frame_idx < start_frame or neighbor_frame_idx > end_frame:
                            continue
                        path_neighbor_frame = osp.join(self.root_video_frame, video_id, neighbor_frame_idx+'.png')
                        neighbor_kp_list = pose_data[path_neighbor_frame]
                        if len(neighbor_kp_list) != 0:
                            idx_chosen = max(range(len(neighbor_kp_list)), key=lambda index: neighbor_kp_list[index]['score'])
                            main_kp_frame = neighbor_kp_list[idx_chosen]
                            
                            if self.refined_kp and len(neighbor_kp_list) > 1: 
                                main_kp_frame = refine_keypoints.find_main_kp_frame(neighbor_kp_list, w, h, video_id, neighbor_frame_idx)
                        
                            break
                    lr_list[0] -= 1
                    lr_list[1] += 1

            array_kp = np.array(main_kp_frame['keypoints'])

            min_hor, min_ver = np.min(array_kp, axis=0)
            max_hor, max_ver = np.max(array_kp, axis=0)

            min_hor, min_ver = max(0, min_hor), max(0, min_ver)
            max_hor, max_ver = min(max_hor, w), min(max_ver, h)
            
            if self.human_box_only:
                # cut human bbox
                cropped_img = rgb[min_ver:max_ver, min_hor:max_hor]
                unnormed_rgb = cv2.resize(cropped_img, (self.img_size[1], self.img_size[2])).astype(float)
                rgb = unnormed_rgb / 255
            else:
                unnormed_rgb = cv2.resize(cropped_img, (self.img_size[1], self.img_size[2])).astype(float)
                rgb = unnormed_rgb / 255

            # process keypoint: normalization
            normed_kp_list = []
            list_model_kp_coordinates = []
            # process keypoint: normalization for output (no norm for process hand)
            for each_kp in main_kp_frame['keypoints']:

                # Method 1: normalize based on image size
                # hor_coor = each_kp[0] / float(dim_img[1])
                # ver_coor = each_kp[1] / float(dim_img[0])

                # Method 2: normalize based on surrounding kp and transform kp
                # hor_coor = (each_kp[0] - float(min_hor)) / float(max_hor - min_hor)
                # ver_coor = (each_kp[1] - float(min_ver)) / float(max_ver - min_ver)
                ratio_hor = self.img_size[2] / float(max_hor - min_hor)
                ratio_ver = self.img_size[1] / float(max_ver - min_ver)
                hor_coor = (each_kp[0] - float(min_hor)) * ratio_hor
                ver_coor = (each_kp[1] - float(min_ver)) * ratio_ver
                list_model_kp_coordinates.extend([hor_coor / float(max_hor - min_hor), ver_coor / float(max_ver - min_ver)])
                normed_kp_list.append([hor_coor, ver_coor])


            # # vis kp coordinates
            # vis_kp_img = copy.copy(unnormed_rgb)
            # for pair in KEYPOINT_CONNECTION_RULES:
            #     name_kp1 = pair[0]
            #     name_kp2 = pair[1]
            #     color = pair[2]
                
            #     idx_kp1 = list(COCO_KEYPOINT_INDEXES.keys())[list(COCO_KEYPOINT_INDEXES.values()).index(name_kp1)]
            #     idx_kp2 = list(COCO_KEYPOINT_INDEXES.keys())[list(COCO_KEYPOINT_INDEXES.values()).index(name_kp2)]

            #     coord_kp1 = normed_kp_list[idx_kp1]
            #     coord_kp2 = normed_kp_list[idx_kp2]

            #     # connect point
            #     vis_kp_img = cv2.line(vis_kp_img, (int(coord_kp1[0]), int(coord_kp1[1])), (int(coord_kp2[0]), int(coord_kp2[1])), color, 4)

            # cv2.imwrite("vis_kp.jpg", vis_kp_img)


            # get hand image 
            left_dis_wrist2elbow = euclidean_wrist2elbow(normed_kp_list[9], normed_kp_list[7])

            left_hand_pos_xymin = list(map(int, [max(normed_kp_list[9][0]-left_dis_wrist2elbow, 0),\
                                max(normed_kp_list[9][1]-left_dis_wrist2elbow, 0)]))

            left_hand_pos_xymax = list(map(int, [min(normed_kp_list[9][0]+left_dis_wrist2elbow, w), \
                                    min(normed_kp_list[9][1]+left_dis_wrist2elbow, h)]))

            
            right_hand_pos_xymin = list(map(int, [max(normed_kp_list[10][0]-left_dis_wrist2elbow, 0),\
                                    max(normed_kp_list[10][1]-left_dis_wrist2elbow, 0)]))

            right_hand_pos_xymax = list(map(int, [min(normed_kp_list[10][0]+left_dis_wrist2elbow, w), \
                                    min(normed_kp_list[10][1]+left_dis_wrist2elbow, h)]))

            left_hand_box = unnormed_rgb[left_hand_pos_xymin[1]:left_hand_pos_xymax[1], left_hand_pos_xymin[0]:left_hand_pos_xymax[0]] 
            right_hand_box = unnormed_rgb[right_hand_pos_xymin[1]:right_hand_pos_xymax[1], right_hand_pos_xymin[0]:right_hand_pos_xymax[0]] 

            if len(left_hand_box) == 0 or len(left_hand_box[0]) == 0:
                left_hand_box = unnormed_rgb 
            if len(right_hand_box) == 0 or len(right_hand_box[0]) == 0:
                right_hand_box = unnormed_rgb
            # debug only        
            # vis_img = copy.copy(unnormed_rgb)
            # vis_img = cv2.rectangle(vis_img, (left_hand_pos_xymin[0], left_hand_pos_xymin[1]),
            #                         (left_hand_pos_xymax[0], left_hand_pos_xymax[1]),
            #                         (255, 0, 0),
            #                         2)
            # vis_img = cv2.rectangle(vis_img, (right_hand_pos_xymin[0], right_hand_pos_xymin[1]),
            #                         (right_hand_pos_xymax[0], right_hand_pos_xymax[1]),
            #                         (0, 255, 0),
            #                         2)
            # cv2.imwrite('vis_hand.jpg', vis_img)
            # print(left_hand_box.shape)
            # print(right_hand_box.shape)

            left_hand_box = cv2.resize(left_hand_box, (self.img_hand_size[1], self.img_hand_size[1]))
            right_hand_box = cv2.resize(right_hand_box, (self.img_hand_size[1], self.img_hand_size[1]))
            hand_img = cv2.hconcat([left_hand_box, right_hand_box])
            
            hand_img = hand_img.astype(float) / 255


            # save sequence info
            pose_embedding_video.append(list_model_kp_coordinates)
            rgb_sequence.append(cv2.split(rgb))
            hand_sequence.append(cv2.split(hand_img))

        rgb_sequence = torch.FloatTensor(np.transpose(rgb_sequence, (1, 0, 2, 3)))
        pose_embedding_video = torch.FloatTensor(pose_embedding_video)
        hand_sequence = torch.FloatTensor(np.transpose(hand_sequence, (1, 0, 2, 3)))

        if self.type_data == "trainval":
            label_id_1 = np.array(int(self.name_component_to_id_1.index(label_1)))
            label_id_2 = np.array(int(self.name_component_to_id_2.index(label_2)))
            label_id_3 = np.array(int(self.name_component_to_id_3.index(label_3)))
            label_id = np.array(int(self.name_stroke_to_id.index(label_join)))
            video_id = torch.Tensor([int(video_id)])

            dict_data = {"pose_embedding_video": pose_embedding_video, 
                        "label_id_1": label_id_1,
                        "label_id_2": label_id_2,
                        "label_id_3": label_id_3,
                        "label_id": label_id,
                        "video_id": video_id,
                        "name_instance": name_instance,
                        "rgb_sequence": rgb_sequence,
                        "hand_sequence": hand_sequence}
                        
        elif self.type_data == "test":
            dict_data = {"pose_embedding_video": pose_embedding_video, 
                        "video_id": video_id,
                        "name_instance": name_instance,
                        "rgb_sequence": rgb_sequence,
                        "hand_sequence": hand_sequence}
        return dict_data

if __name__ == '__main__':
    # test dataloader 

    root_video_frame = '/home/nttung/Challenge/MediaevalSport/baseline/SportTaskME21/data'
    root_json_pose = '/home/nttung/Challenge/MediaevalSport/team_baseline/auxilary-data/out_pose_json_refined'
    root_xml_file = '/home/nttung/Challenge/MediaevalSport/2021_data/data/classificationTask/valid'
    
    
    # root_video_frame = '/home/nttung/Challenge/MediaevalSport/baseline/SportTaskME21/data_2020'
    # root_json_pose = '/home/nttung/Challenge/MediaevalSport/team_baseline/auxilary-data/out_pose_json_refined'
    # root_xml_file = '/home/nttung/Challenge/MediaevalSport/2021_data/data/classificationTask/train'

    frame_interval = 30
    img_size = (frame_interval, 120, 120)
    img_hand_size = (frame_interval, 120, 240)
    refined_kp = True
    dataloader = Keypoint3DCNNSport(root_video_frame, root_json_pose, root_xml_file, frame_interval, img_size, img_hand_size, human_box_only=True, refined_kp = refined_kp)

    for idx_data in tqdm(range(len(dataloader))):
        item = dataloader[idx_data]
        pose_embed, rgb_seq = dataloader[idx_data]["pose_embedding_video"], dataloader[idx_data]["rgb_sequence"]
        # if label not in stat_label:
        #     stat_label[label] = 0 
        # stat_label[label] += 1

        
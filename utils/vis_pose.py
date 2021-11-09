import os
import cv2

import numpy as np
import json
import copy
import pdb
import os.path as osp

from pathlib import Path
from tqdm import tqdm
from PIL import Image

COLOR_PANEL = [
    (255, 0, 0), #BLUE
    (0, 255, 0), # GREEN
    (0, 255, 0), 
    (0, 0, 255), # RED
    (0, 0, 255),
    (187, 121, 133),
    (187, 121, 133),
    (255, 255, 0),
    (255, 255, 0),
    (128, 128, 0),
    (128, 128, 0),
    (255, 255, 0),
    (255, 255, 0),
    (255, 255, 0),
    (255, 255, 0),
    (0, 128, 128),
    (0, 128, 128)
]

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

def vis_pose(path_json, path_out_pose_vis):
    # read json pose
    with open(path_json, 'r') as fp:
        pose_data = json.load(fp)

    skip_frame = 20

    for frame_path in sorted(pose_data.keys()):
        idx_frame = int(frame_path.split("/")[-1][:-4])
        img_name = frame_path.split("/")[-1]

        if idx_frame % 20 != 0:
            continue

        if len(pose_data[frame_path]) != 1:
            continue

        pose_info = pose_data[frame_path][0]["keypoints"]

        # draw res
        img = cv2.imread(frame_path)
        vis_img = copy.copy(img)

        for pair in KEYPOINT_CONNECTION_RULES:
            name_kp1 = pair[0]
            name_kp2 = pair[1]
            color = pair[2]
            
            idx_kp1 = list(COCO_KEYPOINT_INDEXES.keys())[list(COCO_KEYPOINT_INDEXES.values()).index(name_kp1)]
            idx_kp2 = list(COCO_KEYPOINT_INDEXES.keys())[list(COCO_KEYPOINT_INDEXES.values()).index(name_kp2)]

            coord_kp1 = pose_info[idx_kp1]
            coord_kp2 = pose_info[idx_kp2]

            # connect point
            vis_img = cv2.line(vis_img, tuple(coord_kp1), tuple(coord_kp2), color, 4)

        path_out_img = osp.join(path_out_pose_vis, img_name)

        cv2.imwrite(path_out_img, vis_img)


if __name__ == "__main__":
    video_id = '113733964'
    path_json = osp.join('/home/nttung/Challenge/MediaevalSport/team_baseline/auxilary-data/out_pose_json_2020_refined', video_id+'.json')
    path_out_pose_vis = osp.join('/home/nttung/Challenge/MediaevalSport/LSTM_CNN3D_Pose/utils/pose_vis', video_id+'_vis')
    Path(path_out_pose_vis).mkdir(parents=True, exist_ok=True)
    
    # visualize pose of specific video id 
    vis_pose(path_json, path_out_pose_vis)
    
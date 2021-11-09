import os
import numpy as np 
import json
import cv2
import pdb
import copy

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


def euclidean_wrist2elbow(pos_twist, pos_elbow):
    dis = ((pos_twist[0]-pos_elbow[0])**2 + (pos_twist[1]-pos_elbow[1])**2)**0.5

    return dis

def kp_2_hand_box(root_path_json, root_path_vid_frame, path_out):


    already_vis = []
    for file_name in os.listdir(root_path_json):
        print("Visualize:", file_name)
        video_id = file_name.split(".")[0]
        file_pose_json = os.path.join(root_path_json, file_name)
        with open(file_pose_json, "r") as fp:
            pose_data = json.load(fp)

        path_out_vid_id = os.path.join(path_out, video_id)
        if not os.path.isdir(path_out_vid_id):
            os.mkdir(path_out_vid_id)

        all_fr = pose_data.keys()
        fr_get = list(all_fr)[3]
        kp_info = pose_data[fr_get]
        idx_chosen = max(range(len(kp_info)), key=lambda index: kp_info[index]['score'])
        chosen_kp = kp_info[idx_chosen]['keypoints']
        path_out_vid_id = os.path.join(path_out, video_id)

        img_read = cv2.imread(fr_get)
        (h, w, _) = img_read.shape
        left_dis_wrist2elbow = euclidean_wrist2elbow(chosen_kp[9], chosen_kp[7])
        right_dis_wrist2elbow =  euclidean_wrist2elbow(chosen_kp[10], chosen_kp[8])

        left_hand_pos_xymin = list(map(int, [max(chosen_kp[9][0]-left_dis_wrist2elbow, 0),\
                                max(chosen_kp[9][1]-left_dis_wrist2elbow, 0)]))

        left_hand_pos_xymax = list(map(int, [min(chosen_kp[9][0]+left_dis_wrist2elbow, w), \
                                min(chosen_kp[9][1]+left_dis_wrist2elbow, h)]))

        
        right_hand_pos_xymin = list(map(int, [max(chosen_kp[10][0]-left_dis_wrist2elbow, 0),\
                                max(chosen_kp[10][1]-left_dis_wrist2elbow, 0)]))

        right_hand_pos_xymax = list(map(int, [min(chosen_kp[10][0]+left_dis_wrist2elbow, w), \
                                min(chosen_kp[10][1]+left_dis_wrist2elbow, h)]))
        # get left and right hand box
        left_hand_box = img_read[left_hand_pos_xymin[1]:left_hand_pos_xymax[1], left_hand_pos_xymin[0]:left_hand_pos_xymax[0]] 
        right_hand_box = img_read[right_hand_pos_xymin[1]:right_hand_pos_xymax[1], right_hand_pos_xymin[0]:right_hand_pos_xymax[0]] 

        vis_kp_img = copy.copy(img_read)
        for pair in KEYPOINT_CONNECTION_RULES:
            name_kp1 = pair[0]
            name_kp2 = pair[1]
            color = pair[2]
            
            idx_kp1 = list(COCO_KEYPOINT_INDEXES.keys())[list(COCO_KEYPOINT_INDEXES.values()).index(name_kp1)]
            idx_kp2 = list(COCO_KEYPOINT_INDEXES.keys())[list(COCO_KEYPOINT_INDEXES.values()).index(name_kp2)]

            coord_kp1 = chosen_kp[idx_kp1]
            coord_kp2 = chosen_kp[idx_kp2]

            # connect point
            vis_kp_img = cv2.line(vis_kp_img, tuple(coord_kp1), tuple(coord_kp2), color, 4)
        
        if len(left_hand_box) == 0:
            left_hand_box = img_read 
        if len(right_hand_box) == 0:
            right_hand_box = img_read
        join_box = cv2.hconcat([left_hand_box, right_hand_box])
        path_out_full = os.path.join(path_out_vid_id, "full_img.jpg")
        path_out_join_2_hand = os.path.join(path_out_vid_id, "join_hand.jpg")
        path_out_kp = os.path.join(path_out_vid_id, "kp_vis.jpg")
        
        cv2.imwrite(path_out_kp, vis_kp_img)
        cv2.imwrite(path_out_full, img_read)
        cv2.imwrite(path_out_join_2_hand, join_box)

if __name__ == "__main__":

    video_id = ""
    root_path_json = '/home/nttung/Challenge/MediaevalSport/team_baseline/auxilary-data/out_pose_json_refined'
    root_path_vid_frame = '/home/nttung/Challenge/MediaevalSport/baseline/SportTaskME21/data'

    path_out = "./vis_kp2hand"

    if not os.path.isdir(path_out):
        os.mkdir(path_out)


    
    kp_2_hand_box(root_path_json, root_path_vid_frame, path_out)


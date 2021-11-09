def find_main_kp_frame(kp_list, w, h, video_id, frame_idx):
    
    distance_scores = []
    pose_heights = []
    confidence_scores = []
    general_distances = []
    img_center = [w/2, h/2]

    for pose in kp_list:
        center_to_left_wrist = ((pose['keypoints'][9][0] - img_center[0])**2 + (pose['keypoints'][9][1]-img_center[1])**2)**0.5
        center_to_right_wrist = ((pose['keypoints'][10][0] - img_center[0])**2 + (pose['keypoints'][10][1]-img_center[1])**2)**0.5
        
        left_shoulder_to_left_hip = ((pose['keypoints'][5][0] - pose['keypoints'][11][0])**2 + (pose['keypoints'][5][1]-pose['keypoints'][11][1])**2)**0.5
        right_shoulder_to_right_hip = ((pose['keypoints'][6][0] - pose['keypoints'][12][0])**2 + (pose['keypoints'][6][1]-pose['keypoints'][12][1])**2)**0.5

        pose_heights.append((max(left_shoulder_to_left_hip, right_shoulder_to_right_hip)))
        distance_scores.append(min(center_to_left_wrist, center_to_right_wrist))
        confidence_scores.append(pose['score'])

    for i, d in enumerate(distance_scores):
        if float(confidence_scores[i]) < 0.1:
            distance_scores.pop(i)
            pose_heights.pop(i)

    for i in range(0, len(distance_scores)):
        general_distances.append(distance_scores[i]+1/pose_heights[i])
    main_kp_frame = kp_list[general_distances.index(min(general_distances))]

    return main_kp_frame

# if __name__ == '__main__':
#     kp_list = []

#     main_kp_frame = find_main_kp_frame(kp_list, 0,0,0,0)
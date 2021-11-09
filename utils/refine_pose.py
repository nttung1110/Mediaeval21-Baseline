import os.path as osp 
import os
import numpy as np
import json
import pdb 

from pathlib import Path

def refine_pose(root_path_json, path_out_refined):
    '''
        Refine each json file in the given directory 
        by interpolating frame with intervals(default 15)
    '''
    for file_name in sorted(os.listdir(root_path_json)):
        print("Refining:",file_name)

        file_path = osp.join(root_path_json, file_name)
        # read pose data
        with open(file_path, 'r') as fp:
            pose_data = json.load(fp)

        sorted_key_frame = sorted(pose_data.keys())

        for idx in range(len(sorted_key_frame)-1):
            cur_frame_idx = int(sorted_key_frame[idx].split('/')[-1][:-4])
            next_frame_idx = int(sorted_key_frame[idx+1].split('/')[-1][:-4])

            root_frame_data = '/'.join(sorted_key_frame[idx].split('/')[:-1])

            for interpolated_frame_idx in range(cur_frame_idx+1, next_frame_idx):
                path_interpolated_frame = osp.join(root_frame_data, 
                                                    str(interpolated_frame_idx).zfill(8)+'.png')

                pose_data[path_interpolated_frame] = pose_data[sorted_key_frame[idx]]

        # write data
        pose_data = dict(sorted(pose_data.items()))
        path_write_json = osp.join(path_out_refined, file_name)
        with open(path_write_json, 'w') as fp:
            json.dump(pose_data, fp, indent=4)

    
if __name__ == '__main__':
    root_path_json = '/home/nttung/Challenge/MediaevalSport/experiment/mmpose/out_pose_json'
    path_out_refined = osp.join('/home/nttung/Challenge/MediaevalSport/team_baseline/auxilary-data',
                                root_path_json.split('/')[-1] + '_refined')

    Path(path_out_refined).mkdir(parents=True, exist_ok=True)

    refine_pose(root_path_json, path_out_refined)


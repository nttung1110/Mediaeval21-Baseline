import os.path as osp 
import json
import pdb 

def dict2xml(data_dict, path_out):
    for vid_key in data_dict.keys():
        path_out_file = osp.join(path_out, vid_key+'.xml')
        with open(path_out_file, 'w') as fp:
            print('<video>', file=fp)
            for instance in data_dict[vid_key]:
                interval = instance[0]
                label = instance[1]
                print('  <action begin="{}" end="{}" move="{}" />'.format(interval[0], interval[1], label), file=fp)
            print('</video>', file=fp)
 
def split_data():
    path_json_categories = "/home/nttung/Challenge/MediaevalSport/2020_data/data/data_json/train.json"
    path_out_train_xml = "/home/nttung/Challenge/MediaevalSport/2020_data/data/classificationTask/split_data/train"
    path_out_val_xml = "/home/nttung/Challenge/MediaevalSport/2020_data/data/classificationTask/split_data/valid"
    ratio = 0.7
    
    # read json 
    with open(path_json_categories, "r") as fp:
        json_data = json.load(fp)

    train_dict = {}
    val_dict = {}

    for cat_name in json_data.keys():
        print("Process:", cat_name)

        # count total instances for each cat 
        total_count = 0
        for vid_key in json_data[cat_name].keys():
            total_count += len(json_data[cat_name][vid_key])

        threshold_train_idx = int(total_count*ratio)

        # start to add to train and val dict
        ref_dict = train_dict # start with train
        num_instance_run = 0
        for vid_key in json_data[cat_name].keys():
            for interval in json_data[cat_name][vid_key]:
                if vid_key not in ref_dict:
                    ref_dict[vid_key] = []
                ref_dict[vid_key].append([interval, cat_name, num_instance_run])
                num_instance_run += 1

                # debug
                if num_instance_run == threshold_train_idx:
                    # switch to val dictionary
                    ref_dict = val_dict

        
    print("Write to xml....")
    # write to xml for train dict
    dict2xml(train_dict, path_out_train_xml)
    dict2xml(val_dict, path_out_val_xml)
            

if __name__ == '__main__':
    split_data()
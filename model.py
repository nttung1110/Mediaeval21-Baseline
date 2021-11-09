
from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def flatten_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class LSTM3DCNNPose(nn.Module):
    def __init__(self, embed_dim_LSTM, hidden_dim_LSTM, num_class_list, size_data, size_data_hand,
                 hidden_dim_fuse_1 = 1024, hidden_dim_fuse_2 = 128, fc_dim_LSTM=64, fc_dim_3DCNN=500, channels=3):    
        super(LSTM3DCNNPose, self).__init__()
        self.embed_dim = embed_dim_LSTM
        self.hidden_dim = hidden_dim_LSTM
        self.num_class_1 = num_class_list[0]
        self.num_class_2 = num_class_list[1]
        self.num_class_3 = num_class_list[2]
        self.size_data = size_data
        self.size_data_hand = size_data_hand
        

        '''
            Two branches: CNN and LSTM branches
                1/ 3D CNN branch produces flattened feature vector X1
                2/ LSTM  branch produces flattened feature vector X2

            X1 concatenate with X2 to construct the prediction vector
        '''
        
        # First branch of 3D CNN for 1, 2 component
        ####################
        ####### First ######
        ####################
        self.comp13_conv1 = nn.Conv3d(channels, 30, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.comp13_pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        size_data //= 2

        ####################
        ###### Second ######
        ####################
        self.comp13_conv2 = nn.Conv3d(30, 60, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) 
        self.comp13_pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        size_data //= 2

        ####################
        ####### Third ######
        ####################
        self.comp13_conv3 = nn.Conv3d(60, 80, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) 
        self.comp13_pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        size_data //= 2

        ####################
        ####### Last #######
        ####################
        self.linear_comp13_project_rgb = nn.Linear(80*size_data[0]*size_data[1]*size_data[2], fc_dim_3DCNN)

        # Second branch LSTM
        # single LSTM layer
        self.lstm = nn.LSTM(embed_dim_LSTM, hidden_dim_LSTM, batch_first=True)
        # context and hidden
        self.linear_project_pose = nn.Linear(hidden_dim_LSTM*2, fc_dim_LSTM)

        ###### Fuse two branch for two different prediction head ######
        # First component's prediction head
        
        self.linear_comp_1_fuse_1 = nn.Linear(fc_dim_3DCNN+fc_dim_LSTM, hidden_dim_fuse_1)
        self.linear_comp_1_fuse_2 = nn.Linear(hidden_dim_fuse_1, hidden_dim_fuse_2)
        self.out_head_1 = nn.Linear(hidden_dim_fuse_2, self.num_class_1)

        # Second component's prediction head
        
        self.linear_comp_3_fuse_1 = nn.Linear(fc_dim_3DCNN+fc_dim_LSTM, hidden_dim_fuse_1)
        self.linear_comp_3_fuse_2 = nn.Linear(hidden_dim_fuse_1, hidden_dim_fuse_2)
        self.out_head_3 = nn.Linear(hidden_dim_fuse_2, self.num_class_3)

        ##################################################
        # Second branch of 3D CNN for component 2
        ####################
        ####### First ######
        ####################
        self.comp2_conv1 = nn.Conv3d(channels, 30, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.comp2_pool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        size_data_hand //= 2

        ####################
        ###### Second ######
        ####################
        self.comp2_conv2 = nn.Conv3d(30, 60, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) 
        self.comp2_pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        size_data_hand //= 2

        ####################
        ####### Third ######
        ####################
        self.comp2_conv3 = nn.Conv3d(60, 80, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)) 
        self.comp2_pool3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        size_data_hand //= 2

        ####################
        ####### Last #######
        ####################
        self.linear_comp2_project_rgb = nn.Linear(80*size_data_hand[0]*size_data_hand[1]*size_data_hand[2], fc_dim_3DCNN)

        self.linear_comp_2_1 = nn.Linear(fc_dim_3DCNN, hidden_dim_fuse_1)
        self.linear_comp_2_2 = nn.Linear(hidden_dim_fuse_1, hidden_dim_fuse_2)
        self.out_head_2 = nn.Linear(hidden_dim_fuse_2, self.num_class_2)

    def forward(self, rgb_sequence, hand_sequence, pose_sequence):

        # 3DCNN branch for 1, 2
        ####################
        ####### First ######
        ####################
        data = self.comp13_pool1(F.relu(self.comp13_conv1(rgb_sequence)))

        ####################
        ###### Second ######
        ####################
        data = self.comp13_pool2(F.relu(self.comp13_conv2(data)))

        ####################
        ####### Third ######
        ####################
        data = self.comp13_pool3(F.relu(self.comp13_conv3(data)))

        ####################
        ####### Last #######
        ####################
        if len(data.shape) == 1:
            data = data.unsqueeze(0)

        data = data.view(-1, flatten_features(data))
        out_3DCNN_comp13_vec = F.relu(self.linear_comp13_project_rgb(data))

        # LSTM branch

        # refer to https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html for more information
        # pdb.set_trace()

        # zero initialization for hidden state of lstm
        '''
            Experiment should be conducted:
            1/ >>>failed>>> Take only the final hidden state as input to flc 
            2/ >>running>>Take both cell and hidden state of final element in the sequence ...
        '''
        _, (hn, cn) = self.lstm(pose_sequence)
        hn = hn.squeeze()
        cn = cn.squeeze()

        if len(hn.shape) == 1: 
            hn = hn.unsqueeze(0)
        if len(cn.shape) == 1:
            cn = cn.unsqueeze(0)

        out_LSTM_vec = torch.cat((hn, cn), 1)

        out_LSTM_vec = F.relu(self.linear_project_pose(out_LSTM_vec)).squeeze()

        if len(out_LSTM_vec.shape) == 1:
            out_LSTM_vec = out_LSTM_vec.unsqueeze(0)

        '''
          
            The above if statement is due to the note from Pytoch:
            
                torch.nn only supports mini-batches. The entire torch.nn package only supports inputs that are a
                mini-batch of samples, and not a single sample.

                For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.

                If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.

            Access https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html for more information

        '''

        # Fuse information for different prediction head: x1, x2, and x3

        x = torch.cat((out_3DCNN_comp13_vec, out_LSTM_vec), 1)
        
        # First prediction head
        x1 = F.relu(self.linear_comp_1_fuse_1(x))
        x1 = F.relu(self.linear_comp_1_fuse_2(x1))

        x1 = self.out_head_1(x1).squeeze()

        if len(x1.shape) == 1:
            x1 = x1.unsqueeze(0)

        # Second prediction head
        x3 = F.relu(self.linear_comp_3_fuse_1(x))
        x3 = F.relu(self.linear_comp_3_fuse_2(x3))

        x3 = self.out_head_3(x3).squeeze()

        if len(x3.shape) == 1:
            x3 = x3.unsqueeze(0)

        #####################################
        # 3DCNN branch for 3
        ####################
        ####### First ######
        ####################
        data_comp2 = self.comp2_pool1(F.relu(self.comp2_conv1(hand_sequence)))

        ####################
        ###### Second ######
        ####################
        data_comp2 = self.comp2_pool2(F.relu(self.comp2_conv2(data_comp2)))

        ####################
        ####### Third ######
        ####################
        data_comp2 = self.comp2_pool3(F.relu(self.comp2_conv3(data_comp2)))

        ####################
        ####### Last #######
        ####################
        if len(data_comp2.shape) == 1:
            data_comp2 = data_comp2.unsqueeze(0)

        data_comp2 = data_comp2.view(-1, flatten_features(data_comp2))
        out_comp2_3DCNN_vec = F.relu(self.linear_comp2_project_rgb(data_comp2))
        # Third prediction head
        x2 = F.relu(self.linear_comp_2_1(out_comp2_3DCNN_vec))
        x2 = F.relu(self.linear_comp_2_2(x2))

        x2 = self.out_head_2(x2).squeeze()

        if len(x2.shape) == 1:
            x2 = x2.unsqueeze(0)
        
        return x1, x2, x3


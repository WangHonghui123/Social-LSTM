import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time

MAX_NODE = 57


def sinkhorn(log_alpha, n_iters=5):
    n = log_alpha.shape[1]
    log_alpha = log_alpha.view(-1, n, n)
    for i in range(n_iters):
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True)).view(
            -1, n, 1
        )
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(
            -1, 1, n
        )
    return torch.exp(log_alpha)


def rotate_pc(coords, alpha):
    alpha = alpha * np.pi / 180
    M = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    return M @ coords


def torch_seq_to_nodes(seq_):
    seq_ = seq_.squeeze()
    batch = seq_.shape[0]
    seq_len = seq_.shape[1]
    num_ped = seq_.shape[2]

    V = torch.zeros((batch, seq_len, num_ped, 2), requires_grad=True).cuda()
    for s in range(seq_len):
        step_ = seq_[:, s, :, :]
        for h in range(num_ped):
            V[:, s, h, :] = step_[:, h]

    return V.squeeze()


def torch_nodes_rel_to_nodes_abs(nodes, init_node):
    """
    batch enable funct
    """

    nodes_ = torch.zeros_like(nodes, requires_grad=True).cuda()
    """
    nodes: [batch, seq_len, num_ped, feat]
    init : [batch, seq_len, num_ped, feat]
    """

    for s in range(nodes.shape[1]):
        for ped in range(nodes.shape[2]):
            nodes_[:, s, ped, :] = (
                torch.sum(nodes[:, : s + 1, ped, :], axis=1) + init_node[:, ped, :]
            )

    return nodes_.squeeze()


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)


def torch_anorm(p1, p2):
    NORM = torch.sqrt((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2)
    rst = torch.where(NORM != 0.0, (1 / NORM).data, NORM.data)

    return rst


def seq_to_graph(
    seq_,
    seq_rel,
    norm_lap_matr=True,
    alloc=False,
):
    """
    Pytorch Version;
    For this function, input pytorch tensor:
        (seq_rel) has shape [num_ped, 2, seq_len]
    """
    norm_lap_matr = False
    # norm_lap_matr = True

    seq_ = seq_.squeeze() #(pedestrian, normalized position, frame)
    V = seq_rel.permute(2, 0, 1) #(frame, pedestrian, relative position)
    # seq_rel = seq_rel.squeeze()
    """ Decide if use real coords for adj computation or not """
    seq_rel = seq_.clone()

    if len(seq_.shape) < 3: #Only a pedestrian in this sequence
        seq_ = seq_.unsqueeze(-1)
        seq_rel = seq_rel.unsqueeze(-1)

    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    seq_rel = (
        seq_rel.permute(2, 0, 1).unsqueeze(-2).repeat(1, 1, max_nodes, 1)
    )  # convert to [seq_len, node, node, feat]
    # 1. Find relative coordinates in this way;
    # 2. Reduce centroid position information?
    # 3. Trans_seq_rel = seq_rel.permute(0, 2, 1, 3) - center
    trans_seq_rel = seq_rel.permute(0, 2, 1, 3) #Change the index between nodes: (0-0; 0-1; 0-2...) -> (0-0; 1-0; 2-0..)

    # calc relative
    # seq_rel_r is the distance between pedestrians under a frame
    seq_rel_r = seq_rel - trans_seq_rel
    seq_rel_r = torch.sqrt(seq_rel_r[..., 0] ** 2 + seq_rel_r[..., 1] ** 2)

    # set threshold to the number of neighbors?
    # seq_rel_r  = torch.where(seq_rel_r > 3, torch.zeros(1), seq_rel_r)

    """ Find the inverse """
    # When seq_rel_r != 0, seq_rel_r = 1/seq_rel_r or seq_rel_r = 0
    seq_rel_r = torch.where(seq_rel_r != 0.0, (1 / seq_rel_r).data, seq_rel_r.data)

    """ How to deal with dist > 1?, which will lead to unstable? """
    """ Will this play an important factor?(Set the distance between pedestrians under a frame to be [0,1], which can become stable)  """
    if seq_rel.is_cuda:
        # When seq_rel_r > 1.0，seq_rel_r == 1 or unchanged
        seq_rel_r = torch.where(
            # seq_rel_r > 1.0, seq_rel_r, torch.ones(1).cuda())
            seq_rel_r > 1.0,
            torch.ones(1).cuda(),
            seq_rel_r,
        )
    else:
        seq_rel_r = torch.where(
            # seq_rel_r > 1.0, seq_rel_r, torch.ones(1))
            seq_rel_r > 1.0,
            torch.ones(1),
            seq_rel_r,
        )

    """Normalized based on the largest value in column?"""
    # max_column, _ = torch.max(seq_rel_r, -1, keepdim=True)
    # seq_rel_r /= max_column

    if seq_rel.is_cuda:
        diag_ones = torch.eye(max_nodes).cuda()
    else:
        diag_ones = torch.eye(max_nodes)
    seq_rel_r[:, :] = seq_rel_r[:, :] + diag_ones #The distance between himself is 1 (diagonal element == 1)

    A = seq_rel_r

    if norm_lap_matr:
        """
        Laplacian from graph matrix, as in
        1). https://github.com/dimkastan/PyTorch-Spectral-clustering/blob/master/FiedlerVectorLaplacian.py;
        2). https://github.com/huyvd7/pytorch-deepglr/blob/master/deepglr/deepglr.py;
        3). https://github.com/huyvd7/pytorch-deepglr
        """
        A_sumed = torch.sum(A, axis=1).unsqueeze(-1)
        diag_ones_tensor = diag_ones.unsqueeze(0).repeat(seq_len, 1, 1)
        D = diag_ones_tensor * A_sumed
        DH = torch.sqrt(D)
        DH = torch.where(DH != 0, 1.0 / DH, DH)  # avoid inf values
        L = D - A
        A = torch.bmm(DH, torch.bmm(L, DH))
    # else:
    # A = torch.where(A != 0, 1.0/A, A)

    if alloc:
        # for now, adj_rel_shift only admit numpy data;
        A_alloc = adj_rel_shift(seq_rel_r.cpu().numpy())
        A_alloc = torch.from_numpy(A_alloc)

        if norm_lap_matr:
            A_sumed = torch.sum(A_alloc, axis=1).unsqueeze(-1)
            diag_ones_tensor = diag_ones.unsqueeze(0).repeat(seq_len, 1, 1)
            D = diag_ones_tensor * A_sumed
            DH = 1.0 / torch.sqrt(D)
            DH[torch.isinf(DH)] = 0.0
            L = D - A_alloc
            A_alloc = torch.bmm(DH, torch.bmm(L, DH))

        return V, A, A_alloc
    return V, A, A  # the last A acts like padding for A_alloc


def adj_rel_shift(A):
    """shift adj edges w.r.t. cloest neighbor

    # A: adj matrix, [batch, seq_len, num_ped, num_ped]
    A: adj matrix, [seq_len, num_ped, num_ped]


    Notice: the ped could be padded ?? I guess no, this has to happen in preprocessing:

    procedure:
    1). Find cloest neighbor for each nodes;
    2). Replace edges of all other neighbors, except the cloest one, with those of neighbors;
    3). See what happens; (Although, numberically difference is very small;)
    4). numpy.take_along_axis seems a answer;


    Update: use numpy instead of pytorch
    """

    seq_len, num_ped = A.shape[:2]

    # min_values, indices = torch.min(A, dim=-1)
    indices = np.argmin(A, axis=-1)
    min_values = np.min(A, axis=-1)
    # target_index = len(indices.shape)
    # tmp = A.clone()

    # replace current row with min value row, -1 works for second last axis;
    # A[:, :, :] = tmp.gather(target_index-1, indices)
    # out = tmp.gather(2, indices.unsqueeze(-1))
    # indices = indices.unsqueeze(-1).repeat(1, 1, 1, num_ped)
    # indices[:, :, :] = torch.range(
    # 0, num_ped-1, dtype=torch.int32).view(1, 1, 1, num_ped).repeat(batch, seq_len, num_ped, 1)
    # import pdb
    # pdb.set_trace()
    # out = tmp.gather(2, indices)
    indices = np.expand_dims(indices, -1)
    min_values = np.expand_dims(min_values, -1)
    out = np.take_along_axis(A, indices, 1)

    # replace diagonal with zeros, since it should be
    # idenity = torch.eye(num_ped)
    idenity = np.eye(num_ped)
    ivt_idenity = np.where(idenity != 0, 0, 1)
    # ivt_idenity = ivt_idenity.unsqueeze(
    # 0).unsqueeze(0).repeat(batch, seq_len, 1, 1)
    out = out * ivt_idenity

    # replace the anchor nodes with one:
    np.put_along_axis(out, indices, min_values, axis=-1)

    # return A
    return out


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1] #Calculate the residual x when approximating quadratic curve
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1] #Calculate the residual y when approximating quadratic curve
    if res_x + res_y >= threshold:
        return 1.0 #When the residual is too large (>= threshold), trajectory is non-linear
    else:
        return 0.0 #When the residual is < threshold, trajectory is linear


def read_file(_path, delim="\t"):
    data = []
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "
    with open(_path, "r") as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
        self,
        data_dir,
        obs_len=8,
        pred_len=8,
        skip=1,
        threshold=0.002,
        min_ped=1,
        delim="\t",
        norm_lap_matr=True,
        alloc=False,
        angles=[0],
        grad_eff=0.4,
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a sequence
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        global MAX_NODE

        self.max_peds_in_frame = 0
        self.data_dir = data_dir #Get the file root directory
        self.obs_len = obs_len #frame of observed trajectory
        self.pred_len = pred_len #frame of predicted trajectory
        self.skip = skip #Number of frames to skip while making the dataset
        judge_test_train_index = data_dir[:-1].rfind('/')
        judge_test_train = data_dir[judge_test_train_index+1:]
        if judge_test_train == 'test/':
            self.seq_len = self.obs_len + self.pred_len  # the frame of a trajectory
        else:
            self.seq_len = self.obs_len + self.pred_len + 1 # the frame of a trajectory
        self.delim = delim # delimiter in the dataset file
        self.norm_lap_matr = norm_lap_matr
        self.alloc = alloc

        all_files = os.listdir(self.data_dir)
        all_files = [
            os.path.join(self.data_dir, _path)
            for _path in all_files
            if _path.endswith("txt")
        ] #Get the train/validation/test dataset files
        num_peds_in_seq = []
        seq_list = []
        seq_list_ori = []
        seq_list_rel = []
        seq_list_v = []
        seq_list_start = []
        seq_list_end = []
        seq_ped_index = []

        loss_mask_list = []
        non_linear_ped = []

        # data augmentation(rotation)
        # angles = np.arange(0, 360, 15) if "test" not in self.data_dir else [0]
        angles = [0]  # rotated angle is 0, which means unchanged

        # data augmentation(amplify)
        amplify = [1] # amplify scale is 1, which means unchanged
        # amplify = np.arange(1, 8, 0.5) if "test" not in self.data_dir else [0]

        data_scale = 1.0 #data augmentation(lessen)

        for path in all_files: #Get the dataset file
            # Get the original dataset
            # There are four columns. First column is frame; Second column is pedid; Third and Fourth column is coordinate
            print(f'The processing file is {path}')
            data_ori = read_file(path, delim)
            for angle in angles: #Dataset rotation and scale
                # data = data_ori.copy()
                data = np.copy(data_ori) * data_scale
                # Can I perform rotation here? It seems that I can, let me give a try;
                data[:, -2:] = rotate_pc(data[:, -2:].transpose(), angle).transpose() #The rotation on the coordinate

                for amp in amplify: #Amplify on the dataset

                    if "test" not in self.data_dir:
                        data[:, -2:] *= np.array((amp, amp)) #Amplify on the coordinate (No change in this code)

                    frames = np.unique(data[:, 0]).tolist() #Get all frames
                    frame_data = []

                    for frame in frames:
                        frame_data.append(data[frame == data[:, 0], :]) #frame_data is obtained in line with frames
                    num_sequences = int(
                        math.ceil((len(frames) - self.seq_len + 1) / skip)
                    ) #sequence(一个sequence为20个frame)的数量

                    for idx in range(0, num_sequences * self.skip + 1, skip):
                        # The sequence is 0-19s; 1-20s; 2-21s;... 853-872s;
                        # First column is frame; Second column is pedid; Third and Fourth column is coordinate
                        curr_seq_data = np.concatenate(
                            frame_data[idx : idx + self.seq_len], axis=0
                        ) #Get the current sequence (20 frames in a sequence)
                        peds_in_curr_seq = np.unique(curr_seq_data[:, 1]) #Get pedid in the current sequence
                        self.max_peds_in_frame = max(
                            self.max_peds_in_frame, len(peds_in_curr_seq)
                        ) #The maximun pedestrians in the current sequence

                        # All data structure: (pedid, 2, seq)
                        curr_seq_rel = np.zeros(
                            (len(peds_in_curr_seq), 2, self.seq_len)
                        ) #The relative position
                        curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len)) #Position
                        curr_ori_seq = np.zeros((len(peds_in_curr_seq), 3, self.seq_len)) #Pedid with position
                        curr_seq_v = np.zeros((len(peds_in_curr_seq), 2, self.seq_len)) #velocity
                        curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len)) #mask loss(related to attention mechanism)
                        curr_seq_start = np.zeros((len(peds_in_curr_seq), 2)) #original position
                        curr_seq_end = np.zeros((len(peds_in_curr_seq), 2))
                        num_peds_considered = 0 #the number of pedestrians considered
                        _non_linear_ped = [] #trajectory is linear or non-linear
                        curr_peds_index = [] #Store the pedestrian index under a frame
                        # Loop all pedestrians in curr sequence;
                        for _, ped_id in enumerate(peds_in_curr_seq): #Process pedestrians in a sequence
                            curr_ped_seq = curr_seq_data[
                                curr_seq_data[:, 1] == ped_id, :
                            ]
                            curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                            pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                            pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                            #Filter out the trajectory with less than self.seq_len (20 or 21)
                            if pad_end - pad_front != self.seq_len:
                                continue
                            curr_ped_seq_ori = np.transpose(np.copy(curr_ped_seq[:, 1:])) #Pedestrian with position in the current sequence
                            curr_ped_seq = np.transpose(curr_ped_seq[:, 2:]) # position in the current sequence

                            ## Keep record of initial point
                            seq_start = np.array(
                                (curr_ped_seq[0, 0], curr_ped_seq[1, 0])
                            ) #The starting position

                            seq_end = np.array(
                                (curr_ped_seq[0,-1], curr_ped_seq[1,-1])
                            ) #The end position

                            #The normalized trajectory
                            curr_ped_seq[0, :] = curr_ped_seq[0, :] - curr_ped_seq[0, 0]
                            curr_ped_seq[1, :] = curr_ped_seq[1, :] - curr_ped_seq[1, 0]


                            # Make coordinates relative
                            rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)

                            # How about use some simple direct information?
                            rel_curr_ped_seq[:, 1:] = (
                                curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                            )

                            # np.gradient = (f(x+1) - f(x-1))/(2h) + o(h^2) for x is not within the boundary
                            # np.gradient = (f(x+1) - f(x))/(h) + o(h^2) for x is within the boundary
                            # grad_eff = h
                            # axis = 1 means processing column only
                            # v_curr_ped_seq is a velocity
                            v_curr_ped_seq = np.gradient(
                                np.array(curr_ped_seq),
                                grad_eff,
                                # 0.2,
                                axis=1
                                # np.array(curr_ped_seq),
                                # 0.2,
                                # axis=1,
                            )

                            _idx = num_peds_considered
                            curr_seq_start[_idx, :] = seq_start
                            curr_seq_end[_idx, :] = seq_end
                            curr_ori_seq[_idx, :, pad_front:pad_end] = curr_ped_seq_ori
                            curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                            curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                            curr_seq_v[_idx, :, pad_front:pad_end] = v_curr_ped_seq

                            # Linear vs Non-Linear Trajectory
                            _non_linear_ped.append(
                                poly_fit(curr_ped_seq, pred_len, threshold)
                            )
                            curr_loss_mask[_idx, pad_front:pad_end] = 1
                            curr_peds_index.append(_idx)
                            num_peds_considered += 1

                        # if num_peds_considered > min_ped:
                        min_ped = 0
                        # min_ped = 1
                        max_ped = 1000
                        flip = False
                        if (
                            num_peds_considered > min_ped
                            and num_peds_considered <= max_ped
                        ):
                            non_linear_ped.append(_non_linear_ped)
                            num_peds_in_seq.append(num_peds_considered)
                            loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                            seq_list.append(curr_seq[:num_peds_considered])
                            seq_list_ori.append(curr_ori_seq[:num_peds_considered])
                            seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                            seq_list_v.append(curr_seq_v[:num_peds_considered])
                            seq_list_start.append(curr_seq_start[:num_peds_considered])
                            seq_list_end.append(curr_seq_end[:num_peds_considered])

                            # Because we filter out all trajectories without 20 frames, the pedestrian index is all the same under a sequence
                            a = [curr_peds_index for _ in range(self.seq_len)]
                            seq_ped_index.append(a) #Store all pedestrian index under a sequence

                            if flip and "train" in self.data_dir:
                                non_linear_ped.append(_non_linear_ped)
                                num_peds_in_seq.append(num_peds_considered)
                                loss_mask_list.append(
                                    curr_loss_mask[:num_peds_considered]
                                )
                                seq_list.append(
                                    np.flip(curr_seq[:num_peds_considered], 2)
                                )
                                seq_list_ori.append(np.flip(curr_ori_seq[:num_peds_considered], 2))
                                seq_list_rel.append(
                                    np.flip(curr_seq_rel[:num_peds_considered], 2)
                                )
                                seq_list_v.append(
                                    np.flip(curr_seq_v[:num_peds_considered], 2)
                                )

        # For vanilla eth dataset, there are 2785 dataset;
        # For angle augmented eth dataset, there are 24 times 2785 = 66840;
        self.num_seq = len(seq_list)
        self.sequence = seq_list
        self.ori_sequence = seq_list_ori
        self.traj_velocity = seq_list_v
        self.traj_start = seq_list_start
        self.traj_end = seq_list_end
        self.loss_mask = loss_mask_list
        self.non_linear_ped = non_linear_ped
        self.seq_ped_index = seq_ped_index

    def __len__(self):
        num_seq = self.num_seq
        return num_seq

    def __getitem__(self, index):
        out = [
            self.sequence[index],
            self.traj_velocity[index],
            self.traj_start[index],
            self.traj_end[index],
            self.loss_mask[index],
            self.non_linear_ped[index],
        ]
        return out




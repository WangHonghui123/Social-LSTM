'''
Train script for the Social LSTM model

Author: Honghui Wang
Date: 5th April 2023
'''
import torch
from torch.autograd import Variable

import argparse
import os
import time
import pickle

from model import SocialLSTM
from grid import getSequenceGridMask
from train_helper import Gaussian2DLikelihood
import numpy as np
import random
from utils_expert import *
from metric import *



def main(dataset_name):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=8,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=20,
                        help='RNN sequence length')
    parser.add_argument('--obs_length', type=int, default=8,
                        help='Observed length of the trajectory')
    parser.add_argument('--pred_length', type=int, default=12,
                        help='prediction length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout not implemented.
    # Dropout probability parameter
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=int, default=32,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid')
    # The leave out dataset
    parser.add_argument('--leaveDataset', type=int, default=3,
                        help='The dataset index to be left out in training')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.0001,
                        help='L2 regularization parameter')
    args = parser.parse_args()

    # dataset_name = "univ"
    # dataset_name = "eth"
    # dataset_name = "zara1"
    # dataset_name = "zara2"
    # dataset_name = "hotel"

    '''
    For dataset (train set and validation set)
    '''
    # Load the dataset (train set and validation set)
    dataset_path = "../datasets/" + dataset_name + "/"

    # Process TRAIN dataset and store it to the train_dataset object
    grad_eff = 0.4 # one frame is 0.4 second
    train_dataset = TrajectoryDataset(
        dataset_path + "train/",
        obs_len=args.obs_length,
        pred_len=args.pred_length,
        skip=1,
        norm_lap_matr=True,
        grad_eff=grad_eff,
    )

    # Process VALIDATION dataset and store it to the val_dataset object
    grad_eff = 0.4 # one frame is 0.4 second
    val_dataset = TrajectoryDataset(
        dataset_path + "val/",
        obs_len=args.obs_length,
        pred_len=args.pred_length,
        skip=1,
        norm_lap_matr=True,
        grad_eff=grad_eff,
    )

    '''
    For log (train and validation)
    '''
    # Setup for train log
    log_train_directory_path = "./log/" + dataset_name + "/train/"
    #If the directory does not exist, we need to create it using the following code
    if not os.path.exists(log_train_directory_path):
        os.makedirs(log_train_directory_path)

    log_train_file = open(os.path.join(log_train_directory_path, 'train_log.txt'), 'w')
    train_metrics = {'train_loss': []}


    #Setup for validation log
    log_val_directory_path = "./log/" + dataset_name + "/val/"
    #If the directory does not exist, we need to create it using the following code
    if not os.path.exists(log_val_directory_path):
        os.makedirs(log_val_directory_path)

    log_val_file = open(os.path.join(log_val_directory_path, 'val_log.txt'), 'w')
    val_metrics = {'val_loss': []}

    '''
    For saving model
    '''
    save_directory = "./save/" + dataset_name + '/'
    # If the directory does not exist, we need to create it using the following code
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Path to store model file
    def checkpoint_path(x):
        return os.path.join(save_directory, 'social_lstm_model_' + str(x) + '.tar')

    '''
    For the parameter of parsing module
    '''
    # Dump the arguments into the configuration file
    with open(os.path.join(save_directory, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)


    '''
    For network model
    '''
    net = SocialLSTM(args)
    # net.cuda()

    '''
    For training settings
    '''
    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)

    print('Training started ...')
    for epoch in range(args.num_epochs):
        # Training
        net, optimizer, train_metrics = train(epoch, train_dataset, net, optimizer, args, train_metrics, dataset_name)
        # Print the epoch loss
        print('--------------------TRAIN-----------------------------')
        print('TRAIN:', '\t Epoch:', epoch, '\t Epoch_Loss:', train_metrics['train_loss'][epoch])
        print('--------------------TRAIN----------------------------------')
        # Write the epoch loss for training
        log_train_file.write('epoch: ' + str(epoch)+' Epoch_Loss: '+str(train_metrics['train_loss'][epoch])+'\n')

        # Save model
        print('------------------------Saving model--------------------------')
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))
        print('------------------------Saving model------------------------------')

        #Validation
        val_metrics = Validation(epoch, val_dataset, net, args, val_metrics, dataset_name)
        # Print the epoch loss for validation
        print('--------------------VALIDATION-----------------------------')
        print('VALIDATION:', '\t Epoch:', epoch, '\t Epoch_Loss:', val_metrics['val_loss'][epoch])
        print('--------------------VALIDATION----------------------------------')
        # Write the epoch loss for validation
        log_val_file.write('epoch: ' + str(epoch)+' Epoch_Loss: '+str(val_metrics['val_loss'][epoch]) +'\n')

    # Print and write best training epoch loss
    print('------------------------------------------------------------------')
    best_train_loss = min(train_metrics['train_loss'])
    best_train_loss_index = train_metrics['train_loss'].index(best_train_loss)
    print(f'The best training epoch is: {best_train_loss_index};  The best training loss is: {best_train_loss}')
    log_train_file.write('The best training epoch is: ' + str(best_train_loss_index) + ' The best training loss is: ' + str(best_train_loss))
    log_train_file.close()
    # Print and write best validation epoch loss
    print('------------------------------------------------------------------')
    best_val_loss = min(val_metrics['val_loss'])
    best_val_loss_index = val_metrics['val_loss'].index(best_val_loss)
    print(f'The best validation epoch is: {best_val_loss_index};  The best validation loss is: {best_val_loss}')
    log_val_file.write('The best validation epoch is: ' + str(best_val_loss_index) + ' The best validation loss is: ' + str(best_val_loss))
    log_val_file.close()

    #Store the train and validation epoch loss
    with open(os.path.join(log_train_directory_path, 'train_loss.pkl'), 'wb') as f:
        pickle.dump(train_metrics, f)

    with open(os.path.join(log_val_directory_path, 'val_loss.pkl'), 'wb') as f:
        pickle.dump(val_metrics, f)


def train(epoch, train_dataset, net, optimizer, args, train_metrics, dataset_name):
    net.train()
    # Set the dimension of dataset, that is the region of scene in the dataset
    dataset_dimensions = {'eth': [720, 576], 'univ': [720, 576], 'zara1': [720, 576],
                          'zara2': [720, 576],'hotel': [720, 576]}
    dataset_data = dataset_dimensions[dataset_name]

    # calculate the number of batches
    num_seq = train_dataset.num_seq
    if num_seq % args.batch_size == 0:
        num_batches = num_seq / args.batch_size
        batch_set = [args.batch_size] * int(num_batches)
    else:
        num_batches = np.floor(num_seq / args.batch_size) + 1
        batch_set = [args.batch_size] * int(num_batches)
        batch_set[-1] = num_seq - int(np.floor(num_seq / args.batch_size)) * args.batch_size

    loss_epoch = 0
    left_index = 0 #The first index in a batch
    for batch_num, batch in enumerate(batch_set):
        # Get the batch data
        right_index = left_index + batch - 1 #The last index in a batch
        traj_ori_sequence = train_dataset.ori_sequence[left_index:right_index + 1] #The original trajectory
        traj_sequence = train_dataset.sequence[left_index:right_index + 1] #The normalized trajectory
        traj_velocity = train_dataset.traj_velocity[left_index:right_index + 1] #The trajectory velocity
        traj_start = train_dataset.traj_start[left_index:right_index + 1] #The starting position of trajectory
        traj_end = train_dataset.traj_end[left_index:right_index + 1] #The end position of trajectory
        traj_node_index = train_dataset.seq_ped_index[left_index:right_index + 1]
        left_index = right_index + 1

        # calculate the batch loss
        loss_batch = 0
        start = time.time()
        for sequence in range(batch):
            # get the sequence
            ori_seq = traj_ori_sequence[sequence]
            traj_seq = traj_sequence[sequence]
            traj_v = traj_velocity[sequence]
            traj_s = traj_start[sequence]
            traj_e = traj_end[sequence]
            traj_node_present = traj_node_index[sequence]

            # Compute grid masks
            grid_seq = getSequenceGridMask(ori_seq, dataset_data, args.neighborhood_size, args.grid_size)

            # Construct variables
            nodes = np.transpose(traj_seq, (2, 0, 1))
            # nodes = Variable(torch.from_numpy(nodes).float()).cuda()
            nodes = Variable(torch.from_numpy(nodes).float())
            numNodes = nodes.size()[1]
            # hidden_states = Variable(torch.zeros(numNodes, args.rnn_size)).cuda()
            hidden_states = Variable(torch.zeros(numNodes, args.rnn_size))
            # cell_states = Variable(torch.zeros(numNodes, args.rnn_size)).cuda()
            cell_states = Variable(torch.zeros(numNodes, args.rnn_size))

            # Zero out gradients
            # net.zero_grad()
            optimizer.zero_grad()

            # Forward prop
            outputs, _, _ = net(nodes[:-1], grid_seq[:-1], traj_node_present[:-1], hidden_states, cell_states)

            # Compute loss
            loss = Gaussian2DLikelihood(outputs, nodes[1:], traj_node_present[1:], args.pred_length)
            loss_batch += loss

        # Calculate the mini-batch loss
        loss_batch = loss_batch / batch

        # Compute gradients
        loss_batch.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(net.parameters(), args.grad_clip)

        # Update parameters
        optimizer.step()

        # Calculate the epoch loss
        loss_epoch += loss_batch.item()

        end = time.time()


        print('TRAIN: batch: {} / total_batch: {} (epoch {}), train_batch_loss = {:.3f}, time/batch = {:.3f}'.format(
            epoch * len(batch_set) + batch_num,
            args.num_epochs * len(batch_set),
            epoch,
            loss_batch.item(),
            end - start))

    loss_epoch /= len(batch_set)

    train_metrics['train_loss'].append(loss_epoch)
    return net, optimizer, train_metrics



def Validation(epoch, val_dataset, net, args, val_metrics, dataset_name):
    net.eval()
    # Set the dimension of dataset, that is the region of scene in the dataset
    dataset_dimensions = {'eth': [720, 576], 'univ': [720, 576], 'zara1': [720, 576],
                          'zara2': [720, 576],'hotel': [720, 576]}
    dataset_data = dataset_dimensions[dataset_name]

    # calculate the number of batches
    num_seq = val_dataset.num_seq
    if num_seq % args.batch_size == 0:
        num_batches = num_seq / args.batch_size
        batch_set = [args.batch_size] * int(num_batches)
    else:
        num_batches = np.floor(num_seq / args.batch_size) + 1
        batch_set = [args.batch_size] * int(num_batches)
        batch_set[-1] = num_seq - int(np.floor(num_seq / args.batch_size)) * args.batch_size

    loss_epoch = 0
    left_index = 0 #The first index in a batch
    for batch_num, batch in enumerate(batch_set):
        # Get the batch data
        right_index = left_index + batch - 1 #The last index in a batch
        traj_ori_sequence = val_dataset.ori_sequence[left_index:right_index + 1] #The original trajectory
        traj_sequence = val_dataset.sequence[left_index:right_index + 1] #The normalized trajectory
        traj_velocity = val_dataset.traj_velocity[left_index:right_index + 1] #The trajectory velocity
        traj_start = val_dataset.traj_start[left_index:right_index + 1] #The starting position of trajectory
        traj_end = val_dataset.traj_end[left_index:right_index + 1] #The end position of trajectory
        traj_node_index = val_dataset.seq_ped_index[left_index:right_index + 1]
        left_index = right_index + 1

        # calculate the batch loss
        loss_batch = 0
        start = time.time()

        for sequence in range(batch):
            # get the sequence
            ori_seq = traj_ori_sequence[sequence]
            traj_seq = traj_sequence[sequence]
            traj_v = traj_velocity[sequence]
            traj_s = traj_start[sequence]
            traj_e = traj_end[sequence]
            traj_node_present = traj_node_index[sequence]

            # Compute grid masks
            grid_seq = getSequenceGridMask(ori_seq, dataset_data, args.neighborhood_size, args.grid_size)

            # Construct variables
            nodes = np.transpose(traj_seq, (2, 0, 1))
            # nodes = Variable(torch.from_numpy(nodes).float()).cuda()
            nodes = Variable(torch.from_numpy(nodes).float())
            numNodes = nodes.size()[1]
            # hidden_states = Variable(torch.zeros(numNodes, args.rnn_size)).cuda()
            hidden_states = Variable(torch.zeros(numNodes, args.rnn_size))
            # cell_states = Variable(torch.zeros(numNodes, args.rnn_size)).cuda()
            cell_states = Variable(torch.zeros(numNodes, args.rnn_size))

            # Forward prop
            outputs, _, _ = net(nodes[:-1], grid_seq[:-1], traj_node_present[:-1], hidden_states, cell_states)

            # Compute loss
            loss = Gaussian2DLikelihood(outputs, nodes[1:], traj_node_present[1:], args.pred_length)
            loss_batch += loss


        # Calculate the mini-batch loss
        loss_batch = loss_batch / batch

        # Calculate the epoch loss
        loss_epoch += loss_batch.item()

        end = time.time()

        print('VALIDATION: batch: {} / total_batch: {} (epoch {}), validation_batch_loss = {:.3f}, time/batch = {:.3f}'.format(
            epoch * len(batch_set) + batch_num,
            args.num_epochs * len(batch_set),
            epoch,
            loss_batch.item(),
            end - start))

    loss_epoch /= len(batch_set)
    val_metrics['val_loss'].append(loss_epoch)

    return val_metrics




if __name__ == '__main__':
    #Set the fixed random seed
    torch.manual_seed(42)
    #torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    dataset = ['univ']
    # dataset = ['univ','eth','zara1','zara2','hotel']
    for dataset_name in dataset:
        print(f'We are training {dataset_name}...........\n')
        main(dataset_name)

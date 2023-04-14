import random
import torch
import numpy as np
from utils_expert import *
import pickle
import argparse
from model import SocialLSTM
from grid import getSequenceGridMask, getGridMaskInference
from torch.autograd import Variable
from metric import *
from test_helper import sample
from statistics import mean



def main(dataset_name):

    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=8,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=12,
                        help='Predicted length of the trajectory')
    # Test dataset
    parser.add_argument('--test_dataset', type=int, default=3,
                        help='Dataset to be tested on')

    # Model to be loaded
    parser.add_argument('--epoch', type=int, default=30,
                        help='Epoch of model to be loaded')

    # Parse the parameters
    sample_args = parser.parse_args()

    # dataset_name = "univ"
    # dataset_name = "eth"
    # dataset_name = "zara1"
    # dataset_name = "zara2"
    # dataset_name = "hotel"

    '''
       For dataset (test set)
    '''
    # Load the dataset (train set and validation set)
    dataset_path = "../datasets/" + dataset_name + "/"

    # Process TEST dataset and store it to the test_dataset object
    grad_eff = 0.4
    test_dataset = TrajectoryDataset(
        dataset_path + "test/",
        obs_len=sample_args.obs_length,
        pred_len=sample_args.pred_length,
        skip=1,
        norm_lap_matr=True,
        grad_eff=grad_eff,
    )

    '''
     For log (test)
     '''
    # Setup for test log
    log_test_directory_path = "./log/" + dataset_name + "/test/"
    # If the directory does not exist, we need to create it using the following code
    if not os.path.exists(log_test_directory_path):
        os.makedirs(log_test_directory_path)

    log_test_file = open(os.path.join(log_test_directory_path, 'test_log.txt'), 'w')
    test_metrics = {'ADE': [], 'FDE': []}


    '''
    For the parameter of parsing module
    '''
    #Load the parsing module parameters
    save_directory = './save/' + dataset_name + '/'
    with open(os.path.join(save_directory, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    '''
    For network model
    '''
    net = SocialLSTM(saved_args, True)

    '''
    For result
    '''
    trajectory_result = []
    result_directory_path = "./result/" + dataset_name + "/"
    # If the directory does not exist, we need to create it using the following code
    if not os.path.exists(result_directory_path):
        os.makedirs(result_directory_path)




    for epoch in range(sample_args.epoch):
        # Get the checkpoint path
        checkpoint_path = os.path.join(save_directory, 'social_lstm_model_'+str(epoch)+'.tar')
        # checkpoint_path = os.path.join(save_directory, 'srnn_model.tar')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            # model_iteration = checkpoint['iteration']
            model_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            print(f'successfullly load the saved model {model_epoch}')

        print('------------------------------------------------------------------')
        print('Testing started ...')
        print('------------------------------------------------------------------')
        test_metrics, trajectory_result_ = Test(epoch, test_dataset, net, saved_args, sample_args, dataset_name, test_metrics)
        # Store the predicted and true trajectory
        trajectory_result.append(trajectory_result_)
        # Print test ADE and FDE
        print('TEST:', '\t Model:', epoch, '\t Average ADE:', test_metrics['ADE'][epoch])
        print('TEST:', '\t Model:', epoch, '\t Average FDE:', test_metrics['FDE'][epoch])
        # Write test ADE and FDE
        log_test_file.write('Model: ' + str(epoch) + '\n' + 'ADE: ' + str(test_metrics['ADE'][epoch]) + '\n'
                              + 'FDE: ' + str(test_metrics['FDE'][epoch]) + '\n')

    # Print and write best test ADE
    print('------------------------------------------------------------------')
    best_test_ADE = min(test_metrics['ADE'])
    best_test_ADE_index = test_metrics['ADE'].index(best_test_ADE)
    print(f'The best ADE epoch is: {best_test_ADE_index};  The best ADE is: {best_test_ADE}')
    log_test_file.write('The best ADE epoch is: ' + str(best_test_ADE_index) + ' The best ADE is: ' + str(best_test_ADE) + '\n')

    # Print and write best test FDE
    best_test_FDE = min(test_metrics['FDE'])
    best_test_FDE_index = test_metrics['FDE'].index(best_test_FDE)
    print(f'The best FDE epoch is: {best_test_FDE_index};  The best FDE is: {best_test_FDE}')
    log_test_file.write('The best FDE epoch is: ' + str(best_test_FDE_index) + ' The best FDE is: ' + str(best_test_FDE) + '\n')

    log_test_file.close()

    # Store the test ADE and FDE
    with open(os.path.join(log_test_directory_path, 'test_ADE_FDE.pkl'), 'wb') as f:
        pickle.dump(test_metrics, f)

    # Store the true and predicted trajectory
    with open(os.path.join(result_directory_path, 'test_true_predicted_trajectory.pkl'), 'wb') as f:
        pickle.dump(trajectory_result, f)







def Test(epoch, test_dataset, net, saved_args, sample_args, dataset_name, test_metrics):
    net.eval()
    # ADE and FDE
    total_error = 0 #ADE
    final_error = 0 #FDE

    # Set the dimension of dataset, that is the region of scene in the dataset
    dataset_dimensions = {'eth': [720, 576], 'univ': [720, 576], 'zara1': [720, 576],
                          'zara2': [720, 576], 'hotel': [720, 576]}
    dataset_data = dataset_dimensions[dataset_name]

    trajectory_result = {'Predicted_trajectory': [], 'True_trajectory': []}

    for sequence in range(test_dataset.num_seq):
        # Get sequence data
        ori_seq = test_dataset.ori_sequence[sequence]
        traj_seq = test_dataset.sequence[sequence]
        traj_v = test_dataset.traj_velocity[sequence]
        traj_s = test_dataset.traj_start[sequence]
        traj_e = test_dataset.traj_end[sequence]
        traj_node_present = test_dataset.seq_ped_index[sequence]

        # Get the grid masks for the sequence
        grid_seq = getSequenceGridMask(ori_seq, dataset_data, saved_args.neighborhood_size, saved_args.grid_size)

        # Construct variables
        nodes = np.transpose(traj_seq, (2, 0, 1))
        # nodes = Variable(torch.from_numpy(nodes).float()).cuda()
        nodes = Variable(torch.from_numpy(nodes).float())
        numNodes = nodes.size()[1]

        # Extract the observed part of the trajectories
        obs_nodes = nodes[:sample_args.obs_length]
        obs_nodesPresent = traj_node_present[:sample_args.obs_length]
        obs_grid = grid_seq[:sample_args.obs_length]

        # The sample function
        predicted_nodes = sample(obs_nodes, obs_nodesPresent, obs_grid, sample_args, net, nodes, traj_node_present,
                                 grid_seq,
                                 saved_args, dataset_data)

        # Store true and predicted trajectory
        trajectory_result['True_trajectory'].append(nodes.numpy())
        trajectory_result['Predicted_trajectory'].append(predicted_nodes.numpy())

        # Record the mean and final displacement error
        total_error_ = get_mean_error(predicted_nodes[sample_args.obs_length:].data,
                                      nodes[sample_args.obs_length:].data,
                                      traj_node_present[sample_args.obs_length - 1],
                                      traj_node_present[sample_args.obs_length:])
        total_error += total_error_.item()
        final_error_ = get_final_error(predicted_nodes[sample_args.obs_length:].data,
                                       nodes[sample_args.obs_length:].data,
                                       traj_node_present[sample_args.obs_length - 1],
                                       traj_node_present[sample_args.obs_length:])
        final_error += final_error_.item()

        print('TEST:', '\t Model:', epoch, f'\t{sequence+1}/{test_dataset.num_seq}', '\t ADE:', total_error_, '\t FDE:', final_error_)

    test_metrics['ADE'].append(total_error/test_dataset.num_seq)
    test_metrics['FDE'].append(final_error/test_dataset.num_seq)

    return test_metrics, trajectory_result



if __name__ == '__main__':
    torch.manual_seed(20)
    np.random.seed(20)
    random.seed(20)
    dataset = ['univ']
    # dataset = ['univ','eth','zara1','zara2','hotel']
    for dataset_name in dataset:
        print(f'We are testing {dataset_name}\n')
        main(dataset_name)
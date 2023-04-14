import torch
from torch.autograd import Variable
from grid import getSequenceGridMask, getGridMaskInference
import numpy as np


def sample(nodes, nodesPresent, grid, args, net, true_nodes, true_nodesPresent, true_grid, saved_args, dimensions):
    '''
    The sample function
    params:
    nodes: Input positions
    nodesPresent: Peds present in each frame
    args: arguments
    net: The model
    true_nodes: True positions
    true_nodesPresent: The true peds present in each frame
    true_grid: The true grid masks
    saved_args: Training arguments
    dimensions: The dimensions of the dataset
    '''
    # Number of peds in the sequence
    numNodes = nodes.size()[1]

    # Construct variables for hidden and cell states
    #hidden_states = Variable(torch.zeros(numNodes, net.args.rnn_size), volatile=True).cuda()
    #cell_states = Variable(torch.zeros(numNodes, net.args.rnn_size), volatile=True).cuda()
    hidden_states = Variable(torch.zeros(numNodes, net.args.rnn_size), volatile=True)
    cell_states = Variable(torch.zeros(numNodes, net.args.rnn_size), volatile=True)

    # For the observed part of the trajectory
    for tstep in range(args.obs_length-1):
        # Do a forward prop
        out_obs, hidden_states, cell_states = net(nodes[tstep].view(1, numNodes, 2), [grid[tstep]], [nodesPresent[tstep]], hidden_states, cell_states)
        # loss_obs = Gaussian2DLikelihood(out_obs, nodes[tstep+1].view(1, numNodes, 2), [nodesPresent[tstep+1]], args.pred_length)

    # Initialize the return data structure
    #ret_nodes = Variable(torch.zeros(args.obs_length+args.pred_length, numNodes, 2), volatile=True).cuda()
    ret_nodes = Variable(torch.zeros(args.obs_length+args.pred_length, numNodes, 2), volatile=True)
    #ret_nodes[:args.obs_length, :, :] = nodes.clone()
    ret_nodes[:args.obs_length, :, :] = nodes

    # Last seen grid
    prev_grid = grid[-1].clone()

    # For the predicted part of the trajectory
    for tstep in range(args.obs_length-1, args.pred_length + args.obs_length - 1):
        # Do a forward prop
        outputs, hidden_states, cell_states = net(ret_nodes[tstep].view(1, numNodes, 2), [prev_grid], [nodesPresent[args.obs_length-1]], hidden_states, cell_states)
        # loss_pred = Gaussian2DLikelihoodInference(outputs, true_nodes[tstep+1].view(1, numNodes, 2), nodesPresent[args.obs_length-1], [true_nodesPresent[tstep+1]])

        # Extract the mean, std and corr of the bivariate Gaussian
        mux, muy, sx, sy, corr = getCoef(outputs)
        # Sample from the bivariate Gaussian
        next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, nodesPresent[args.obs_length-1])

        # Store the predicted position
        ret_nodes[tstep + 1, :, 0] = next_x
        ret_nodes[tstep + 1, :, 1] = next_y

        # List of nodes at the last time-step (assuming they exist until the end)
        #list_of_nodes = Variable(torch.LongTensor(nodesPresent[args.obs_length-1]), volatile=True).cuda()
        list_of_nodes = Variable(torch.LongTensor(nodesPresent[args.obs_length-1]), volatile=True)

        # Get their predicted positions
        current_nodes = torch.index_select(ret_nodes[tstep+1], 0, list_of_nodes)

        # Compute the new grid masks with the predicted positions
        prev_grid = getGridMaskInference(current_nodes.data.cpu().numpy(), dimensions, saved_args.neighborhood_size, saved_args.grid_size)
        #prev_grid = Variable(torch.from_numpy(prev_grid).float(), volatile=True).cuda()
        prev_grid = Variable(torch.from_numpy(prev_grid).float(), volatile=True)

    return ret_nodes


def getCoef(outputs):
    '''
    Extracts the mean, standard deviation and correlation
    params:
    outputs : Output of the SRNN model
    '''
    mux, muy, sx, sy, corr = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4]

    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)
    return mux, muy, sx, sy, corr


def sample_gaussian_2d(mux, muy, sx, sy, corr, nodesPresent):
    '''
    Parameters
    ==========

    mux, muy, sx, sy, corr : a tensor of shape 1 x numNodes
    Contains x-means, y-means, x-stds, y-stds and correlation

    nodesPresent : a list of nodeIDs present in the frame

    Returns
    =======

    next_x, next_y : a tensor of shape numNodes
    Contains sampled values from the 2D gaussian
    '''
    o_mux, o_muy, o_sx, o_sy, o_corr = mux[0, :], muy[0, :], sx[0, :], sy[0, :], corr[0, :]

    numNodes = mux.size()[1]

    next_x = torch.zeros(numNodes)
    next_y = torch.zeros(numNodes)
    for node in range(numNodes):
        if node not in nodesPresent:
            continue
        mean = [o_mux[node], o_muy[node]]
        cov = [[o_sx[node]*o_sx[node], o_corr[node]*o_sx[node]*o_sy[node]], [o_corr[node]*o_sx[node]*o_sy[node], o_sy[node]*o_sy[node]]]

        next_values = np.random.multivariate_normal(mean, cov, 1)
        next_x[node] = next_values[0][0]
        next_y[node] = next_values[0][1]

    # return torch.from_numpy(next_x).cuda(), torch.from_numpy(next_y).cuda()
    return next_x, next_y

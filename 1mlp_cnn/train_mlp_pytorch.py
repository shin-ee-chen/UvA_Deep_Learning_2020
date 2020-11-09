"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim

import plot_utils



# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100, 100, 100, 100, 100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1100
# MAX_STEPS_DEFAULT = 1400
# BATCH_SIZE_DEFAULT = 200
BATCH_SIZE_DEFAULT = 512
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    _, predicted = torch.max(predictions, 1)
    _, labels = torch.max(targets, 1)
    
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return accuracy


def train():
    """
    Performs training and evaluation of MLP model.
  
    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    
    # neg_slope = FLAGS.neg_slope
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # load data
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    train_x, train_y = cifar10['train'].next_batch(batch_size= FLAGS.batch_size)
    train_x = torch.from_numpy(train_x.reshape([train_x.shape[0], -1]))
    train_y = torch.from_numpy(train_y)
    print("train_x shape is {}, train_y.shape is {}".format(train_x.shape, train_y.shape))

    # Define network
    mlp = MLP(train_x.shape[1], dnn_hidden_units, train_y.shape[1])
    train_loss = []
    test_loss = []
    accs = []

    if FLAGS.optimizer.lower() == 'adam':
        optimizer = optim.Adam(mlp.parameters(), lr = FLAGS.learning_rate, weight_decay = FLAGS.weight_decay)
    else:
        optimizer = optim.SGD(mlp.parameters(), lr = FLAGS.learning_rate, weight_decay = FLAGS.weight_decay)
    
    cross_entro = nn.CrossEntropyLoss()

    for step in range(FLAGS.max_steps):
        # forward prop
        optimizer.zero_grad()
        out = mlp.forward(train_x)
        loss = cross_entro(out, torch.max(train_y, 1)[1])
        loss.backward()
        optimizer.step()

        if step % FLAGS.eval_freq == (FLAGS.eval_freq - 1):
            train_loss.append(loss.item())
            
            test_x, test_y = cifar10["test"].images, cifar10["test"].labels
            test_x =  torch.from_numpy(test_x.reshape([test_x.shape[0], -1]))
            test_y = torch.from_numpy(test_y)

            test_out = mlp.forward(test_x)
            test_loss.append(cross_entro(test_out, torch.max(test_y, 1)[1]).item())
            acc = accuracy(test_out, test_y)
            accs.append(acc)
            print("Step {}, accuracy is {}".format(step + 1, acc))
            # print("Train Loss {}, test loss {}".format(loss.item(), train_loss[-1]) )
        train_x, train_y = cifar10['train'].next_batch(batch_size= FLAGS.batch_size)
        train_x = torch.from_numpy(train_x.reshape([train_x.shape[0], -1]))
        train_y = torch.from_numpy(train_y)

    loss_img_path = os.path.join("results", "pytorch_loss_curve")
    plot_utils.plot_loss_curve(train_loss, test_loss, loss_img_path, FLAGS.eval_freq)

    acc_img_path = os.path.join("results", "pytorch_acc_curve")
    plot_utils.plot_acc_curve(accs, acc_img_path, FLAGS.eval_freq)
    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    parser.add_argument('--weight_decay', type=str, default=0.001,
                        help='weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default="Adam",
                        help='Type of optimizer')

    FLAGS, unparsed = parser.parse_known_args()
    
    main()

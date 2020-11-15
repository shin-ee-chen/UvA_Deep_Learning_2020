"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils

import torch
import torch.nn as nn

import torch.optim as optim

import plot_utils

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

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
    Performs training and evaluation of ConvNet model.
  
    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # load data
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    train_x, train_y = cifar10['train'].next_batch(batch_size= FLAGS.batch_size)
    train_x = torch.from_numpy(train_x).to(device)
    train_y = torch.from_numpy(train_y).to(device)
    print("train_x shape is {}, train_y.shape is {}".format(train_x.shape, train_y.shape))

    # Define network
    vgg = ConvNet(train_x.shape[1], train_y.shape[1]).to(device)
    train_loss = []
    test_loss = []
    accs = []


    optimizer = optim.Adam(vgg.parameters(), lr = FLAGS.learning_rate)
    cross_entro = nn.CrossEntropyLoss()

    for step in range(FLAGS.max_steps):
        # forward prop
        optimizer.zero_grad()
        out = vgg.forward(train_x)
        loss = cross_entro(out, torch.max(train_y, 1)[1])
        loss.backward()
        optimizer.step()

        if step % FLAGS.eval_freq == (FLAGS.eval_freq - 1):
            train_loss.append(loss.item())
            
            # test_x, test_y = cifar10["test"].images, cifar10["test"].labels
            # test_x =  torch.from_numpy(test_x).to(device)
            # test_y = torch.from_numpy(test_y).to(device)

            acc = 0
            t_loss = 0
            n_batch = 0
            while n_batch * FLAGS.batch_size < cifar10["test"].num_examples:
                test_x, test_y = cifar10["test"].next_batch(batch_size= FLAGS.batch_size)
                test_x = torch.from_numpy(test_x).to(device)
                test_y = torch.from_numpy(test_y).to(device)

                test_out = vgg.forward(test_x)
                acc += accuracy(test_out, test_y)
                t_loss += cross_entro(test_out, torch.max(test_y, 1)[1]).item()
                n_batch += 1
            
            print("Step {}, accuracy is {}".format(step + 1, acc / n_batch))
            test_loss.append(t_loss / n_batch)
            accs.append(acc / n_batch)
            # test_out = vgg.forward(test_x)
            # test_loss.append(cross_entro(test_out, torch.max(test_y, 1)[1]).item())
            # acc = accuracy(test_out, test_y)
            # accs.append(acc)
            # print("Step {}, accuracy is {}".format(step + 1, acc))
            # print("Train Loss {}, test loss {}".format(loss.item(), train_loss[-1]) )
        train_x, train_y = cifar10['train'].next_batch(batch_size= FLAGS.batch_size)
        train_x = torch.from_numpy(train_x).to(device)
        train_y = torch.from_numpy(train_y).to(device)

    loss_img_path = os.path.join("results", "pytorch_cnn_loss_curve")
    plot_utils.plot_loss_curve(train_loss, test_loss, loss_img_path, FLAGS.eval_freq)

    acc_img_path = os.path.join("results", "pytorch_cnn_acc_curve")
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
    FLAGS, unparsed = parser.parse_known_args()
    
    main()

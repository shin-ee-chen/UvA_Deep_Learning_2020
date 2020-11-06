"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule, SoftMaxModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

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
    pred_indices = np.expand_dims(np.argmax(predictions, axis=1), axis= -1)
    pred_labels = np.take_along_axis(targets, pred_indices, axis=1)
    accuracy = np.sum(pred_labels) / targets.shape[0]
    
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

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # Load data
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    train_x, train_y = cifar10['train'].next_batch(batch_size= FLAGS.batch_size)
    train_x = train_x.reshape([train_x.shape[0], -1])
    print("train_x shape is {}, train_y.shape is {}".format(train_x.shape, train_y.shape))

    # Define network
    mlp = MLP(train_x.shape[1], dnn_hidden_units, train_y.shape[1])
    # Training
    for step in range(FLAGS.max_steps):
        # forward prop
        out = mlp.forward(train_x)
        softmax = SoftMaxModule()
        out = softmax.forward(out)
        cross_entro = CrossEntropyModule()
        loss = cross_entro.forward(out, train_y)

        #backward prob
        dout = cross_entro.backward(out, train_y)
        dout = softmax.backward(dout)
        dx = mlp.backward(dout)

        # update params
        n_linear_layers = len(dnn_hidden_units) + 1
        # print("n_l ",n_linear_layers)
        # print(len(mlp.layers))
        for n in range(n_linear_layers):
            linear = mlp.layers[2 * n]
            linear.params["weight"] -= FLAGS.learning_rate * linear.grads["weight"]
            linear.params["bias"] -=  FLAGS.learning_rate * linear.grads["bias"]
        
        # Evaluation
        if step % FLAGS.eval_freq == 0:
            print(loss)
    # Prediction
    # print("Prediction is ")
    # mlp.forward(test)
    # softmax = SoftMaxModule()
    # out = softmax.forward(out)
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
    FLAGS, unparsed = parser.parse_known_args()

    main()
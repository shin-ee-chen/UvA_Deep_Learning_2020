"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *

class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.layers = []
        input_size = n_inputs
        for hidden_size in n_hidden:
          self.layers.append(LinearModule(input_size, hidden_size))
          self.layers.append(ELUModule())
          input_size = hidden_size
        
        self.layers.append(LinearModule(input_size, n_classes))
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        out = x
        for layer in self.layers:
          out = layer.forward(out)
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        for layer in self.layers[::-1]:
          dout = layer.backward(dout)

        ########################
        # END OF YOUR CODE    #
        #######################

        return dout

if __name__ == '__main__':
  S, M, N =3, 4, 5
  x = np.random.randn(S, M)
  n_hidden = [10,12]
  mlp = MLP(M, n_hidden, 2)
  out = mlp.forward(x)
  print(out.shape)
  print(mlp.backward(out).shape)
  
  # a = np.random.randn(S, M)
  # print(a)
  # b = np.random.randn(S, M)
  # a_max = np.expand_dims(np.argmax(a, axis=1), axis= -1)
  # print(np.take_along_axis(a, a_max, axis=1))
"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import cifar10_utils
class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """
    
    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          
        
        TODO:
        Implement initialization of the network.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super(ConvNet, self).__init__()

        self.conv_0 = nn.Conv2d(n_channels, 64, 3, 1, 1)
        self.conv_1 = nn.Conv2d(64, 128, 1, 1, 0)
        self.conv_2 = nn.Conv2d(128, 256, 1, 1, 0)
        self.conv_3 = nn.Conv2d(256, 512, 1, 1, 0)

        self.PreAct_1 = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, 64, 3, 1, 1)
        )

        self.PreAct_2 = nn.Sequential(
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.Conv2d(128, 128, 3, 1, 1)
        )

        # self.PreAct_2_b = nn.Sequential(
        #   nn.BatchNorm2d(128),
        #   nn.ReLU(),
        #   nn.Conv2d(128, 128, 3, 1, 1)
        # )

        self.PreAct_3 = nn.Sequential(
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.Conv2d(256, 256, 3, 1, 1)
        )

        # self.PreAct_3_b = nn.Sequential(
        #   nn.BatchNorm2d(256),
        #   nn.ReLU(),
        #   nn.Conv2d(256, 256, 3, 1, 1)
        # )

        self.PreAct_4 = nn.Sequential(
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.Conv2d(512, 512, 3, 1, 1)
        )

        # self.PreAct_4_b = nn.Sequential(
        #   nn.BatchNorm2d(512),
        #   nn.ReLU(),
        #   nn.Conv2d(512, 512, 3, 1, 1)
        # )

        self.PreAct_5 = nn.Sequential(
          nn.BatchNorm2d(512),
          nn.ReLU(),
          nn.Conv2d(512, 512, 3, 1, 1)
        )

        # self.PreAct_5_b = nn.Sequential(
        #   nn.BatchNorm2d(512),
        #   nn.ReLU(),
        #   nn.Conv2d(512, 512, 3, 1, 1)
        # )

        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.batchNorm = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(512, n_classes)
          
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
        out_0 = self.conv_0(x)

        z_1 = out_0 + self.PreAct_1(out_0)
        out_1 = self.maxpool(self.conv_1(z_1))
        
        z_2_a = out_1 + self.PreAct_2(out_1)
        z_2_b = z_2_a + self.PreAct_2(z_2_a)
        out_2 = self.maxpool(self.conv_2(z_2_b))

        z_3_a = out_2 + self.PreAct_3(out_2)
        z_3_b = z_3_a + self.PreAct_3(z_3_a)
        out_3 = self.maxpool(self.conv_3(z_3_b))

        z_4_a = out_3 + self.PreAct_4(out_3)
        z_4_b = z_4_a + self.PreAct_4(z_4_a)
        out_4 = self.maxpool(z_4_b)

        z_5_a = out_4 + self.PreAct_5(out_4)
        z_5_b = z_5_a + self.PreAct_5(z_5_a)
        out_5 = self.maxpool(z_5_b)
        # out_5 = out_5.view(-1, 512)
       
        out_5 = self.relu(self.batchNorm(out_5))
        out_5 = out_5.view(-1, 512)
        out = self.linear(out_5)

        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out

if __name__ == '__main__':
      # np.random.seed(42)
      torch.manual_seed(42)
      vgg13 = ConvNet(3, 10)
      cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
      data, labels = cifar10['train'].next_batch(batch_size= 64)
      out = vgg13.forward(torch.from_numpy(data[0:2]))
      print(out)

################################################################################
# MIT License
#
# Copyright (c) 2020 Phillip Lippe
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2020
# Date Created: 2020-11-22
################################################################################

import torch
from torchvision.utils import make_grid
import numpy as np
from scipy.stats import norm


def sample_reparameterize(mean, std, device = "cpu"):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std. 
            The tensor should have the same shape as the mean and std input tensors.
    """
    epsilon = torch.randn(mean.shape).to(device)
    z = mean + std * epsilon
    # raise NotImplementedError
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See Section 1.3 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """
    
    log_var = 2 * log_std
    KLD =  torch.sum(0.5 * (torch.exp(log_var) + mean * mean - 1 - log_var),dim = -1)
    # raise NotImplementedError
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    bpd = elbo * np.log2(np.e) / (img_shape[1]* img_shape[2]* img_shape[3])
    # raise NotImplementedError
    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/(grid_size+1)
    Outputs:
        img_grid - Grid of images representing the manifold.
    """

    ## Hints:
    # - You can use scipy's function "norm.ppf" to obtain z values at percentiles.
    # - Use the range [0.5/(grid_size+1), 1.5/(grid_size+1), ..., (grid_size+0.5)/(grid_size+1)] for the percentiles.
    # - torch.meshgrid might be helpful for creating the grid of values
    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    # - Remember to apply a sigmoid after the decoder

    # img_grid = None
    # raise NotImplementedError
    # z = torch.zeros(grid_size)
    # for i in range(grid_size):
    #     z[i] = norm.ppf((i+0.5) / (grid_size+1))
    # mean = torch.sigmoid(decoder(z))
    # https://www.quora.com/How-can-I-draw-a-manifold-from-a-variational-autoencoder-in-Keras
    
    z = torch.zeros([grid_size, 2]) #not sure the dimension is right, z should be[Batch, 2]
    for i in range(grid_size):
        for j in range(grid_size):
           z[i][0] = norm.ppf((i+0.5) / (grid_size+1))
           z[i][1] = norm.ppf((j+0.5) / (grid_size+1))
    print(z)
    img_grid = None
    return img_grid

if __name__ == '__main__':
    # x = torch.randn(3, 2, 1, 4)
    # print(x)
    # x = x.reshape(3,-1)
    # print(x.reshape(3,-1))
    # print(torch.sum(x, dim = 1).shape)
    x = [1,1,2,2,3,3]
    visualize_manifold(10, grid_size=3)
    print(z)
    # print(torch.tensor(x).reshape(3,2))
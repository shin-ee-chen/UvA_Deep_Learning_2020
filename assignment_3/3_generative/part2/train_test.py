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

import argparse
import os
import datetime
import statistics
import random

from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from mnist import mnist
from models import GeneratorMLP, DiscriminatorMLP
from utils import TensorBoardLogger

import math
from torch.autograd import Variable
cuda = True if torch.cuda.is_available() else False
# check drop last batch, added params to generative step and discriminative step
class GAN(nn.Module):

    def __init__(self, hidden_dims_gen, hidden_dims_disc, dp_rate_gen,
                 dp_rate_disc, z_dim, *args, **kwargs):
        """
        PyTorch Lightning module that summarizes all components to train a GAN.
        Inputs:
            hidden_dims_gen  - List of hidden dimensionalities to use in the
                              layers of the generator
            hidden_dims_disc - List of hidden dimensionalities to use in the
                               layers of the discriminator
            dp_rate_gen      - Dropout probability to use in the generator
            dp_rate_disc     - Dropout probability to use in the discriminator
            z_dim            - Dimensionality of latent space
        """
        super().__init__()
        self.z_dim = z_dim

        self.generator = GeneratorMLP(z_dim=z_dim,
                                      hidden_dims=hidden_dims_gen,
                                      dp_rate=dp_rate_gen)
        self.discriminator = DiscriminatorMLP(hidden_dims=hidden_dims_disc,
                                              dp_rate=dp_rate_disc)

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Function for sampling a new batch of random images from the generator.
        Inputs:
            batch_size - Number of images to generate
        Outputs:
            x - Generated images of shape [B,C,H,W]
        """
        z = torch.randn((batch_size, self.z_dim)).to(self.device)
        x = self.generator(z)

        return x

    @torch.no_grad()
    def interpolate(self, batch_size, interpolation_steps):
        """
        Function for interpolating between a batch of pairs of randomly sampled
        images. The interpolation is performed on the latent input space of the
        generator.
        Inputs:
            batch_size          - Number of image pairs to generate
            interpolation_steps - Number of intermediate interpolation points
                                  that should be generated.
        Outputs:
            x - Generated images of shape [B,interpolation_steps+2,C,H,W]
        """
        # fake_imgs = []
        # for x in range(batch_size):
        z_1 = np.random.normal(-0.7, 0.7, self.z_dim)
        z_2 = np.random.normal(0.7, 0.7, self.z_dim)
        interpolate_space = np.linspace(z_1,z_2,interpolation_steps+2)
        digits_list = []
        for digit in interpolate_space:
            z = torch.from_numpy(digit).float().to(self.device) * torch.ones((args.z_dim)).to(self.device)
            digits_list.append(z)
        # stack tensors
        z = torch.stack(digits_list, dim=0).to(self.device)
        # print("uhmmm", z.shape)
        x = self.generator(z)
        # print("uhmmm", z.shape, x.shape)
        # fake_imgs.append(fake_img)
        # x = torch.stack(fake_imgs, dim=0).to(device)
        return x

    def generator_step(self, x_real, fake_imgs, criterion):
        """
        Training step for the generator. Note that you do *not* need to take
        any special care of the discriminator in terms of stopping the
        gradients to its parameters, as this is handled by having two different
        optimizers. Before the discriminator's gradients in its own step are
        calculated, the previous ones have to be set to zero.
        Inputs:
            x_real - Batch of images from the dataset
        Outputs:
            loss - The loss for the generator to optimize
            logging_dict - Dictionary of string to Tensor that should be added
                           to our TensorBoard logger
        """
        
        # real_labels = torch.ones(args.batch_size, 1).to(device) * torch.FloatTensor(1).uniform_(0.0, 0.3).to(device)
        # fake_labels = torch.ones(args.batch_size, 1).to(device) * torch.FloatTensor(1).uniform_(0.7, 1.2).to(device)
        # fake_imgs = self.sample(args.batch_size)
        
        

        d_fake = self.discriminator(fake_imgs)
        loss = criterion(d_fake, x_real)

        # # Adversarial ground truths
        # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        # valid = Variable(Tensor(x_real.size(0), 1).fill_(1.0), requires_grad=False)
        # fake = Variable(Tensor(x_real.size(0), 1).fill_(0.0), requires_grad=False)
        # # # Configure input
        # # real_imgs = Variable(x_real.type(Tensor))

        # # Sample noise as generator input
        # z = Variable(Tensor(np.random.normal(0, 1, (x_real.shape[0], self.z_dim))))

        # # Generate a batch of images
        # gen_imgs = self.generator(z)

        # # Loss measures generator's ability to fool the discriminator
        # loss = criterion(self.discriminator(gen_imgs), valid)

        logging_dict = {"loss": loss}
        

        return loss, logging_dict

    def discriminator_step(self, x_real, real_labels, fake_labels, fake_imgs, criterion):
        """
        Training step for the discriminator. Note that you do not have to use
        the same generated images as in the generator_step. It is simpler to
        sample a new batch of "fake" images, and use those for training the
        discriminator. It has also been shown to stabilize the training.
        Remember to log the training loss, and other potentially interesting 
        metrics.
        Inputs:
            x_real - Batch of images from the dataset
        Outputs:
            loss - The loss for the discriminator to optimize
            logging_dict - Dictionary of string to Tensor that should be added
                           to our TensorBoard logger
        """

        # Remark: there are more metrics that you can add. 
        # For instance, how about the accuracy of the discriminator?
        # real_labels = torch.ones(args.batch_size, 1).to(device) * torch.FloatTensor(1).uniform_(0.0, 0.3).to(device)
        # fake_labels = torch.ones(args.batch_size, 1).to(device) * torch.FloatTensor(1).uniform_(0.7, 1.2).to(device)

        # fake_imgs = self.sample(args.batch_size)

        
        
        d_real = self.discriminator(x_real)
        

        d_real_loss = criterion(d_real, real_labels)
        d_fake = self.discriminator(fake_imgs)
        d_fake_loss = criterion(d_fake, fake_labels)
        loss = d_real_loss + d_fake_loss
        logging_dict = {"loss": loss}
        return loss, logging_dict

        # # Sample noise as generator input
        # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        # z = Variable(Tensor(np.random.normal(0, 1, (args.z_dim,784))))
        # # print(z.shape)
        # # Generate a batch of images
        # gen_imgs = self.generator(z)
        # criterion = nn.BCEWithLogitsLoss()

        # # Adversarial ground truths
        # valid = Variable(Tensor(x_real.size(0), 1).fill_(1.0), requires_grad=False)
        # fake = Variable(Tensor(x_real.size(0), 1).fill_(0.0), requires_grad=False)

        # # Measure discriminator's ability to classify real from generated samples
        # real_loss = criterion(self.discriminator(x_real), valid)
        # fake_loss = criterion(self.discriminator(gen_imgs.detach()), fake)
        # loss = (real_loss + fake_loss) / 2
        

        

    @property
    def device(self):
        """
        Property function to get the device on which the model is.
        """
        return self.generator.device
    



def generate_and_save(model, epoch, summary_writer, batch_size=64):
    """
    Function that generates and save samples from the GAN.
    The generated samples images should be added to TensorBoard and,
    eventually saved inside the logging directory.
    Inputs:
        model - The GAN model that is currently being trained.
        epoch - The epoch number to use for TensorBoard logging and saving of the files.
        summary_writer - A TensorBoard summary writer to log the image samples.
        batch_size - Number of images to generate/sample
    """
    # Hints:
    # - You can access the logging directory via summary_writer.log_dir
    # - Use torchvision function "make_grid" to create a grid of multiple images
    # - Use torchvision function "save_image" to save an image grid to disk
    
    fake_imgs = model.sample(batch_size)
    # Save generated images
    # save_image(fake_imgs.view(-1, 1, 28, 28),
    #             f"{summary_writer.log_dir}/gen_images/"+'gan_{}.png'.format(epoch),
    #             nrow=int(math.sqrt(batch_size)), normalize=True)
    log_dir = summary_writer.log_dir
    samples = make_grid(fake_imgs)
    summary_writer.add_image(f"samples at epoch={epoch}", samples)
    save_image(samples, os.path.join(log_dir, f"samples_{epoch}.png"))



def interpolate_and_save(model, epoch, summary_writer, batch_size=64,
                         interpolation_steps=5):
    """
    Function that generates and save the interpolations from the GAN.
    The generated samples and mean images should be added to TensorBoard and,
    if self.save_to_disk is True, saved inside the logging directory.
    Inputs:
        model - The VAE model that is currently being trained.
        epoch - The epoch number to use for TensorBoard logging and saving of the files.
        summary_writer - A TensorBoard summary writer to log the image samples.
        batch_size - Number of images to generate/sample
        interpolation_steps - Number of interpolation steps to perform
                              between the two random images.
    """
    # Hints:
    # - You can access the logging directory path via summary_writer.log_dir
    # - Use the torchvision function "make_grid" to create a grid of multiple images
    # - Use the torchvision function "save_image" to save an image grid to disk
    
    # You also have to implement this function in a later question of the assignemnt. 
    # By default it is skipped to allow you to test your other code so far. 
    interpolation = model.interpolate(batch_size, interpolation_steps)
    
    save_image(interpolation.view(-1,1,28,28),'./interpolation/interpolate_digits_{}.png'.format(epoch), nrow=interpolation_steps+2, normalize=True)
    # print("WARNING: Interpolation function has not been implemented yet.")
    print('\nProduced interpolation between two digits. Saved as: interpolate_digits.png\n')


def train_gan(model, train_loader,
              logger_gen, logger_disc,
              optimizer_gen, optimizer_disc):
    """
    Function for training a GAN model on a dataset for a single epoch.
    Inputs:
        model - GAN model to train
        train_loader - Data Loader for the dataset you want to train on
        logger_gen - Logger object for the generator (see utils.py)
        logger_disc - Logger object for the discriminator (see utils.py) 
        optimizer - The optimizer used to update the parameters
    """
    model.train()
    discriminator = model.discriminator.to(model.device)
    generator = model.generator.to(model.device)
    criterion = nn.BCEWithLogitsLoss()
    for imgs, _ in train_loader:
        imgs = imgs.to(model.device)

        # Remark: add the logging dictionaries via 
        # "logger_gen.add_values(logging_dict)" for the generator, and
        # "logger_disc.add_values(logging_dict)" for discriminator
        # (both loggers should get different dictionaries, the outputs 
        #  of the respective step functions)

        real_labels = torch.zeros(args.batch_size, 1).to(model.device)
        fake_labels = torch.ones(args.batch_size, 1).to(model.device)
        
        # fake_imgs = model.sample(args.batch_size)
        z = torch.randn((args.batch_size, args.z_dim)).to(model.device)
        fake_imgs = generator(z)
        
        # Discriminator update
        # optimizer_disc.zero_grad()
        # dloss ,_ = model.discriminator_step(imgs, real_labels, fake_labels, fake_imgs, model.criterion)
        # dloss.backward()
        # optimizer_disc.step()
        # print("Dloss: ", dloss.item())

        optimizer_disc.zero_grad()
        d_real = discriminator(imgs)
        d_real_loss = criterion(d_real, real_labels)
        d_fake = discriminator(fake_imgs)
        d_fake_loss = criterion(d_fake, fake_labels)
        dloss = d_real_loss + d_fake_loss
        dloss.backward(retain_graph=True)
        optimizer_disc.step()
        # print("Dloss: ", dloss.item())

        # Generator update
        # optimizer_gen.zero_grad()
        # gloss, _ = model.generator_step(real_labels, fake_imgs, model.criterion)
        # gloss.backward()
        # optimizer_gen.step()
        # print("Gloss: ", gloss.item())

        optimizer_gen.zero_grad()
        d_fake = discriminator(fake_imgs)
        gloss = criterion(d_fake, real_labels)
        gloss.backward()
        optimizer_gen.step()
        # print("Gloss: ", gloss.item())

        

        
        


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """
    Main Function for the full training loop of a GAN model.
    Makes use of a separate train function for a single epoch.
    Remember to implement the optimizers, everything else is provided.
    Inputs:
        args - Namespace object from the argument parser
    """
    if args.seed is not None:
        seed_everything(args.seed)

    # Preparation of logging directories
    experiment_dir = os.path.join(
        args.log_dir, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    checkpoint_dir = os.path.join(
        experiment_dir, 'checkpoints')
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader = mnist(batch_size=args.batch_size,
                         num_workers=args.num_workers)

    # Create model
    model = GAN(hidden_dims_gen=args.hidden_dims_gen,
                hidden_dims_disc=args.hidden_dims_disc,
                dp_rate_gen=args.dp_rate_gen,
                dp_rate_disc=args.dp_rate_disc,
                z_dim=args.z_dim)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Create two separate optimizers for generator and discriminator
    # You can use the Adam optimizer for both models.
    # It is recommended to reduce the momentum (beta1) to e.g. 0.5
    optimizer_gen = torch.optim.Adam(model.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_disc = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    

    # TensorBoard logger
    # See utils.py for details on "TensorBoardLogger" class
    summary_writer = SummaryWriter(experiment_dir)
    logger_gen = TensorBoardLogger(summary_writer, name="generator")
    logger_disc = TensorBoardLogger(summary_writer, name="discriminator")

    # Initial generation before training
    generate_and_save(model, 0, summary_writer, 64)

    # Training loop
    print(f"Using device {device}")
    epoch_iterator = (trange(args.epochs, desc="GAN")
                      if args.progress_bar else range(args.epochs))
    for epoch in epoch_iterator:
        print("hello epoch: ", epoch)
        # Training epoch
        train_iterator = (tqdm(train_loader, desc="Training", leave=False)
                          if args.progress_bar else train_loader)
        train_gan(model, train_iterator,
                  logger_gen, logger_disc,
                  optimizer_gen, optimizer_disc)

        # Logging images
        if (epoch + 1) % 10 == 0:
            print("interpolate epoch: ", epoch+1)
            generate_and_save(model, epoch+1, summary_writer)
            interpolate_and_save(model, epoch+1, summary_writer)

        # Saving last model (only every 10 epochs to reduce IO traffic)
        # As we do not have a validation step, we cannot determine the "best"
        # checkpoint during training except looking at the samples.
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, "model_checkpoint.pt"))


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument('--z_dim', default=32, type=int,
                        help='Dimensionality of latent space')
    parser.add_argument('--hidden_dims_gen', default=[128, 256, 512],
                        type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the ' +
                             'generator. To specify multiple, use " " to ' +
                             'separate them. Example: \"128 256 512\"')
    parser.add_argument('--hidden_dims_disc', default=[512, 256],
                        type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the ' +
                             'discriminator. To specify multiple, use " " to ' +
                             'separate them. Example: \"512 256\"')
    parser.add_argument('--dp_rate_gen', default=0.1, type=float,
                        help='Dropout rate in the discriminator')
    parser.add_argument('--dp_rate_disc', default=0.3, type=float,
                        help='Dropout rate in the discriminator')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=2e-4, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size to use for training')

    # Other hyperparameters
    parser.add_argument('--epochs', default=250, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders.' +
                             'To have a truly deterministic run, this has to be 0.')
    parser.add_argument('--log_dir', default='GAN_logs/', type=str,
                        help='Directory where the PyTorch Lightning logs ' +
                             'should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help='Use a progress bar indicator for interactive experimentation. ' +
                             'Not to be used in conjuction with SLURM jobs.')

    args = parser.parse_args()

    main(args)
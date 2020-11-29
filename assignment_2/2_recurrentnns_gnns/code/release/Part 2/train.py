# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel
from torch.utils.tensorboard import SummaryWriter

import os
import random
###############################################################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, 
                                dataset.vocab_size, config.lstm_num_hidden,
                                config.lstm_num_layers, config.device, 
                                config.dropout_keep_prob)  # FIXME

    model.to(device)
    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # FIXME
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)  # FIXME
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,config.milestones, 
                                                     config.learning_rate_decay)

    # Use pre-trained checkpoints
    output_file_name = "{}_{}_{}".format(os.path.basename(config.txt_file), 
                                         config.gen_sentence_len, config.temperature)
    checkpoint_path = os.path.join(config.checkpoint_path, 
                                           "{}.pth".format(output_file_name))
    if os.path.exists(checkpoint_path):
        print("Continue training model {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)

    if not os.path.exists(config.summary_path):
        os.makedirs(config.summary_path)
    
    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)

    writer = SummaryWriter(config.summary_path, filename_suffix=output_file_name)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Only for time measurement of step through network
        t1 = time.time()

    
        #######################################################
        # Add more code here ...
        #######################################################
        # batch_inputs/batch_targets shape: (seq_len, batch_size)
        batch_inputs = torch.stack(batch_inputs).to(device) #(30,64)
        batch_targets = torch.stack(batch_targets).to(device) #(30,64)
        
        model.train()
        optimizer.zero_grad()

        model.prev_state = None
        probs, _ = model.forward(batch_inputs) # shape:(seq_len,batch_size,vocab_size)(30,64,87)
        
        predictions = torch.argmax(probs, dim=2) #shape:(30,64)
        correct = (predictions == batch_targets).sum().item() #
 
        loss = criterion(probs.permute(0,2,1) , batch_targets)  # fixme
        accuracy = correct / (probs.size(0) * probs.size(1))  # fixme

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optimizer.step()
        scheduler.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if (step + 1) % config.print_every == 0:
            torch.save(model.state_dict(), checkpoint_path)
            writer.add_scalar("{} Loss: ".format(output_file_name), loss, step)
            writer.add_scalar("{} Accuracy:".format(output_file_name), accuracy, step)

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                    Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
                    ))
        

        if (step + 1) % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            inference(model, config, dataset)
            


        if step == config.train_steps:
            # If you receive a PyTorch data-loader error,
            # check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')
    print("Random sampling examples, temperature = {}".format(config.temperature))
    inference(model, config, dataset, "random")


def inference(model, config, dataset, sampling = 'greedy'):
    model.eval()
    gen_chars = random.sample(range(0,dataset.vocab_size),config.gen_sentence_num)
    gen_chars = torch.LongTensor(gen_chars).view(1, -1).to(config.device)

    input_x = gen_chars
    model.prev_state = None
        # generate senetence longer than T = 30 to see its effect
    for l in range(config.gen_sentence_len * 3):
        #probs shape:(seq_len,batch_size,vocab_size)
        probs, model.prev_state = model(input_x)
        if sampling == "greedy":
            predictions = torch.argmax(probs, dim=2)
        else:
            probs = torch.nn.functional.softmax(probs * config.temperature, dim = 2) 
            predictions = torch.multinomial(torch.squeeze(probs), 1).view(1,-1)
        gen_chars = torch.cat((gen_chars, predictions))
        input_x = predictions
            
    for i in range(gen_chars.shape[1]):
        sentence = dataset.convert_to_string(gen_chars[:,i].tolist())
        print("\nSentence {}".format(i))
        print(sentence[:config.gen_sentence_len])
        print(sentence)

###############################################################################
###############################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=False, 
                        default='{}/assets/book_EN_grimms_fairy_tails.txt'.format(BASE_DIR),
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--milestones', type=list, default= [100, 150],
                        help='Milestones for MultiStepLR')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1e6),
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="{}/summaries/".format(BASE_DIR),
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=500,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=2799,
                        help='How often to sample from the model')
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")
    parser.add_argument('--checkpoint_path', type=str, default="{}/checkpoints/".format(BASE_DIR),
                        help='Output path for checkpoints')

    # If needed/wanted, feel free to add more arguments
    # params for sequence generation
    parser.add_argument('--gen_sentence_len', type=int, default=30,
                        help='Length of generated sequences')
    parser.add_argument('--gen_sentence_num', type=int, default=8,
                        help='Number of sentence generated each time')
    # parser.add_argument('--sampling', type=str, default='random',
    #                     help='Sampling method: "greedy" or "random"')
    parser.add_argument('--temperature', type=float, default= 0.5,
                        help = 'tempaerature parameter for softmax')
                        

    config = parser.parse_args()

    # Train the model
    train(config)

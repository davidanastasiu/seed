#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
import torch

class Options():
   
    def __init__(self):
       
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        self.parser.add_argument('--train_seed', default=1010, help='random seed for train sampling')
        self.parser.add_argument('--val_seed', default=2007, help='random seed for val sampling')
        self.parser.add_argument('--stream_sensor', default='Ross_S_fixed', help='stream dataset')
        self.parser.add_argument('--rain_sensor', default='Ross_R_fixed', help='rain dataset')
        self.parser.add_argument('--is_stream', default=1, help='stream:1, reservoir:0')
        
        self.parser.add_argument('--batchsize', type=int, default=48, help='batch size of train data')
        self.parser.add_argument('--epochs', type=int, default=50, help='train epochs')
        self.parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
        self.parser.add_argument('--lradj', type=str, default='type4', help='learning rate adjustment policy')
        
        self.parser.add_argument('--train_volume', type=int, default=30000, help='train set size')
        self.parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dim of basic layers')
        self.parser.add_argument('--cnn_dim', type=int, default=384, help='hidden dim of cnn layers')
        self.parser.add_argument('--layer', type=int, default=1, help='number of layers')
        self.parser.add_argument('--stack_types', type=str, default= '"encoder","decoder","residue"', help='stacks')
        
        self.parser.add_argument('--input_dim', type=int, default=1, help='input dimension')
        self.parser.add_argument('--output_dim', type=int, default=1, help='output dimension')
        self.parser.add_argument('--input_len', type=int, default=15*24*4, help='length of input vector')
        self.parser.add_argument('--output_len', type=int, default=24*4*3, help='length of output vector')
        self.parser.add_argument('--inf_interval', type=int, default=24*1, help='hours in each inferencing loop')
        self.parser.add_argument('--times', type=int, default=8, help='output_len should be inf_interval * times')
        self.parser.add_argument('--sub_mean_threshold', type=float, default=4.5, help='define over threshold sub sequence as far')
        
        self.parser.add_argument('--event_focus_level', type=int, default=18, 
                                 help='validation sampling, 1 means most focus on extreme events, 0 follows the original distribution')
        self.parser.add_argument('--quantile', type=int, default=95, 
                                 help='quantile value')
        self.parser.add_argument('--val_size', type=int, default=120, help='validation set size')
        self.parser.add_argument('--val_near_days', type=int, default=288, help='validation point window avoid participate training')
        
        self.parser.add_argument('--start_point', type=str, default='1988-01-01 14:30:00', help='start time of the train set')
        self.parser.add_argument('--train_point', type=str, default='2021-08-31 23:30:00', help='end time of the train set')
        self.parser.add_argument('--test_start', type=str, default='2021-09-01 00:30:00', help='start time of the test set')
        self.parser.add_argument('--test_end', type=str,  default='2022-05-31 23:30:00', help='end time of the test set')
        self.parser.add_argument('--hinter_dim', type=int, default=1, help='dimension of hinter vector, must be 1 for multivariate')
        self.parser.add_argument('--oversampling', type=str, default=100, help='kruskal statistics threshold')
        self.parser.add_argument('--r_shift', type=int, default=288, help='shift positions of rain hinter')
                
        self.parser.add_argument('--watershed', type=int, default=0, help='1 if trained with rain info')
        self.parser.add_argument('--is_prob_feature', type=int, default=1, help='use gmm indicator')
        
        self.parser.add_argument('--gpu_id', type=int, default=0, help='gpu ids: e.g. 0. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        
        self.parser.add_argument('--model', type=str, default="ROSS_SEED", help='model label')
        self.parser.add_argument('--outf', default ='./output', help='output folder')

        self.opt = None

    def parse(self):
       
        self.opt = self.parser.parse_known_args()[0]

        torch.cuda.set_device(self.opt.gpu_id)

        args = vars(self.opt)

        self.opt.name = "%s" % (self.opt.model)
        expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        val_dir = os.path.join(self.opt.outf, self.opt.name, 'val')
        test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(val_dir):
            os.makedirs(val_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt


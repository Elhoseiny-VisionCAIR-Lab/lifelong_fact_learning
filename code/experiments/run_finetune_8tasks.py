import argparse
import sys
import os
import torch
sys.path.append('../')
from models.Finetune_elastic import *
from models.Finetune_objective_test import *
from utils.Eval_mAP import *

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--split', dest='split', required=True,
        help='Split to use, random or semantic')
    parser.add_argument(
        '--base_lr', dest='base_lr',
        help='Base learning rate', type=float)
    parser.add_argument(
        '--epochs', dest='epochs',
        help='Number of epochs', default=None)
    parser.add_argument(
        '--batch_size', dest='batch_size',
        help='The batch size', default=70)
    parser.add_argument(
        '--batch_not_multiplied', dest='batch_not_multiplied',
        help='The batch size', default=False, action='store_true')
    return parser.parse_args()

args = parse_args()
print('Called with args:')
print(args)
reg_lambda = 0
split = str(args.split) # random or semantic
num_of_gpu = torch.cuda.device_count()
if args.batch_not_multiplied:
    batch_size = int(args.batch_size)
else:
    batch_size = int(args.batch_size) * num_of_gpu
epochs = int(args.epochs)

batch_str = '_batch{}'.format(batch_size)


try:
    base_lr = float(args.base_lr)
    lr = base_lr * num_of_gpu * 2
    model_name = 'finetune_8tasks_{}_reg{}_lr{:.0e}_epochs{}{}'.format(split, reg_lambda, base_lr, epochs, batch_str)
except TypeError as e:
    base_lr = None
    lr=None
    model_name= 'finetune_8tasks_{}_reg{}_lr_def_epochs{}{}'.format(split, reg_lambda, epochs, batch_str)

print('reg_lambda', reg_lambda)
print('split', split)
print('base_lr', base_lr)
print('lr', lr)
print('epochs', epochs)
print('batch_size', batch_size)

def train_task(task_num):
    print('Training task:', task_num)

    if task_num == 1:
        ###1st task
        test_data_path = data_root + 'splits/8tasks_{}/B1_test.csv'.format(split)
        train_data_path = data_root + 'splits/8tasks_{}/B1_train.csv'.format(split)
        previous_task_model_path = ''

        exp_dir = exp_root + 't1/'

        finetune_elastic(root=data_root, batch=batch_size, train_data_path=train_data_path, test_data_path=test_data_path,
                         previous_task_model_path=previous_task_model_path, exp_dir=exp_dir, reg_lambda=0, epochs=epochs,
                         lr=lr, use_multiple_gpu=1)
    else:
        test_data_path = data_root + 'splits/8tasks_{}/B{}_test.csv'.format(split, task_num)
        train_data_path = data_root + 'splits/8tasks_{}/B{}_train.csv'.format(split, task_num)
        previous_task_model_path = exp_root + 't{}/'.format(task_num - 1) + 'model_best.pth.tar'
        exp_dir = exp_root + 't{}/'.format(task_num)

        finetune_elastic(root=data_root, batch=batch_size, train_data_path=train_data_path, test_data_path=test_data_path,
                                      previous_task_model_path=previous_task_model_path, exp_dir=exp_dir,
                                      reg_lambda=reg_lambda, epochs=epochs, lr=lr, use_multiple_gpu=1)

def train_tasks(number_of_tasks):
    for task_n in range(1, number_of_tasks + 1):
        train_task(task_n)

root = '../../'
data_root = root + 'data/large_scale/'

exp_root = root + 'checkpoints/large_scale/{}/'.format(model_name)

train_tasks(8)


save_CV_root = root + '/outputs/CV_feat/'
model_to_evaluate_path = exp_root + 't8/model_best.pth.tar'

batch=40
if not os.path.exists(save_CV_root + model_name):
    os.mkdir(save_CV_root + model_name)
if not os.path.exists(save_CV_root + model_name + '/T8'):
    os.mkdir(save_CV_root + model_name + '/T8')

for k in range(1, 9):
    test_data_path = root + '/data/large_scale/splits/8tasks_{}/B{}_test.csv'.format(split, k)
    save_CV_dir =  save_CV_root + model_name +  '/T8' + '/B%s'%str(k)
    #save in save_CV_dir + .mat
    extract_feat_mat(batch=batch,root=root,test_data_path=test_data_path, model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)

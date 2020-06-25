import argparse
import sys
import os
import torch
sys.path.append('/home/abdelksa/c2044/lifelong_fact_learning/code/')
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
        '--reg_lambda', dest='reg_lambda', required=True,
        help='Lambda value to use', type=float)
    parser.add_argument(
        '--base_lr', dest='base_lr',
        help='Base learning rate', type=float)
    parser.add_argument(
        '--trainval', dest='trainval',
        help='Whether to regularize on train and val sets', action='store_true')
    parser.add_argument(
        '--epochs', dest='epochs',
        help='Number of epochs', default=None)
    return parser.parse_args()

args = parse_args()
print('Called with args:')
print(args)
reg_lambda = float(args.reg_lambda)
split = str(args.split) # random or semantic
num_of_gpu = torch.cuda.device_count()
batch_size = 70 * num_of_gpu
trainval = args.trainval
epochs = int(args.epochs)
if epochs == 301:
    epochs_str = ''
else:
    epochs_str = '_epochs{}'.format(epochs)
if trainval:
    tv = '_trainval'
else:
    tv = ''
try:
    base_lr = float(args.base_lr)
    lr = base_lr * num_of_gpu * 2
    model_name = 'mas_8tasks_{}_reg{}_lr{:.0e}{}'.format(split, reg_lambda, base_lr, epochs_str) + tv
except TypeError as e:
    base_lr = None
    lr=None
    model_name= 'mas_8tasks_{}_reg{}_lr_def{}'.format(split, reg_lambda, epochs_str) + tv


print('reg_lambda', reg_lambda)
print('split', split)
print('base_lr', base_lr)
print('lr', lr)
print('trainval', trainval)
print('epochs', epochs)

def train_task(task_num):
    print('Training task:', task_num)

    if task_num == 1:
        ###1st task
        if split=='semantic':
            test_data_path = root + 'Benchmark_supplementary/large-scale_benchmarks/8tasks_{}/B1_BD_complete_test.csv'.format(split)
            train_data_path = root + 'Benchmark_supplementary/large-scale_benchmarks/8tasks_{}/B1_BD_complete_train.csv'.format(split)
        else:
            test_data_path = root + 'Benchmark_supplementary/large-scale_benchmarks/8tasks_{}/B1_test.csv'.format(split)
            train_data_path = root + 'Benchmark_supplementary/large-scale_benchmarks/8tasks_{}/B1_train.csv'.format(split)

        previous_task_model_path = ''  # put the path that you got from extracting this folder homes.esat.kuleuven.be/~raljundi/8tasks_t1_model.zip

        exp_dir = '/home/abdelksa/c2044/lifelong_learning/checkpoint/pytorch_models/{}/t1/'.format(model_name)

        finetune_elastic(root=root, batch=batch_size, train_data_path=train_data_path, test_data_path=test_data_path,
                         previous_task_model_path=previous_task_model_path, exp_dir=exp_dir, reg_lambda=0, epochs=epochs,
                         lr=lr, use_multiple_gpu=1)
    else:
        if split=='semantic':
            test_data_path = root + 'Benchmark_supplementary/large-scale_benchmarks/8tasks_{}/B{}_BD_complete_test.csv'.format(split, task_num)
            train_data_path = root + 'Benchmark_supplementary/large-scale_benchmarks/8tasks_{}/B{}_BD_complete_train.csv'.format(split, task_num)
        else:
            test_data_path = root + 'Benchmark_supplementary/large-scale_benchmarks/8tasks_{}/B{}_test.csv'.format(split, task_num)
            train_data_path = root + 'Benchmark_supplementary/large-scale_benchmarks/8tasks_{}/B{}_train.csv'.format(split, task_num)

        previous_task_model_path = exp_root + 't{}/'.format(task_num - 1) + 'model_best.pth.tar'  # put the path that you got from extracting this folder homes.esat.kuleuven.be/~raljundi/4tasks_t1_model.zip

        if split=='semantic':
            reg_sets = ['/Benchmark_supplementary/large-scale_benchmarks/8tasks_{}/B{}_BD_complete_train.csv'.format(split, task_num - 1)]
            if trainval:
                reg_sets.append('/Benchmark_supplementary/large-scale_benchmarks/8tasks_{}/B{}_BD_complete_test.csv'.format(split, task_num - 1))
        else:
            reg_sets = ['/Benchmark_supplementary/large-scale_benchmarks/8tasks_{}/B{}_train.csv'.format(split, task_num - 1)]
            if trainval:
                reg_sets.append('/Benchmark_supplementary/large-scale_benchmarks/8tasks_{}/B{}_test.csv'.format(split, task_num - 1))

        # '/data_splits/8tasks/B1_test.csv']
        exp_dir = exp_root + 't{}/'.format(task_num)

        finetune_objective_cumulative(root=root, batch=batch_size, train_data_path=train_data_path, test_data_path=test_data_path,
                                      previous_task_model_path=previous_task_model_path, exp_dir=exp_dir, norm='L2',
                                      reg_lambda=reg_lambda, reg_sets=reg_sets, epochs=epochs, lr=lr, use_multiple_gpu=1)

def train_tasks(number_of_tasks):
    for task_n in range(1, number_of_tasks + 1):
        train_task(task_n)

root = '/home/abdelksa/c2044/lifelong_learning/checkpoint/sherlock_LSC/'

# exp_root = '/home/abdelksa/c2044/lifelong_learning/checkpoint/pytorch_models/disjoint_8tasks/objective_reg30_comulative_train_val/'
# exp_root = '/home/abdelksa/c2044/lifelong_learning/6DS/Sherlock_data/pytorch_models/B2_elastic/reg_1/lr_06/'
exp_root = '/home/abdelksa/c2044/lifelong_learning/checkpoint/pytorch_models/{}/'.format(model_name)

train_tasks(8)


save_CV_root = root + '/outputs/CV_feat/'

# MAS

model_to_evaluate_path='/home/abdelksa/c2044/lifelong_learning/checkpoint/pytorch_models/{}/t8/model_best.pth.tar'.format(model_name)

batch=40
for k in range(1, 9):
    #test_data_path = root + '/data_splits/8task_complete_BD_replaced/B%s_BD_complete_test.csv'%str(k)
    if split == 'semantic':
        test_data_path = root + '/Benchmark_supplementary/large-scale_benchmarks/8tasks_{}/B{}_BD_complete_test.csv'.format(split, k)
    else:
        test_data_path = root + '/Benchmark_supplementary/large-scale_benchmarks/8tasks_{}/B{}_test.csv'.format(split, k)

    save_CV_dir =  save_CV_root + model_name + '/B%s'%str(k)
    #save in save_CV_dir + .mat
    extract_feat_mat(batch=batch,root=root,test_data_path=test_data_path, model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)

import argparse
import sys
import os
sys.path.append('/home/abdelksa/c2044/lifelong_learning/code/code/')
from Finetune_elastic import *
from Finetune_objective_test import *
import torch

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--split', dest='split', required=True,
        help='Split to use, random or semantic')
    parser.add_argument(
        '--base_lr', dest='base_lr',
        help='Base learning rate', default=None)
    parser.add_argument(
        '--epochs', dest='epochs',
        help='Number of epochs', default=None)
    return parser.parse_args()

args = parse_args()
print('Called with args:')
print(args)
reg_lambda = 0
split = str(args.split) # random or semantic
num_of_gpu = torch.cuda.device_count()
batch_size = 70 * num_of_gpu
epochs = int(args.epochs)

try:
    base_lr = float(args.base_lr)
    lr = base_lr * num_of_gpu * 2
    model_name = 'finetune_4tasks_{}_reg{}_lr{:.0e}_epochs{}'.format(split, reg_lambda, base_lr, epochs)
except TypeError as e:
    base_lr = None
    lr=None
    model_name= 'finetune_4tasks_{}_reg{}_lr_def_epochs{}'.format(split, reg_lambda, epochs)

print('reg_lambda', reg_lambda)
print('split', split)
print('base_lr', base_lr)
print('lr', lr)
print('epochs', epochs)

def train_task(task_num):
    print('Training task:', task_num)

    if task_num == 1:
        ###1st task
        test_data_path = root + 'Benchmark_supplementary/mid_scale_benchmarks/4tasks_{}/B1_test.cvs'.format(split)
        train_data_path = root + 'Benchmark_supplementary/mid_scale_benchmarks/4tasks_{}/B1_train.cvs'.format(split)
        previous_task_model_path = ''  # put the path that you got from extracting this folder homes.esat.kuleuven.be/~raljundi/4tasks_t1_model.zip

        exp_dir = '/home/abdelksa/c2044/lifelong_learning/6DS/Sherlock_data/pytorch_models/{}/t1/'.format(model_name)

        finetune_elastic(root=root, batch=batch_size, train_data_path=train_data_path, test_data_path=test_data_path,
                         previous_task_model_path=previous_task_model_path, exp_dir=exp_dir, reg_lambda=0, epochs=epochs,
                         lr=lr, use_multiple_gpu=1)
    else:
        test_data_path = root + 'Benchmark_supplementary/mid_scale_benchmarks/4tasks_{}/B{}_test.cvs'.format(split, task_num)
        train_data_path = root + 'Benchmark_supplementary/mid_scale_benchmarks/4tasks_{}/B{}_train.cvs'.format(split, task_num)
        # previous_task_model_path = '/private/home/elhoseiny/sherlock_continual/pytorch_models/disjoint_4tasks/model_best.pth.tar'  # put the path that you got from extracting this folder homes.esat.kuleuven.be/~raljundi/4tasks_t1_model.zip
        previous_task_model_path = exp_root + 't{}/'.format(task_num - 1) + 'model_best.pth.tar'  # put the path that you got from extracting this folder homes.esat.kuleuven.be/~raljundi/4tasks_t1_model.zip
        # previous_task_model_path = ''
        exp_dir = exp_root + 't{}/'.format(task_num)

        finetune_elastic(root=root, batch=batch_size, train_data_path=train_data_path, test_data_path=test_data_path,
                                      previous_task_model_path=previous_task_model_path, exp_dir=exp_dir,
                                      reg_lambda=reg_lambda, epochs=epochs, lr=lr, use_multiple_gpu=1)

def train_tasks(number_of_tasks):
    for task_n in range(1, number_of_tasks + 1):
        train_task(task_n)

from Eval_mAP import *
root = '/home/abdelksa/c2044/lifelong_learning/6DS/Sherlock_data/'

# exp_root = '/home/abdelksa/c2044/lifelong_learning/checkpoint/pytorch_models/disjoint_4tasks/objective_reg30_comulative_train_val/'
# exp_root = '/home/abdelksa/c2044/lifelong_learning/6DS/Sherlock_data/pytorch_models/B2_elastic/reg_1/lr_06/'
exp_root = '/home/abdelksa/c2044/lifelong_learning/6DS/Sherlock_data/pytorch_models/{}/'.format(model_name)

train_tasks(4)

save_CV_root = root + '/CV_feat/'
#model_to_evaluate_path='/home/abdelksa/c2044/lifelong_learning/6DS/Sherlock_data/pytorch_models/4tasks/t4/model_best.pth.tar'
model_to_evaluate_path='/home/abdelksa/c2044/lifelong_learning/6DS/Sherlock_data/pytorch_models/{}/t4/model_best.pth.tar'.format(model_name)

batch=40
for k in range(1,5):
    #test_data_path = root + '/data_splits/8task_complete_BD_replaced/B%s_BD_complete_test.csv'%str(k)
    test_data_path = root + '/Benchmark_supplementary/mid_scale_benchmarks/4tasks_{}/B{}_test.cvs'.format(split, k)
    save_CV_dir =  save_CV_root + model_name + '/B%s'%str(k)
    #save in save_CV_dir + .mat
    extract_feat_mat(batch=batch,root=root,test_data_path=test_data_path, model_to_evaluate_path=model_to_evaluate_path,save_CV_dir=save_CV_dir)
# # test the results
# print('Training Done.')
# print('Starting Evaluation.')
# from Eval_mAP import *
#
# final_task_exp_dir = exp_root + 't4/'
# model_to_evaluate_path = os.path.join(final_task_exp_dir, 'checkpoint.pth.tar')
# output_results_path = '4tasks_objective_reg30_comulative_train_val_onall.pth.tar'
# save_CV_dir = '/home/abdelksa/c2044/lifelong_learning/dump/'
# use_gpu_test=False
# print('\ntest_4tasks_model_onal')
# test_4tasks_model_onal(root=root, model_to_evaluate_path=model_to_evaluate_path,
#                                  output_results_path=output_results_path,
#                                  save_CV_dir=save_CV_dir, batch=int(batch_size/2))
#
# print('\neval_map_not_harsh_onall')
# eval_map_not_harsh_onall(root=root, model_to_evaluate_path=model_to_evaluate_path,
#                                    save_CV_dir=save_CV_dir, batch=int(batch_size/2))
#
# print('\neval_map_not_harsh_oneach')
# eval_map_not_harsh_oneach(root=root, model_to_evaluate_path=model_to_evaluate_path,
#                                      target_batch_path=root + 'data_splits/4tasks/B1_test.cvs',
#                                    save_CV_dir=save_CV_dir, batch=int(batch_size/2))
# eval_map_not_harsh_oneach(root=root, model_to_evaluate_path=model_to_evaluate_path,
#                                      target_batch_path=root + 'data_splits/4tasks/B2_test.cvs',
#                                      save_CV_dir=save_CV_dir, batch=int(batch_size/2))
# eval_map_not_harsh_oneach(root=root, model_to_evaluate_path=model_to_evaluate_path,
#                                      target_batch_path=root + 'data_splits/4tasks/B3_test.cvs',
#                                      save_CV_dir=save_CV_dir, batch=int(batch_size/2))
# eval_map_not_harsh_oneach(root=root, model_to_evaluate_path=model_to_evaluate_path,
#                                      target_batch_path=root + 'data_splits/4tasks/B4_test.cvs',
#                                      save_CV_dir=save_CV_dir, batch=int(batch_size/2))
#
# print('\neval_map_with_results')
# eval_map_with_results(root=root, model_to_evaluate_path=model_to_evaluate_path,
#                                  target_batch_path=root + 'data_splits/4tasks/B1_test.cvs',
#                                  save_CV_dir=save_CV_dir, batch=int(batch_size/2))
# eval_map_with_results(root=root, model_to_evaluate_path=model_to_evaluate_path,
#                                  target_batch_path=root + 'data_splits/4tasks/B2_test.cvs',
#                                  save_CV_dir=save_CV_dir, batch=int(batch_size/2))
# eval_map_with_results(root=root, model_to_evaluate_path=model_to_evaluate_path,
#                                  target_batch_path=root + 'data_splits/4tasks/B3_test.cvs',
#                                  save_CV_dir=save_CV_dir, batch=int(batch_size/2))
# eval_map_with_results(root=root, model_to_evaluate_path=model_to_evaluate_path,
#                                  target_batch_path=root + 'data_splits/4tasks/B4_test.cvs',
#                                  save_CV_dir=save_CV_dir, batch=int(batch_size/2))

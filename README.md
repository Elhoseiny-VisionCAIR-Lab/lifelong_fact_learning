#Continual Fact Learning

This a pytorch implementation of [insert paper link]
##Requirements

##Training:

To train one of the experiments in the paper follow the following steps:
1. `cd code/experiments`
2. run one of the scripts in the experiments folder as the below example: 
`python run_mas_8tasks.py --split random --reg_lambda 5 --base_lr 0.00005 --epochs 301 --trainval`

`split` the data split to train on. The options are: random and semantic

`reg_lambda` the regulizer lambda to use

`base_lr` the base learning rate

`epochs` the number of epochs for each task

`trainval` whether to use the validation sets along with the training sets of previous tasks as regularization sets

After training is done the CV_features are automatically extracted and saved in CV_feat, which is later used 
for evaluation.

##Evaluation:

To evaluate the trained models run on of the following files:
* `extract_results_lrs_random.py` for evaluating all the large-scale models on random split within `CV_feat` 
* `extract_results_lrs_semantic.py` for evaluating all the large-scale models on semantic within `CV_feat` 
* `extract_results_mds_random.py` for evaluating all the mid-scale models on random split within `CV_feat` 
* `extract_results_mds_semantic.py` for evaluating all the mid-scale models on semantic split within `CV_feat` 
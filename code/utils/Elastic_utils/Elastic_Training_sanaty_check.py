from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import copy
import os
import pdb
import shutil
from torch.utils.data import DataLoader
#end of imports#ELASTIC SGD
#from  torch.optim import Optimizer, required
class Elastic_SGD(optim.SGD):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        
        super(Elastic_SGD, self).__init__(params, lr,momentum,dampening,weight_decay,nesterov)
        
    def __setstate__(self, state):
        super(Elastic_SGD, self).__setstate__(state)
       
        
    def step(self, reg_params,closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        #print('************************DOING A STEP************************')
        #loss=super(Elastic_SGD, self).step(closure)
        loss = None
        if closure is not None:
            loss = closure()
        index=0
        reg_lambda=reg_params.get('lambda')
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
    
            for p in group['params']:
                #print('************************ONE PARAM************************')
                
                if p.grad is None:
                    continue
               
                #print('GRAD IS NON')
                #print( index)
                
                #pdb.set_trace()
                d_p = p.grad.data
                unreg_dp = p.grad.data.clone()
                reg_param=reg_params.get(p)
                if unreg_dp.equal(reg_param.get('grad_val')):
                    print('grad entering optimizer correctly')
                else:
                     print('grad NOT *** entering optimizer correctly')
                if p.data.equal(reg_param.get('param_val')):
                    print('param data entering optimizer correctly')
                else:
                    print('param data NOT ** entering optimizer correctly')
                #HERE MY CODE GOES
                
                #pdb.set_trace()
                omega=reg_param.get('omega')
                zero=torch.FloatTensor(p.data.size()).zero_()
                #if omega.equal(zero.cuda()):
                #    print('omega zero')
                #del zero
                init_val=reg_param.get('init_val')
                w=reg_param.get('w')
                curr_wegiht_val=p.data.clone()
                #pdb.set_trace()
                #move the variables to cuda
                init_val=init_val.cuda()
                w=w.cuda()
                omega=omega.cuda()
                #get the difference
                weight_dif=curr_wegiht_val.add(-1,init_val)
                
                regulizer=torch.mul(weight_dif,2*reg_lambda*omega)#hehehererere
                #pdb.set_trace()
                #JUST NOW PUT BACK
                d_p.add_(regulizer)
                del weight_dif
                del omega
                del init_val
                del regulizer
                #HERE MY CODE ENDS
                #sanity check
                 
                pdb.set_trace()
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                #I have to check  this shit
                
                if momentum != 0:
                    #pdb.set_trace()
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                        #pdb.set_trace()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                        #pdb.set_trace()
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                        #pdb.set_trace()
                    else:
                        d_p = buf
                        #pdb.set_trace()
                #
                p.data.add_(-group['lr'], d_p)
                #pdb.set_trace()
                w_diff=p.data.add(-1,curr_wegiht_val);
                #w_diff=torch.mul(d_p,-group['lr'])
                
                del curr_wegiht_val
                
                change=w_diff.mul(unreg_dp)
                del unreg_dp
                change=torch.mul(change,-1)
                del w_diff
                if 0:
                    if change.equal(zero.cuda()):
                        print('change zero')
                    if w.equal(zero.cuda()):
                        print('w zero')
                   
                    if w.equal(zero.cuda()):
                        print('w after zero')
                    x=p.data.add(-init_val)
                    if x.equal(zero.cuda()):
                        print('path diff is zero')
                del zero
                w.add_(change)
                reg_param['w']=w
                #pdb.set_trace()
                reg_param['param_val'] = p.data.clone()
                reg_param['data_val']=p.data.clone()
                reg_params[p]=reg_param
                index+=1
        return loss
#importance_dictionary: contains all the information needed for computing the w and omega


def train_model(model, criterion, optimizer, lr_scheduler,lr,dset_loaders,dset_sizes,use_gpu, num_epochs,exp_dir='./',resume=''):
    print('dictoinary length'+str(len(dset_loaders)))
    #reg_params=model.reg_params
    since = time.time()

    best_model = model
    best_acc = 0.0
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        #best_prec1 = checkpoint['best_prec1']
        #model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        #modelx = checkpoint['model']
        #model.reg_params=modelx.reg_params
        print('load')
        optimizer.load_state_dict(checkpoint['optimizer'])
        #pdb.
        #model.reg_params=reg_params
        
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
            start_epoch=0
            print("=> no checkpoint found at '{}'".format(resume))
    
    print(str(start_epoch))
    #pdb.set_trace()
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch,lr)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                model.zero_grad()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    #print('step')
                    record_grad_reg_params(model)
                    pdb.set_trace()
                    optimizer.step(model.reg_params)
                    check_recorded_reg_params(model)
                    pdb.set_trace()
                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                del outputs
                del labels
                del inputs
                del loss
                del preds
                best_acc = epoch_acc
                #best_model = copy.deepcopy(model)
                torch.save(model,os.path.join(exp_dir, 'best_model.pth.tar'))
                
        #epoch_file_name=exp_dir+'/'+'epoch-'+str(epoch)+'.pth.tar'
        epoch_file_name=exp_dir+'/'+'epoch'+'.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'alexnet',
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
                },epoch_file_name)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return model#initialize importance dictionary


def initialize_reg_params(model):

    reg_params={}
    for param in list(model.parameters()):
        w=torch.FloatTensor(param.size()).zero_()
        omega=torch.FloatTensor(param.size()).zero_()
        init_val=param.data.clone()
        reg_param={}
        reg_param['omega'] = omega
        reg_param['w'] = w
        reg_param['init_val'] = init_val
        reg_params[param]=reg_param
    return reg_params
   #initialize importance dictionary


def record_grad_reg_params(model):

    
    for param in list(model.parameters()):
        reg_param=model.reg_params.get(param)
       
       
        reg_param['grad_val'] = param.grad.data.clone()
        reg_param['param_val'] = param.data.clone()
        model.reg_params[param]=reg_param
   
   #initialize importance dictionary


def check_recorded_reg_params(model):

    
    for param in list(model.parameters()):
        reg_param=model.reg_params.get(param)
       
       
       
        if reg_param.get('param_val').equal( param.data):
            print('param after optimization is still correct')
        else:
            print('param after optimization is NOT **** correct')
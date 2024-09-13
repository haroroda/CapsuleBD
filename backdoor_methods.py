# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 23:24:37 2022

@author: haroroda
"""

import copy
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
from torchvision import datasets, transforms
from itertools import cycle
from torch.autograd import Variable

###---------------------------------DBA--------------------------------------###

def Mytest_poison(test_data_poison, dba_params, epoch, model):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    for batch_id, batch in enumerate(test_data_poison):
        data, targets, poison_num = get_poison_batch(dba_params, batch, adversarial_index=-1, evaluation=True)
        poison_data_count += poison_num
        dataset_size += len(data)
        output = model(data)
        total_loss += F.cross_entropy(output, targets,
                                                  reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()


    acc = 100.0 * (float(correct) / float(poison_data_count))  if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0
    #print('epoch: {}: Average loss: {:.4f}, '
    #                 'Accuracy: {}/{} ({:.4f}%)'.format(epoch, total_l, correct, poison_data_count, acc))
    model.train()
    return total_l, acc, correct, poison_data_count


def Mytest_poison_trigger(test_data_poison, dba_params, model, adver_trigger_index):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    adv_index = adver_trigger_index
    for batch_id, batch in enumerate(test_data_poison):
        data, targets, poison_num = get_poison_batch(dba_params, batch, adversarial_index=adv_index, evaluation=True)

        poison_data_count += poison_num
        dataset_size += len(data)
        output = model(data)
        total_loss += nn.functional.cross_entropy(output, targets,
                                                  reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0
    #print('Average loss: {:.4f}, '
    #                 'Accuracy: {}/{} ({:.4f}%)'.format(total_l, correct, poison_data_count,acc))

    model.train()
    return total_l, acc, correct, poison_data_count

def add_pixel_pattern(params, ori_image,adversarial_index):
    image = copy.deepcopy(ori_image)
    poison_patterns= []
    if adversarial_index==-1:
        for i in range(0,params['trigger_num']):
            poison_patterns = poison_patterns+ params[str(i) + '_poison_pattern']
    else :
        poison_patterns = params[str(adversarial_index) + '_poison_pattern']
    if params['type'] == 'cifar':
        for i in range(0,len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1
            image[1][pos[0]][pos[1]] = 1
            image[2][pos[0]][pos[1]] = 1


    elif params['type'] == 'mnist':

        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1
   
    return image


def get_poison_batch(params, bptt, adversarial_index=-1, evaluation=False):

    images, targets = bptt

    poison_count= 0
    new_images=copy.deepcopy(images)
    new_targets=copy.deepcopy(targets)

    for index in range(0, len(images)):
        if evaluation: # poison all data when testing
            new_targets[index] = params['poison_label_swap']
            new_images[index] = add_pixel_pattern(params, images[index],adversarial_index)
            poison_count += 1

        else: # poison part of data when training
            '''
            if index < params['poisoning_per_batch']:
                new_targets[index] = params['poison_label_swap']
                new_images[index] = add_pixel_pattern(params, images[index],adversarial_index)
                poison_count += 1
            else:
                new_images[index] = images[index]
                new_targets[index]= targets[index]
            '''
            new_targets[index] = params['poison_label_swap']
            new_images[index] = add_pixel_pattern(params, images[index],adversarial_index)
            poison_count += 1
       
    new_images = new_images.to(params['device'])
    new_targets = new_targets.to(params['device']).long()
    if evaluation:
        new_images.requires_grad_(False)
        new_targets.requires_grad_(False)

    #print(torch.mean(torch.norm((new_images-images.to(params['device'])).view((new_images-images.to(params['device'])).shape[0], -1), 2, dim=1)))
    return new_images,new_targets,poison_count

def model_dist_norm_var(model, target_params_variables, device, norm=2):
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    sum_var = torch.FloatTensor(size).fill_(0)
    sum_var= sum_var.to(device)
    size = 0
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (
                layer - target_params_variables[name]).view(-1)
        size += layer.view(-1).shape[0]

    return torch.norm(sum_var, norm)

def dba(model_idx, train_data, start_epoch, model, local_model_param, target_model_param, optimizer, mask, device, interval_series):
    
    ###---------------param--------------------###
    params = dict()
    params['adversary_list'] = [6,7,8,9]
    params['poison_epochs'] = interval_series
    '''[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58 ,59, 60,
                61, 62, 63, 64, 65, 66, 67, 68, 69, 70]'''
    params['aggr_epoch_interval'] = 1
    params['internal_poison_epochs'] = 6
    params['poison_label_swap'] = 0
    params['poisoning_per_batch'] = 5
    params['trigger_num'] = 4
    params['0_poison_pattern'] = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
    params['1_poison_pattern'] = [[0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14]]
    params['2_poison_pattern'] = [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]]
    params['3_poison_pattern'] = [[4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]
    params['type'] = 'cifar'
    params['alpha_loss'] = 1
    params['baseline'] = True
    params['scale_weights_poison'] = 1
    params['device'] = device
    params['batch_size'] = 128
    ###---------------param--------------------###
   
    train_dataloader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=False, num_workers=1)
    #for model_id in agent_name_keys:#range(helper.params['no_models']):
    last_local_model = copy.deepcopy(local_model_param)
    '''
    last_local_model = dict()
    
    for key in target_model_param.keys():
        if 'conv' in key:
            last_local_model[key] = target_model_param[key].cpu()*mask[key].clone()
        else:
            last_local_model[key] = target_model_param[key].cpu().clone()
    '''
    ## Synchronize LR and models
    model.load_state_dict(local_model_param)
    
    model.train()
    adversarial_index= -1
    localmodel_poison_epochs = params['poison_epochs']
    
    for temp_index in range(0, len(params['adversary_list'])):
        if model_idx == params['adversary_list'][temp_index]:
            adversarial_index= temp_index
            #localmodel_poison_epochs = params[str(temp_index) + '_poison_epochs']
            #print(
            #    f'poison local model {model_idx} index {adversarial_index} ')
            break
    if len(params['adversary_list']) == 1:
        adversarial_index = -1  # the global pattern

    for epoch in range(start_epoch, start_epoch + params['aggr_epoch_interval']):

        if epoch in localmodel_poison_epochs:
            internal_epoch_num = params['internal_poison_epochs']
            poison_data_count = 0
            total_loss = 0.
            correct = 0
            dataset_size = 0
            for internal_epoch in range(1, internal_epoch_num + 1):
                for batch_id, batch in enumerate(train_dataloader):
                    data, targets, poison_num = get_poison_batch(params, batch, adversarial_index=adversarial_index,evaluation=False)
                    L2_distance = torch.mean(torch.norm((data-batch[0].to(device)).view((data-batch[0].to(device)).shape[0], -1), 2, dim=1))
                    #print(L2_distance,L2_distance/3)
                    optimizer.zero_grad()
                    dataset_size += len(data)
                    poison_data_count += poison_num

                    output = model(data)
                    class_loss = nn.functional.cross_entropy(output, targets)

                    distance_loss = model_dist_norm_var(model, last_local_model, device)
                    # Lmodel = αLclass + (1 − α)Lano; alpha_loss =1 fixed
                    loss =params['alpha_loss'] * class_loss + \
                           (1 - params['alpha_loss']) * distance_loss
                    loss.backward()

                    # get gradients
                    optimizer.step()
                    total_loss += loss.data
                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size
            if start_epoch % 10 == 9:
                print(
                    '___PoisonTrain,  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%), train_poison_data_count: {}'.format(epoch, model_idx,
                                                                                  internal_epoch,
                                                                                  total_l, correct, dataset_size,
                                                                                 acc, poison_data_count))     
            if not params['baseline']:
                print('will scale.')
                clip_rate = params['scale_weights_poison']
                print("Scaling by  ",clip_rate)
                temp_param = dict()
                for key, value in model.cpu().state_dict().items():
                    target_value  = last_local_model[key]
                    new_value = target_value + (value.cpu() - target_value) * clip_rate
                    temp_param[key] = copy.deepcopy(model.state_dict()[key]+new_value.type_as(model.state_dict()[key]))
                    
                model.load_state_dict(temp_param)
                model.to(device)        

        
    return params, model.state_dict()

###---------------------------------DBA--------------------------------------###

###----------------------- -----neurotoxin-----------------------------------###
def test_poison_cv(params, data_source, model, device):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    num_data = 0.0
    poisoned_test_data = poison_dataset(data_source)
    for batch_id, batch in enumerate(poisoned_test_data):

        for pos in range(len(batch[0])):
            batch[1][pos] = params['poison_label_swap']

        data, target = batch
        data = data.to(device)
        target = target.to(device)
        data.requires_grad_(False)
        target.requires_grad_(False)

        output = model(data)
        total_loss += nn.functional.cross_entropy(output, target,
                                          reduction='sum').data.item()  # sum up batch loss
        num_data += target.size(0)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().to(dtype=torch.float)

    acc = 100.0 * (float(correct) / float(num_data))

    return acc

def sample_poison_data(target_class, test_dataset):
    cifar_poison_classes_ind = []
    for ind, x in enumerate(test_dataset):
        imge, label = x

        if label == target_class:
            cifar_poison_classes_ind.append(ind)

    return cifar_poison_classes_ind

def poison_dataset(data):
    size_of_secret_dataset = 500
    batch_size = 128
    # base_case即为标准对抗样本生成
    indices = list()

    ### Base case sample attackers training and testing data
    range_no_id = sample_poison_data(5, data)

    while len(indices) < size_of_secret_dataset:
        range_iter = random.sample(range_no_id,
                                   np.min([batch_size, len(range_no_id) ]))
        indices.extend(range_iter)

    poison_images_ind = indices
    ### self.poison_images_ind_t = list(set(range_no_id) - set(indices))

    return torch.utils.data.DataLoader(data,
                       batch_size=batch_size,
                       sampler=torch.utils.data.sampler.SubsetRandomSampler(poison_images_ind))

def grad_mask_cv(model, dataset_clearn, device, ratio=0.5):
    """Generate a gradient mask based on the given dataset"""
    model.train()
    model.zero_grad()

    train_data = torch.utils.data.DataLoader(dataset_clearn, batch_size=128, shuffle=True)
    for batch_id, batch in enumerate(train_data):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        output = model(inputs)
        loss = F.cross_entropy(output, labels)
        loss.backward(retain_graph=True)

    mask_grad_list = []
    
    grad_list = []
    grad_abs_sum_list = []
    k_layer = 0
    for _, parms in model.named_parameters():
        if parms.requires_grad:
            grad_list.append(parms.grad.abs().view(-1))

            # 将梯度求和
            grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())

            k_layer += 1

    grad_list = torch.cat(grad_list)
    # 取每组梯度(行)最小的百分之k个
    _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
    mask_flat_all_layer = torch.zeros(len(grad_list))
    mask_flat_all_layer[indices] = 1.0

    count = 0
    percentage_mask_list = []
    k_layer = 0
    grad_abs_percentage_list = []
    # mask_grad_list为0,1矩阵，标记每组梯度最小的百分之k个的位置
    for _, parms in model.named_parameters():
        if parms.requires_grad:

            gradients_length = len(parms.grad.abs().view(-1))

            mask_flat = mask_flat_all_layer[count:count + gradients_length ]
            mask_grad_list.append(mask_flat.reshape(parms.grad.size()))

            count += gradients_length

            percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

            percentage_mask_list.append(percentage_mask1)

            grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))

            k_layer += 1

    model.zero_grad()
    return mask_grad_list

def apply_grad_mask(model, mask_grad_list, device):
    mask_grad_list_copy = iter(mask_grad_list)
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            parms.grad = parms.grad * next(mask_grad_list_copy).to(device)
    return model

def train_cv_poison(params, model, poison_optimizer, mask_grad_list, epoch, device,
                    poisoned_train_data, benign_train_data):

    # 更改标签生成对抗样本并训练
    benign_train_data = torch.utils.data.DataLoader(benign_train_data, batch_size=128, shuffle=True)
    for (x1, x2) in zip(poisoned_train_data, benign_train_data):
        inputs_p, labels_p = x1
        inputs_c, labels_c = x2
        inputs = torch.cat((inputs_p,inputs_c))

        # 将干净样本的7标签改为9
        for pos in range(labels_c.size(0)):
            if labels_c[pos] == 7:
                labels_c[pos] = params['poison_label_swap']

        # 所有有毒数据标签均改为9
        for pos in range(labels_p.size(0)):
            labels_p[pos] = params['poison_label_swap']

        labels = torch.cat((labels_p,labels_c))

        inputs, labels = inputs.to(device), labels.to(device)
        poison_optimizer.zero_grad()

        output = model(inputs)
        loss = F.cross_entropy(output, labels)
        loss.backward(retain_graph=True)

        # 攻击者只保留与bottom-k%位置的梯度上传Server
        if params['gradmask_ratio'] != 1:
            #model = apply_grad_mask(model, mask_grad_list, device)
            mask_grad_list_copy = iter(mask_grad_list)
            for name, parms in model.named_parameters():
                if parms.requires_grad:
                    parms.grad = parms.grad * next(mask_grad_list_copy).to(device)
        poison_optimizer.step()

    return model.state_dict()
    
def neurotoxin(model_idx, epoch, train_data, model, local_model_param, optimizer, device, testloder):

    ###---------------param--------------------###
    params = {}
    params['gradmask_ratio'] = 0.95
    params['retrain_poison'] = 10
    params['poison_label_swap'] = 9
    ###---------------param--------------------###
    weight_accumulator = dict()
    for name in local_model_param.keys():
        weight_accumulator[name] = torch.zeros_like(local_model_param[name])

    # 指定客户端训练并提供梯度信息
    model.load_state_dict(local_model_param)
    poisoned_train_data = poison_dataset(train_data)
    model.train()

    # 开始攻击
    print('P o i s o n - n o w ! ----------')

    # get gradient mask use global model and clearn data
    # ——————重建后bottom-k%的梯度mask_grad_list—————— 文章创新点，gradmask_ratio参数即为k
    if params['gradmask_ratio'] != 1 :

        # 得到bottom-k%的掩码
        mask_grad_list = grad_mask_cv(model, train_data, device, ratio=params['gradmask_ratio'])
    else:
        mask_grad_list = None

    # 攻击10轮
    for internal_epoch in range(params['retrain_poison']):
        
        param = train_cv_poison(params, model, optimizer, mask_grad_list, epoch, device,
                                poisoned_train_data, train_data)
        model.load_state_dict(param)
    
    print(test_poison_cv(params, testloder, model,device))
    return params, model.state_dict()

###----------------------- -----neurotoxin-----------------------------------###

###----------------------- --------tail- ------------------------------------###
def load_poisoned_dataset(dataset, dataset_name, testset):
    
    if dataset_name in ("mnist"):
        
        fraction=0.1 #10

        with open("poisoned_dataset_fraction_{}".format(fraction), "rb") as saved_data_file:
            poisoned_dataset = torch.load(saved_data_file)
        emnist_test_dataset = datasets.EMNIST('./data', split="digits", train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        fashion_mnist_test_dataset = datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        vanilla_test_loader = torch.utils.data.DataLoader(emnist_test_dataset,
             batch_size=128, shuffle=False)
        poisoned_train_loader = torch.utils.data.DataLoader(poisoned_dataset,
             batch_size=128, shuffle=True)
        targetted_task_test_loader = torch.utils.data.DataLoader(fashion_mnist_test_dataset,
             batch_size=128, shuffle=False)

    
    elif dataset_name == "cifar10":

        poisoned_trainset = copy.deepcopy(dataset)

        with open('./saved_datasets/southwest_images_new_train.pkl', 'rb') as train_f:
            saved_southwest_dataset_train = pickle.load(train_f)

        with open('./saved_datasets/southwest_images_new_test.pkl', 'rb') as test_f:
            saved_southwest_dataset_test = pickle.load(test_f)
        
        sampled_targets_array_test = 9 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int)

        #
        print("OOD (Southwest Airline) train-data shape we collected: {}".format(saved_southwest_dataset_train.shape))
        #sampled_targets_array_train = 2 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as bird
        sampled_targets_array_train = 9 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as truck
        
        print("OOD (Southwest Airline) test-data shape we collected: {}".format(saved_southwest_dataset_test.shape))
        
        # downsample the poisoned dataset #################
        #num_sampled_poisoned_data_points = 100 # N
        num_sampled_poisoned_data_points = saved_southwest_dataset_train.shape[0]
        samped_poisoned_data_indices = np.random.choice(saved_southwest_dataset_train.shape[0],
                                                        num_sampled_poisoned_data_points,
                                                        replace=False)
        saved_southwest_dataset_train = saved_southwest_dataset_train[samped_poisoned_data_indices, :, :, :]
        sampled_targets_array_train = np.array(sampled_targets_array_train)[samped_poisoned_data_indices]
        print(saved_southwest_dataset_train.shape[0])
        ######################################################


        # downsample the raw cifar10 dataset #################
        #num_sampled_data_points = 400 # M
        #samped_data_indices = np.random.choice(len(poisoned_trainset), len(poisoned_trainset), replace=False)
        #poisoned_trainset.data = poisoned_trainset[samped_data_indices, :, :, :][0]
        #poisoned_trainset.targets = np.array(poisoned_trainset.targets)[samped_data_indices]
        # keep a copy of clean data
       
        ########################################################
        poisoned_trainset.data = saved_southwest_dataset_train
        poisoned_trainset.targets = sampled_targets_array_train
        for i, (data,label) in enumerate(dataset):
            
            data = data.reshape(1,32,32,3)
            poisoned_trainset.data = np.append(data, poisoned_trainset.data, axis=0)
            poisoned_trainset.targets = np.insert(poisoned_trainset.targets,0,label)
        
        poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=128,  shuffle=True)
        vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
        
        poisoned_testset = copy.deepcopy(testset)
        poisoned_testset.data = saved_southwest_dataset_test
        poisoned_testset.targets = sampled_targets_array_test
        targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=128, shuffle=False)
        
    return poisoned_train_loader, vanilla_test_loader, targetted_task_test_loader

def test(model, device, test_loader, test_batch_size, criterion, mode="raw-task", dataset="cifar10"):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    if dataset in ("mnist", "emnist"):
        target_class = 7
        if mode == "raw-task":
            classes = [str(i) for i in range(10)]
        elif mode == "targetted-task":
            classes = ["T-shirt/top", 
                        "Trouser",
                        "Pullover",
                        "Dress",
                        "Coat",
                        "Sandal",
                        "Shirt",
                        "Sneaker",
                        "Bag",
                        "Ankle boot"]
    elif dataset == "cifar10":
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # target_class = 2 for greencar, 9 for southwest
        
        target_class = 9

    model.eval()
    test_loss = 0
    correct = 0
    final_acc = 0
    task_acc = None

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            c = (predicted == target).squeeze()

            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # check backdoor accuracy
            
            #for image_index in range(test_batch_size):
            for image_index in range(len(target)):
                label = target[image_index]
                class_correct[label] += c[image_index].item()
                class_total[label] += 1
    test_loss /= len(test_loader.dataset)
    print(class_total)
    if mode == "raw-task":
        for i in range(10):
            print('Accuracy of %5s : %.2f %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))

            if i == target_class:
                task_acc = 100 * class_correct[i] / class_total[i]

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        final_acc = 100. * correct / len(test_loader.dataset)

    elif mode == "targetted-task":

        if dataset in ("mnist", "emnist"):
            for i in range(10):
                print('Accuracy of %5s : %.2f %%' % (
                    classes[i], 100 * class_correct[i] / class_total[i]))
            
            # trouser acc
            final_acc = 100 * class_correct[1] / class_total[1]
        
        elif dataset == "cifar10":
            print('#### Targetted Accuracy of %5s : %.2f %%' % (classes[target_class], 100 * class_correct[target_class] / class_total[target_class]))
            final_acc = 100 * class_correct[target_class] / class_total[target_class]
    return final_acc, task_acc



def train(model, device, train_loader, optimizer, epoch, criterion, eps=5e-4, model_original=None,
        proj="l_2", project_frequency=1):
    """
        train function for both honest nodes and adversary.
        NOTE: this trains only for one epoch
    """
    model.train()
    # get learning rate

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        loss.backward()
        
        optimizer.step()

    #print('Train Epoch: {} ,tLoss: {:.6f}'.format(epoch, loss.item()))
    
            
def tail(flr, net_avg, model_param, device, dataset_name,  poisoned_dataset, vanilla_emnist_test_loader, targetted_task_test_loader, optimizer):
    params = {}
    params['gamma'] = 0.998
    params['adversarial_local_training_period'] = 20
    params['eps'] = 2
    params['project_frequency'] = 10
    
    #optimizer = torch.optim.Adam(net_avg.parameters(),lr=0.001)
    # randomly select participating clients
    # in this current version, we sample `part_nets_per_round-1` per FL round since we assume attacker will always participates
    
    #pdb.set_trace()
    net_avg.load_state_dict(model_param)
    # we need to reconstruct the net list at the beginning
    model_original = list(model_param)
    # super hacky but I'm doing this for the prox-attack
   
    criterion = nn.CrossEntropyLoss()

    for e in range(1, params['adversarial_local_training_period']+1):
       # we always assume net index 0 is adversary
       train(net_avg, device, poisoned_dataset, optimizer, e, criterion=criterion,
                eps=params['eps'], model_original=model_original, project_frequency=params['project_frequency'])

           
    if flr % 10 == 9:
        test(net_avg, device, vanilla_emnist_test_loader, test_batch_size=128, criterion=criterion, mode="raw-task", dataset=dataset_name)
        test(net_avg, device, vanilla_emnist_test_loader, test_batch_size=128, criterion=criterion, mode="targetted-task", dataset=dataset_name)
        #test(net_avg, device, targetted_task_test_loader, test_batch_size=128, criterion=criterion, mode="raw-task", dataset=dataset_name)
        test(net_avg, device, targetted_task_test_loader, test_batch_size=128, criterion=criterion, mode="targetted-task", dataset=dataset_name)

    # at here we can check the distance between w_bad and w_g i.e. `\|w_bad - w_g\|_2`
    # we can print the norm diff out for debugging
    return net_avg.state_dict()

###----------------------- --------tail- ------------------------------------###
    
###----------------------- --------PFedBA- ------------------------------------###
def PFedBA_init(args, data):
    '''
    class Trigger(nn.Module):
        def __init__(self):
            super(Trigger, self).__init__()
            # 使用 nn.Parameter 创建可训练参数
            self.weight = nn.Parameter(torch.rand(10,10), requires_grad=True)
        def forward(self, x):
            pad=nn.ZeroPad2d(padding=(11,11,11,11))
            mask = torch.ones_like(x)
            mask[:,:,11:21,11:21] = 0
            return x*mask + pad(self.weight)
        
    trigger = Trigger()
    '''
    trigger = torch.rand(10,10)
    poison_data = []
    benign_data = []
    for i in range(len(data)):
        if i < int(len(data)/4):
            poison_data.append(data[i])
        else:
            benign_data.append(data[i])
    return trigger, poison_data, benign_data

def PFedBA_test(args, trigger, test_data, target_model,device):
    target_label = 0
    target_model.eval()
    #trigger.eval()
    data = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)
    data_count = 0
    num_correct_fack = 0
    num_correct_real = 0
    for i, (images, labels) in enumerate(data, start=0):
        
        images, labels = images.to(device), labels.to(device)
        data_count += len(labels)
        #m_data = trigger(images)
        pad=nn.ZeroPad2d(padding=(11,11,11,11))
        mask = torch.ones_like(images)
        mask[:,:,11:21,11:21] = 0
        m_data = images*mask + pad(trigger)    
            
            
        predict = target_model(m_data)
        pred_lab = torch.argmax(predict, 1)
        target_labels = torch.ones(labels.shape).long()*target_label
        target_labels = target_labels.to(device)
        num_correct_fack += torch.sum(pred_lab==target_labels,0)
        
        
        predict = target_model(images)
        pred_lab = torch.argmax(predict, 1)
        num_correct_real += torch.sum(pred_lab==labels,0)
    
    return num_correct_fack/data_count, num_correct_real/data_count

def PFedBA(args, trigger, poison_data, benign_data, target_model, device, epoch):
    target_label = 0
    target_model2 = copy.deepcopy(target_model)
    poison_data = torch.utils.data.DataLoader(poison_data, batch_size=64, shuffle=True, num_workers=1)
    benign_data = torch.utils.data.DataLoader(benign_data, batch_size=64, shuffle=True, num_workers=1)
    #optimizer_t = torch.optim.Adam(trigger.parameters(),lr=0.0005)
    optimizer_m1 = torch.optim.Adam(target_model.parameters(),lr=0.001)
    optimizer_m2 = torch.optim.Adam(target_model2.parameters(),lr=0.001)
    trigger = trigger.to(device)
    #trigger.train()
    criterion = nn.CrossEntropyLoss()
    criterion_mse = torch.nn.MSELoss()
    '''
    if epoch == 0:
        for _ in range(5):
            num_correct_real = 0
            data_count = 0
    
            for i, (img,label) in enumerate(poison_data):
                img, label = img.to(device), label.to(device)
                data_count += len(label)
                m_data = trigger(img)
                
                target_labels = torch.ones(label.shape).long()*target_label
                optimizer_t.zero_grad()
                output = target_model(m_data)
    
                loss = criterion(output, target_labels.to(device))
                pred_lab = torch.argmax(output, 1)
                num_correct_real += torch.sum(pred_lab==target_labels.to(device),0)
                loss.backward()
                
            
                optimizer_t.step()
         
            print(num_correct_real/data_count)
    target_model.eval()
    '''
    '''
    for i, (img,label) in enumerate(poison_data):
        img, label = img.to(device), label.to(device)
        m_data =  trigger(img)
        target_labels = torch.ones(label.shape).long()*target_label
        optimizer_m1.zero_grad()
        output = target_model(m_data)
        loss_model1 = criterion(output, target_labels.to(device))
        loss_model1.backward()
        optimizer_m1.step()
        
        
        optimizer_m2.zero_grad()
        output2 = target_model2(img)
        loss_model2 = criterion(output2, label.to(device))
        loss_model2.backward()
        optimizer_m2.step()
        
        loss_mse = 0
        for param1,param2 in zip(target_model.parameters(),target_model2.parameters()):
            optimizer_t.zero_grad()
            loss_mse += criterion_mse(param1, param2)
        print(loss_mse)
        loss_mse.backward()
        optimizer_t.step()
          
    ''' 
    target_model.train()
    #trigger.eval()
    for _ in range(5):
        '''
        for _, (d1,d2) in enumerate(zip(cycle(poison_data),benign_data)):
            d1[0], d2[0] = d1[0].to(device), d2[0].to(device)
            d1[1], d2[1] = d1[1].to(device), d2[1].to(device)
            pad=nn.ZeroPad2d(padding=(11,11,11,11))
            mask = torch.ones_like(d1[0])
            mask[:,:,11:21,11:21] = 0
            m_data = d1[0]*mask + pad(trigger)    
            images = torch.cat((m_data,d2[0]),0)
            labels = torch.cat((torch.ones(d1[1].shape).to(device).long()*target_label,d2[1].to(device)),0)
            optimizer_m1.zero_grad()
            output = target_model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer_m1.step()
        
        '''
        for i, (images, labels) in enumerate(poison_data, start=0):
        
            images, labels = images.to(device), labels.to(device)
            #m_data = trigger(images)
            pad=nn.ZeroPad2d(padding=(11,11,11,11))
            mask = torch.ones_like(images)
            mask[:,:,11:21,11:21] = 0
            m_data = images*mask + pad(trigger)    
            target_labels = torch.ones(labels.shape).long()*target_label
            target_labels = target_labels.to(device)
            optimizer_m1.zero_grad()
            output = target_model(m_data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer_m1.step()
        print(loss)
        '''
        for i, (images, labels) in enumerate(poison_data, start=0):
        
            images, labels = images.to(device), labels.to(device)

            optimizer_m1.zero_grad()
            output = target_model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer_m1.step()
            '''
    return trigger,target_model.state_dict()

    
###----------------------- --------PFedBA- ------------------------------------###
    
###----------------------- --------cerp- ------------------------------------###
def get_batch(device, train_data, bptt, evaluation=False):
    data, target = bptt
    data = data.to(device)
    target = target.to(device)
    if evaluation:
        data.requires_grad_(False)
        target.requires_grad_(False)
    return data, target

def cerp_get_poison_batch(params, bptt, noise_trigger, device, adversarial_index=-1, evaluation=False):

    images, targets = bptt
    params['poisoning_per_batch'] = 5
    poison_count = 0
    new_images = images
    new_targets = targets

    
    for index in range(0, images.shape[0]):
        if evaluation:  # poison all data when testing
            new_targets[index] = params['poison_label_swap']
            new_images[index] = cerp_add_pixel_pattern(params, images[index], noise_trigger)
            poison_count += 1

        else:  # poison part of data when training

            if index < params['poisoning_per_batch']:
                new_targets[index] = params['poison_label_swap']
                new_images[index] = cerp_add_pixel_pattern(params, images[index], noise_trigger)
                poison_count += 1
            else:
                new_images[index] = images[index]
                new_targets[index] = targets[index]

    new_images = new_images.to(device)
    new_targets = new_targets.to(device).long()
    if evaluation:
        new_images.requires_grad_(False)
        new_targets.requires_grad_(False)
    return new_images, new_targets, poison_count

def cerp_add_pixel_pattern(params, ori_image, noise_trigger):
    image = copy.deepcopy(ori_image)
    noise = torch.tensor(noise_trigger).cpu()
    poison_patterns = []
    for i in range(0, params['trigger_num']):
        poison_patterns = poison_patterns + params[str(i) + '_poison_pattern']
    for i in range(0, len(poison_patterns)):
        pos = poison_patterns[i]
        image[0][pos[0]][pos[1]] = noise[0][pos[0]][pos[1]]
        image[1][pos[0]][pos[1]] = noise[1][pos[0]][pos[1]]
        image[2][pos[0]][pos[1]] = noise[2][pos[0]][pos[1]]

    image = torch.clamp(image, -1, 1)

    return image

def cifar100_trigger(params, train_data, device, local_model, target_model, noise_trigger,intinal_trigger):
    print("start trigger fine-tuning")
    init = False
    # load model
    mmodel = copy.deepcopy(local_model)
    mmodel.load_state_dict(target_model.state_dict())
    mmodel.eval()
    mmodel = mmodel.to(device)
    pre_trigger = torch.tensor(noise_trigger).to(device)
    aa = copy.deepcopy(intinal_trigger).to(device) 

    corrects = 0
    datasize = 0
    for poison_id in params['adversary_list']:
        data_iterator = train_data[poison_id]
        data_iterator = torch.utils.data.DataLoader(data_iterator, batch_size=256, shuffle=True, num_workers=1)
        for batch_id, (datas, labels) in enumerate(data_iterator):
            datasize += len(datas)
            x = Variable(datas.to(device))
            y = Variable(labels.to(device))
            y_target = torch.LongTensor(y.size()).fill_(int(params['poison_label_swap']))
            y_target = Variable(y_target.to(device), requires_grad=False)
            if not init:
                noise = copy.deepcopy(pre_trigger)
                noise = Variable(noise.to(device), requires_grad=True)
                init = True

            for index in range(0, len(x)):
                for i in [0, 4]:
                    for j in [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14]:
                        x[index][0][i][j] = 0
                        x[index][1][i][j] = 0
                        x[index][2][i][j] = 0

            output = mmodel((x + noise).float())
            classloss = nn.functional.cross_entropy(output, y_target)
            loss = classloss
            mmodel.zero_grad()
            if noise.grad:
                noise.grad.fill_(0)
            loss.backward(retain_graph=True)

            noise = noise - noise.grad * 0.1
            for i in range(32):
                for j in range(32):
                    if i in [0, 4] and j in [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14]:
                        continue
                    else:
                        noise[0][i][j] = 0
                        noise[1][i][j] = 0
                        noise[2][i][j] = 0

            delta_noise = noise - aa
            proj_lp = delta_noise * min(1, 10 / torch.norm(delta_noise))
            noise = aa + proj_lp

            noise = Variable(noise.data.to(device), requires_grad=True)
            pred = output.data.max(1)[1]
            corrects += pred.eq(y_target.data.view_as(pred)).cpu().sum().item()

    dataset_size = 0
    total_loss = 0
    correctt = 0
    for i in range(0, len(params['adversary_list'])):
        state_key = params['adversary_list'][i]
        data_iterator = train_data[state_key]
        data_iterator = torch.utils.data.DataLoader(data_iterator, batch_size=256, shuffle=True, num_workers=1)
        for batch_id, batch in enumerate(data_iterator):
            data, targets = get_batch(device, data_iterator, batch, evaluation=True)
            y_target = torch.LongTensor(targets.size()).fill_(int(params['poison_label_swap']))
            y_target = Variable(y_target.to(device), requires_grad=False)
            dataset_size += len(data)
            data = torch.clamp(data + noise, -1, 1)
            output = mmodel((data).float())
            total_loss += nn.functional.cross_entropy(output, y_target, reduction='sum').item()
            pred = output.data.max(1)[1]
            correctt += pred.eq(y_target.data.view_as(pred)).cpu().sum().item()

    return noise

def cerp_initial(test_data,device):
    params = {}
    params['adversary_list'] = [6,7,8,9]
    params['poison_label_swap'] = 2
    params['trigger_num'] = 4
    params['poisoning_per_batch'] = 5
    params['0_poison_pattern'] = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]
    params['1_poison_pattern'] = [[0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14]]
    params['2_poison_pattern'] = [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]]
    params['3_poison_pattern'] = [[4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]
    params['poison_lr'] = 0.005#0.005
    params['internal_poison_epochs'] = 1
    params['poison_step_lr'] = True
    params['momentum'] = 0.9
    params['decay'] = 0.0005
    params['lr'] = 0.01#0.01
    params['alpha_loss'] = 0.0001
    params['beta_loss'] = 0.0001
    params['batch_size'] = 64
    
    data_iterator = test_data
    triggervalue = 1
    for batch_id, (datas, labels) in enumerate(data_iterator):
        x = Variable(datas.to(device))
        sz = x.size()#[1:]
        intinal_trigger = torch.zeros(sz)
        break
    poison_patterns = []
    for i in range(0, params['trigger_num']):
        poison_patterns = poison_patterns + params[str(i) + '_poison_pattern']
    for i in range(0, len(poison_patterns)):
        pos = poison_patterns[i]
        intinal_trigger[0][pos[0]][pos[1]] = triggervalue  # +delta i  #？
        intinal_trigger[1][pos[0]][pos[1]] = triggervalue  # +delta i
        intinal_trigger[2][pos[0]][pos[1]] = triggervalue  # +delta i

    noise_trigger = copy.deepcopy(intinal_trigger)
    return params, noise_trigger, intinal_trigger

def cerp_model_dist_norm_var(model, target_params_variables, device, norm=2):
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    sum_var = torch.FloatTensor(size).fill_(0)
    sum_var = sum_var.to(device)
    size = 0
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (
                layer - target_params_variables[name]).view(-1)
        size += layer.view(-1).shape[0]

    return torch.norm(sum_var, norm)

def model_cosine_similarity(model, target_params_variables,
                                model_id='attacker'):

    cs_list = list()
    for name, data in model.named_parameters():
        if name == 'decoder.weight':
            continue

        model_update = (data.view(-1) - target_params_variables[name].view(-1)) + target_params_variables[
            name].view(-1)

        cs = F.cosine_similarity(model_update,
                                 target_params_variables[name].view(-1), dim=0)
        cs_list.append(cs)

    return sum(cs_list) / len(cs_list)

def poison_test_dataset(params, test_dataset):
    print('get poison test loader')
    # delete the test data with target label
    test_classes = {}
    for ind, x in enumerate(test_dataset):
        _, label = x
        if label in test_classes:
            test_classes[label].append(ind)
        else:
            test_classes[label] = [ind]

    range_no_id = list(range(0, len(test_dataset)))
    for image_ind in test_classes[params['poison_label_swap']]:
        if image_ind in range_no_id:
            range_no_id.remove(image_ind)
    poison_label_inds = test_classes[params['poison_label_swap']]

    return torch.utils.data.DataLoader(test_dataset,
                                       batch_size=params['batch_size'],
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                           range_no_id)), \
           torch.utils.data.DataLoader(test_dataset,
                                       batch_size=params['batch_size'],
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                           poison_label_inds))


def cerp(params, IsTrigger, train_data, model_idx, device, targeted_model, noise_trigger, intinal_trigger, epoch):
    
    normalOptimizer = torch.optim.SGD(targeted_model.parameters(), lr=params['lr'],
                                          momentum=params['momentum'],
                                          weight_decay=params['decay'])
    mmodel = targeted_model
    normalmodel = copy.deepcopy(targeted_model)
    normalmodel.load_state_dict(targeted_model.state_dict())
    last_local_model = dict()
    poisonloss_dict = dict()
    client_grad = []
    for name, data in targeted_model.state_dict().items():
        last_local_model[name] = targeted_model.state_dict()[name].clone()
    print('poison_now')
    
    if not IsTrigger:
        tuned_trigger = cifar100_trigger(params, train_data, device, targeted_model, targeted_model, noise_trigger, intinal_trigger)
        IsTrigger = True

    poison_lr = params['poison_lr']
    internal_epoch_num = params['internal_poison_epochs']
    step_lr = params['poison_step_lr']

    data_iterator = train_data[model_idx]
    data_iterator = torch.utils.data.DataLoader(data_iterator, batch_size=256, shuffle=True, num_workers=1)
    normalData_size = 0
    for batch_id, batch in enumerate(data_iterator):
        normalOptimizer.zero_grad()
        normalData, normalTargets = get_batch(device, data_iterator, batch, evaluation=False)
        normalData_size += len(normalData)
        normaloutput = normalmodel(normalData)
        loss = nn.functional.cross_entropy(normaloutput, normalTargets)
        loss.backward()
        normalOptimizer.step()
    normal_params_variables = dict()
    for name, param in normalmodel.named_parameters():
        normal_params_variables[name] = normalmodel.state_dict()[name].clone().detach().requires_grad_(
            False)

    normalmodel_updates_dict = dict()

    for name, data in normalmodel.state_dict().items():
        normalmodel_updates_dict[name] = torch.zeros_like(data)
        normalmodel_updates_dict[name] = (data - last_local_model[name])

    poison_optimizer = torch.optim.SGD(mmodel.parameters(), lr=poison_lr,
                                       momentum=params['momentum'],
                                       weight_decay=params['decay'])


    scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                     milestones=[0.2 * internal_epoch_num,
                                                                 0.8 * internal_epoch_num], gamma=0.1)
    temp_local_epoch = (epoch - 1) * internal_epoch_num

    for internal_epoch in range(1, internal_epoch_num + 1):
        temp_local_epoch += 1
        data_iterator = train_data[model_idx]
        data_iterator = torch.utils.data.DataLoader(data_iterator, batch_size=256, shuffle=True, num_workers=1)
        poison_data_count = 0
        total_loss = 0.
        correct = 0
        dataset_size = 0
        poisonupdate_dict = dict()
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = cerp_get_poison_batch(params, batch, tuned_trigger, device,
                                                                adversarial_index=model_idx,
                                                                evaluation=False)
            poison_optimizer.zero_grad()
            dataset_size += len(data)
            poison_data_count += poison_num
            output = mmodel(data)
            class_loss = nn.functional.cross_entropy(output, targets)
            loss = class_loss

            malDistance_Loss = cerp_model_dist_norm_var(mmodel, normal_params_variables, device)
            sum_cs = 0
            otheradnum = 0

            if poisonupdate_dict:
                for otherAd in params['adversary_list']:
                    poisonupdate_dict[otherAd] = copy.deepcopy(mmodel.state_dict())
                    if otherAd == model_idx:
                        continue
                    else:
                        if otherAd in poisonupdate_dict.keys():
                            otheradnum += 1
                            otherAd_variables = dict()
                            for name, data in poisonupdate_dict[otherAd].items():
                                otherAd_variables[name] = poisonupdate_dict[otherAd][
                                    name].clone().detach().requires_grad_(False)

                            sum_cs += model_cosine_similarity(mmodel, otherAd_variables)
            loss = class_loss + params['alpha_loss'] * malDistance_Loss + \
                   params['beta_loss'] * sum_cs
            poisonloss_dict[model_idx] = loss
            loss.backward()

            for i, (name, param) in enumerate(mmodel.named_parameters()):
                if param.requires_grad:
                    if internal_epoch == 1 and batch_id == 0:
                        client_grad.append(param.grad.clone())
                    else:
                        client_grad[i] += param.grad.clone()

            poison_optimizer.step()
            total_loss += loss.data
            pred = output.data.max(1)[1]
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

        if step_lr:
            scheduler.step()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        print(
            '___PoisonTrain , epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
            'Accuracy: {}/{} ({:.4f}%), train_poison_data_count: {}'.format(epoch,
                                                                            model_idx,
                                                                            internal_epoch,
                                                                            total_l, correct, dataset_size,
                                                                            acc, poison_data_count))
    
        

    return tuned_trigger, mmodel.state_dict()
    
def cerp_Mytest_poison(params, test_data, epoch, mmodel, noise_trigger, device, is_poison=False, visualize=False, agent_name_key=""):
    mmodel.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0
    test_data_poison, test_targetlabel_data = poison_test_dataset(params, test_data)
    data_iterator = test_data_poison
    for batch_id, batch in enumerate(data_iterator):
        data, targets, poison_num = cerp_get_poison_batch(params, batch, noise_trigger,  device,adversarial_index=-1,
                                                            evaluation=True)
     
        poison_data_count += poison_num
        dataset_size += len(data)
        output = mmodel(data)
        total_loss += nn.functional.cross_entropy(output, targets, reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count != 0 else 0
    total_l = total_loss / poison_data_count if poison_data_count != 0 else 0

    print('___Test poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     'Accuracy: {}/{} ({:.4f}%)'.format(is_poison, epoch,
                                                        total_l, correct, poison_data_count,
                                                        acc))

    mmodel.train()
    return acc

###----------------------- --------cerp- ------------------------------------###
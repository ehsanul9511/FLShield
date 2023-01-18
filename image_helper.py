from collections import defaultdict
import shutil
from typing import OrderedDict
import matplotlib.pyplot as plt
from prometheus_client import Counter

import torch
import torch.utils.data

from helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np

from models.resnet_cifar import ResNet18
from models.MnistNet import MnistNet
from models.resnet_tinyimagenet import resnet18
from models.resnet_celebA import Resnet18
logger = logging.getLogger("logger")
import config
from config import device
from attack_of_the_tails.utils import load_poisoned_dataset

import copy
import cv2

import yaml

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import datetime
import json

from tqdm import tqdm
from collections import Counter


class ImageHelper(Helper):

    def create_model(self):
        local_model=None
        target_model=None
        if self.params['type']==config.TYPE_CIFAR:
            local_model = ResNet18(name='Local',
                                   created_time=self.params['current_time'])
            target_model = ResNet18(name='Target',
                                   created_time=self.params['current_time'])

        elif self.params['type'] in [config.TYPE_MNIST, config.TYPE_FMNIST, config.TYPE_EMNIST]:
            local_model = MnistNet(name='Local',
                                   created_time=self.params['current_time'])
            target_model = MnistNet(name='Target',
                                    created_time=self.params['current_time'])

        elif self.params['type']==config.TYPE_TINYIMAGENET:

            local_model= resnet18(name='Local',
                                   created_time=self.params['current_time'])
            target_model = resnet18(name='Target',
                                    created_time=self.params['current_time'])

        elif self.params['type']==config.TYPE_CELEBA:

            local_model= Resnet18(name='Local',
                                   created_time=self.params['current_time'])
            target_model = Resnet18(name='Target',
                                    created_time=self.params['current_time'])

        local_model=local_model.to(device)
        target_model=target_model.to(device)
        if self.params['resumed_model']:
            if torch.cuda.is_available() :
                # loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}")
                loaded_params = torch.load(f"{self.params['resumed_model_name']}")
            else:
                # loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}",map_location='cpu')
                loaded_params = torch.load(f"{self.params['resumed_model_name']}",map_location='cpu')
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']+1
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model

    def new_model(self):
        if self.params['type']==config.TYPE_CIFAR:
            new_model = ResNet18(name='Dummy',
                                   created_time=self.params['current_time'])

        elif self.params['type'] in [config.TYPE_MNIST, config.TYPE_FMNIST, config.TYPE_EMNIST]:
            new_model = MnistNet(name='Dummy',
                                    created_time=self.params['current_time'])

        elif self.params['type']==config.TYPE_TINYIMAGENET:
            new_model = resnet18(name='Dummy',
                                    created_time=self.params['current_time'])

        new_model=new_model.to(device)
        return new_model        

    def build_classes_dict(self):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):  # for cifar: 50000; for tinyimagenet: 100000
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        return cifar_classes

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = self.classes_dict
        class_size = len(cifar_classes[0]) #for cifar: 5000
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())  # for cifar: 10

        image_nums = []
        for n in range(no_classes):
            image_num = []
            random.seed(42+n)
            random.shuffle(cifar_classes[n])
            np.random.seed(42+n)
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                if self.params['aggregation_methods'] == config.AGGR_FLTRUST and user==0:
                    no_imgs = 10
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                image_num.append(len(sampled_list))
                if self.params['aggregation_methods'] == config.AGGR_FLTRUST and user in [0, no_participants-1]:
                    per_participant_list[no_participants-1-user].extend(sampled_list)
                else:
                    per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
            image_nums.append(image_num)
        # self.draw_dirichlet_plot(no_classes,no_participants,image_nums,alpha)
        logger.info(f"Per participant list length is {[len(x) for x in per_participant_list.values()]}")
        return per_participant_list

    def draw_dirichlet_plot(self,no_classes,no_participants,image_nums,alpha):
        fig= plt.figure(figsize=(10, 5))
        s = np.empty([no_classes, no_participants])
        for i in range(0, len(image_nums)):
            for j in range(0, len(image_nums[0])):
                s[i][j] = image_nums[i][j]
        s = s.transpose()
        left = 0
        y_labels = []
        category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, no_participants))
        for k in range(no_classes):
            y_labels.append('Label ' + str(k))
        vis_par=[0,10,20,30]
        for k in range(no_participants):
        # for k in vis_par:
            color = category_colors[k]
            plt.barh(y_labels, s[k], left=left, label=str(k), color=color)
            widths = s[k]
            xcenters = left + widths / 2
            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            # for y, (x, c) in enumerate(zip(xcenters, widths)):
            #     plt.text(x, y, str(int(c)), ha='center', va='center',
            #              color=text_color,fontsize='small')
            left += s[k]
        plt.legend(ncol=20,loc='lower left',  bbox_to_anchor=(0, 1),fontsize=4) #
        # plt.legend(ncol=len(vis_par), bbox_to_anchor=(0, 1),
        #            loc='lower left', fontsize='small')
        plt.xlabel("Number of Images", fontsize=16)
        # plt.ylabel("Label 0 ~ 199", fontsize=16)
        # plt.yticks([])
        fig.tight_layout(pad=0.1)
        # plt.ylabel("Label",fontsize='small')
        fig.savefig(self.folder_path+'/Num_Img_Dirichlet_Alpha{}.pdf'.format(alpha))

    def poison_test_dataset(self):
        logger.info('get poison test loader')
        # delete the test data with target label
        test_classes = {}
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label in test_classes:
                test_classes[label].append(ind)
            else:
                test_classes[label] = [ind]

        range_no_id = list(range(0, len(self.test_dataset)))
        for image_ind in test_classes[self.params['poison_label_swap']]:
            if image_ind in range_no_id:
                range_no_id.remove(image_ind)
        poison_label_inds = test_classes[self.params['poison_label_swap']]

        return torch.utils.data.DataLoader(self.test_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                               range_no_id)), \
               torch.utils.data.DataLoader(self.test_dataset,
                                            batch_size=self.params['batch_size'],
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                poison_label_inds))

    def calculate_lsr(self, dataset,id):
        y_labels = []
        for x, y in dataset:
            y_labels.append(y)
        dataset_dict = OrderedDict(Counter(y_labels))
        dataset_dict = OrderedDict(sorted(dataset_dict.items()))
        logger.info(f'id: {id}, dataset_dict: {dataset_dict}')


    def get_label_skew_ratios_v2(self, dataloader, id, num_of_classes=10):
        # get y labels
        # y_labels = dataset.targets
        # y_labels = np.array(y_labels)
        dataset_dict = OrderedDict({i: 0 for i in range(num_of_classes)})

        # count non-zero labels
        for _, y_labels in dataloader:
            y = y_labels.numpy()
            for i in range(num_of_classes):
                dataset_dict[i] += np.count_nonzero(y == i)

        dataset_classes = np.array(list(dataset_dict.values()))
        dataset_classes = dataset_classes/np.sum(dataset_classes)
        return dataset_classes


    def get_label_skew_ratios(self, dataset, id, num_of_classes=10):
        dataset_classes = {}
        # for ind, x in enumerate(dataset):
        #     _, label = x
        #     #if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
        #     #    continue
        #     if label in dataset_classes:
        #         dataset_classes[label] += 1
        #     else:
        #         dataset_classes[label] = 1
        # for key in dataset_classes.keys():
        #     # dataset_classes[key] = dataset_classes[key] 

        #     dataset_classes[key] = float("{:.2f}".format(dataset_classes[key]/len(dataset)))
        # if self.params['noniid']:
        y_labels = []
        for x, y in dataset:
            y_labels.append(y)
        # else:
        #     y_labels=[t.item() for t in dataset.targets]
        #     indices = self.indices_per_participant[id]
        #     y_labels = np.array(y_labels)
        #     y_labels = y_labels[indices]
        dataset_dict = OrderedDict(Counter(y_labels))
        dataset_dict = OrderedDict(sorted(dataset_dict.items()))
        for ky in range(num_of_classes):
            if ky not in dataset_dict.keys():
                dataset_dict[ky] = 0
        # for c in range(num_of_classes):
        #     dataset_classes.append(dataset_dict[c])
        # dataset_classes = np.array(dataset_classes)
        # print(dataset_classes)
        dataset_classes = np.array(list(dataset_dict.values()))
        dataset_classes = dataset_classes/np.sum(dataset_classes)
        return dataset_classes

    def assign_data(self, train_data, bias, num_labels=10, num_workers=100, server_pc=100, p=0.01, server_case2_cls=0, dataset="FashionMNIST", seed=1, flt_aggr=True):
        # assign data to the clients
        other_group_size = (1 - bias) / (num_labels - 1)
        worker_per_group = num_workers / num_labels

        #assign training data to each worker
        each_worker_data = [[] for _ in range(num_workers)]
        each_worker_label = [[] for _ in range(num_workers)]   
        server_data = []
        server_label = []

        if 'ablation_study' in self.params.keys() and 'fltrust_privacy' in self.params['ablation_study']:
            server_pc = 500

            if 'missing' in self.params['ablation_study']:
                server_case2_cls = self.source_class
                p = 0
        
        # compute the labels needed for each class
        real_dis = [1. / num_labels for _ in range(num_labels)]
        samp_dis = [0 for _ in range(num_labels)]
        num1 = int(server_pc * p)
        samp_dis[server_case2_cls] = num1
        average_num = (server_pc - num1) / (num_labels - 1)
        resid = average_num - np.floor(average_num)
        sum_res = 0.
        for other_num in range(num_labels - 1):
            if other_num == server_case2_cls:
                continue
            samp_dis[other_num] = int(average_num)
            sum_res += resid
            if sum_res >= 1.0:
                samp_dis[other_num] += 1
                sum_res -= 1
        samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[:num_labels - 1])

        logger.info('samp_dis: {}'.format(samp_dis))

        # privacy experiment only
        server_additional_label_0_samples_counter = 0    
        server_add_data=[]
        server_add_label=[]

        # randomly assign the data points based on the labels
        server_counter = [0 for _ in range(num_labels)]
        for iidx, (x, y) in enumerate(train_data):


            upper_bound = y * (1. - bias) / (num_labels - 1) + bias
            lower_bound = y * (1. - bias) / (num_labels - 1)

            upper_bound_offset = 0
            np.random.seed(42 + iidx)
            rd = np.random.random_sample()


            other_group_size = (1 - upper_bound - upper_bound_offset + lower_bound) / (num_labels - 1)

            if rd > upper_bound + upper_bound_offset:
                worker_group = int(np.floor((rd - upper_bound - upper_bound_offset) / other_group_size) + y + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            # experiment 2 only
            elif rd > upper_bound:
                continue
            else:
                worker_group = y

            if server_counter[int(y)] < samp_dis[int(y)] and flt_aggr:
                server_data.append(x)
                server_label.append(y)
                server_counter[int(y)] += 1
            else:
                np.random.seed(73 + iidx)
                rd = np.random.random_sample()
                selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
                each_worker_data[selected_worker].append(x)
                each_worker_label[selected_worker].append(y)

        return server_data, server_label, each_worker_data, each_worker_label, server_add_data, server_add_label

    def get_group_sizes(self, num_labels=10):
        if self.params['type'] in config.random_group_size_dict.keys():
            if 'save_data' in self.params.keys():
                which_data_dist = self.params['save_data']
            else:
                which_data_dist = random.sample(range(1,4), 1)[0]
            group_sizes = config.random_group_size_dict[self.params['type']][which_data_dist]
        else:
            group_sizes = [num_labels for _ in range(num_labels)]
        return group_sizes

    def assign_data_nonuniform(self, train_data, bias, num_labels=10, num_workers=100, server_pc=100, p=0.01, server_case2_cls=0, dataset="FashionMNIST", seed=1, flt_aggr=True):
        server_data, server_label, each_worker_data, each_worker_label, server_add_data, server_add_label = self.assign_data(train_data, bias, num_labels, num_workers//num_labels, server_pc, p, server_case2_cls, dataset, seed, flt_aggr)
        ewd = [[] for _ in range(num_workers)]
        ewl = [[] for _ in range(num_workers)]
        # group_sizes = [np.random.randint(5, 16) for i in range(9)]
        # group_sizes.append(num_workers-sum(group_sizes))
        # if group_sizes[-1] < 5 or group_sizes[-1] > 15:
        #     avg_last_2 = (group_sizes[-1] + group_sizes[-2])/2

        group_sizes = self.get_group_sizes()
            
        copylist = []
        for i, group_size in enumerate(group_sizes):
            for _ in range(group_size):
                copylist.append(i)

        for i in range(len(each_worker_data)):
            group_size = group_sizes[i]
            group_frac = 1./group_size
            label_data = np.array(each_worker_label[i])
            i_indices = np.where(label_data==i)[0].tolist()
            not_i_indices = np.where(label_data!=i)[0].tolist()
            split_map_for_i = []
            split_map_for_i.append(0)
            split_map_for_not_i = [0]
            for ii in range(1, group_size):
                np.random.seed(42 + i)
                split_ratio_for_i = np.random.normal(ii*group_frac, group_frac//num_labels)
                split_ratio_for_not_i = ii*group_frac*2 - split_ratio_for_i
                split_map_for_i.append(int(split_ratio_for_i*len(i_indices)))
                split_map_for_not_i.append(int(split_ratio_for_not_i*len(not_i_indices)))
            split_map_for_i.append(len(i_indices))
            split_map_for_not_i.append(len(not_i_indices))
            i_indices_list = [i_indices[split_map_for_i[ii]:split_map_for_i[ii+1]] for ii in range(group_size)]
            not_i_indices_list = [not_i_indices[split_map_for_not_i[ii]:split_map_for_not_i[ii+1]] for ii in range(group_size)]
            indice_map = [0]*len(each_worker_data[i])
            for ii in range(group_size):
                for iii in i_indices_list[ii]:
                    indice_map[iii] = ii 
                for iii in not_i_indices_list[ii]:
                    indice_map[iii] = ii 
            size_of_group = int(len(each_worker_data[i])//num_labels)
            stop_val = num_labels * size_of_group
            for idx in range(len(each_worker_data[i])):
                ewd[sum(group_sizes[:i]) + indice_map[idx]].append(each_worker_data[i][idx])
                ewl[sum(group_sizes[:i]) + indice_map[idx]].append(each_worker_label[i][idx])
        return server_data, server_label, ewd, ewl, server_add_data, server_add_label

    def load_saved_data(self):
        train_loaders=[]
        for i in range(self.params['number_of_total_participants']):
            train_loaders.append(torch.load(f'./saved_data/{self.params["type"]}/{self.params["load_data"]}/train_data_{i}.pt'))
        if self.params['aggregation_methods'] == config.AGGR_FLTRUST:
            train_loaders[-1] = torch.load(f'./saved_data/{self.params["type"]}/{self.params["load_data"]}/train_data_{self.params["number_of_total_participants"]}.pt')
        
        self.train_data = [(i, train_loader) for i, train_loader in enumerate(train_loaders)]
        self.test_data = torch.load(f'./saved_data/{self.params["type"]}/{self.params["load_data"]}/test_data.pt')
        # self.test_data_poison = torch.load(f'./saved_data/{self.params["type"]}/{self.params["load_data"]}/test_data_poison.pt')
        # self.test_targetlabel_data = torch.load(f'./saved_data/{self.params["type"]}/{self.params["load_data"]}/test_targetlabel_data.pt')
        self.test_dataset = self.test_data.dataset
        self.test_data_poison, self.test_targetlabel_data = self.poison_test_dataset()
        logger.info(f'Loaded data')

    def split_train_val_single(self, train_data, val_size, seed=1):
        np.random.seed(seed)
        train_data, val_data = torch.utils.data.random_split(train_data, [len(train_data)-val_size, val_size])
        return train_data, val_data

    def split_train_val(self, all_train_loaders, val_pcnt=0.3, seed=1):
        train_loaders = all_train_loaders
        val_loaders = []
        reused_val_loaders = []
        for i in range(self.params['number_of_total_participants']):
            # local data of each worker
            train_data = train_loaders[i][1].dataset
            train_size = int((1 - val_pcnt) * len(train_data))
            val_size = len(train_data) - train_size
            train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
            if self.params['attack_methods'] == config.ATTACK_AOTT and False:
                if i > self.params[f'number_of_adversary_{self.params["attack_methods"]}'] or True:
                    edge_data = self.clean_val_loader.dataset
                else:
                    edge_data = self.poison_trainloader.dataset
                samp_indices = np.random.choice(len(edge_data), int(self.params['edge_split']*len(val_data)), replace=False)
                samp_indices_2 = np.random.choice(len(val_data), int((1-self.params['edge_split'])*len(val_data)), replace=False)
                val_data = torch.utils.data.Subset(val_data, samp_indices_2)
                edge_data = torch.utils.data.Subset(edge_data, samp_indices)
                val_data = torch.utils.data.ConcatDataset([val_data, edge_data])
            reused_val_data = train_data + val_data
            train_loaders[i] = (i, torch.utils.data.DataLoader(train_data, batch_size=self.params['batch_size'], shuffle=True))
            val_loaders.append(torch.utils.data.DataLoader(val_data, batch_size=self.params['batch_size'], shuffle=True))
            reused_val_loaders.append(torch.utils.data.DataLoader(reused_val_data, batch_size=self.params['batch_size'], shuffle=True))
        return train_loaders, val_loaders, reused_val_loaders

    def load_data(self):
        logger.info('Loading data')
        if 'load_data' in self.params:
            self.load_saved_data()
        else:
            dataPath = './data'
            dataPath_emnist = '/dartfs-hpc/rc/home/9/f0059f9/OOD_Federated_Learning/data'
            dataPath_emnist = './data'
            num_labels = 10
            if self.params['type'] == config.TYPE_CIFAR:
                ### data load
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                ])

                self.train_dataset = datasets.CIFAR10(dataPath, train=True, download=True,
                                                transform=transform_train)

                self.test_dataset = datasets.CIFAR10(dataPath, train=False, transform=transform_test)

                if self.params['attack_methods'] == config.ATTACK_AOTT:
                    self.poison_trainloader, _, self.poison_testloader, _, self.clean_val_loader = load_poisoned_dataset(dataset = self.params['type'], fraction = 1, batch_size = self.params['batch_size'], test_batch_size = self.params['test_batch_size'], poison_type='southwest', attack_case='edge-case', edge_split = 0.5)

                    logger.info('poison train and test data from southwest loaded')

                elif self.params['attack_methods'] == config.ATTACK_SEMANTIC:
                    green_car_indices = config.green_car_indices
                    cifar10_whole_range = np.arange(self.train_dataset.data.shape[0])
                    semantic_dataset = []
                    semantic_dataset_correct = []
                    remaining_dataset = []
                    for ind, (data, target) in enumerate(self.train_dataset):
                        if ind in green_car_indices:
                            semantic_dataset.append((data, 2))
                            # semantic_dataset.append((data, target))
                            semantic_dataset_correct.append((data, target))
                        else:
                            remaining_dataset.append((data, target))
                    
                    self.semantic_dataloader = torch.utils.data.DataLoader(semantic_dataset, batch_size=self.params['batch_size'], shuffle=True)
                    self.semantic_dataloader_correct = torch.utils.data.DataLoader(semantic_dataset_correct, batch_size=self.params['batch_size'], shuffle=True)
                    self.train_dataset = remaining_dataset


                    # remaining_indices = [i for i in cifar10_whole_range if i not in green_car_indices]
                    # self.semantic_dataset = torch.utils.data.Subset(self.train_dataset, green_car_indices)
                    # sampled_targets_array_train = 2 * np.ones((len(self.semantic_dataset),), dtype =int) # green car -> label as bird
                    # self.semantic_dataset.targets = torch.from_numpy(sampled_targets_array_train)
                    # self.train_dataset = torch.utils.data.Subset(self.train_dataset, remaining_indices)


            elif self.params['type'] == config.TYPE_MNIST:

                self.train_dataset = datasets.MNIST('./data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.1307,), (0.3081,))
                                ]))
                self.test_dataset = datasets.MNIST('./data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize((0.1307,), (0.3081,))
                    ]))
            elif self.params['type'] == config.TYPE_FMNIST:
                self.train_dataset = datasets.FashionMNIST('./data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.1307,), (0.3081,))
                                ]))
                self.test_dataset = datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize((0.1307,), (0.3081,))
                    ]))
            elif self.params['type'] == config.TYPE_EMNIST:
                self.train_dataset = datasets.EMNIST(dataPath_emnist, split='digits', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
                self.test_dataset = datasets.EMNIST(dataPath_emnist, split='digits', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

                if self.params['attack_methods'] == config.ATTACK_AOTT:
                    self.poison_trainloader, _, self.poison_testloader, _, _ = load_poisoned_dataset(dataset = self.params['type'], fraction = 1, batch_size = self.params['batch_size'], test_batch_size = self.params['test_batch_size'], poison_type='ardis')

                    logger.info('poison train and test data from ARDIS loaded')
            elif self.params['type'] == config.TYPE_TINYIMAGENET:

                _data_transforms = {
                    'train': transforms.Compose([
                        # transforms.Resize(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]),
                    'val': transforms.Compose([
                        # transforms.Resize(224),
                        transforms.ToTensor(),
                    ]),
                }
                _data_dir = './data/tiny-imagenet-200/'
                self.train_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'train'),
                                                        _data_transforms['train'])
                self.test_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'val'),
                                                    _data_transforms['val'])
                logger.info('reading data done')
            elif self.params['type'] == config.TYPE_CELEBA:
                num_labels = 5
                _data_transforms = {
                    'train': transforms.Compose([
                        # transforms.Resize(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]),
                    'val': transforms.Compose([
                        # transforms.Resize(224),
                        transforms.ToTensor(),
                    ]),
                }
                _data_dir = './data/celebA/'
                self.train_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'train'),
                                                        _data_transforms['train'])
                self.test_dataset = datasets.ImageFolder(os.path.join(_data_dir, 'val'),
                                                    _data_transforms['val'])
                logger.info('reading data done')
                logger.info(f'train data size: {len(self.train_dataset)}')

            self.lsrs = []

            if self.params['noniid']:
                sd, sl, ewd, ewl, sad, sal = self.assign_data_nonuniform(self.train_dataset, bias=self.params['bias'], p=0.1, flt_aggr=1, num_workers=self.params['number_of_total_participants'], num_labels=num_labels)
                if self.params['aggregation_methods'] == config.AGGR_FLTRUST:
                    ewd.append(sd)
                    ewl.append(sl)

                train_loaders = []
                for id_worker in range(len(ewd)):
                    dataset_per_worker=[]
                    for idx in range(len(ewd[id_worker])):
                        dataset_per_worker.append((ewd[id_worker][idx], ewl[id_worker][idx]))
                    if len(dataset_per_worker) != 0:
                        train_loader = torch.utils.data.DataLoader(dataset_per_worker, batch_size=self.params['batch_size'], shuffle=True)
                        train_loaders.append((id_worker, train_loader))
                train_loaders[-2] = train_loaders[-1] 
            elif self.params['sampling_dirichlet']:
                ## sample indices for participants using Dirichlet distribution
                preload_data = False
                if preload_data:
                    train_loaders=[]
                    for i in range(self.params['number_of_total_participants']):
                        train_loaders.append((i,torch.load(f'./saved_data/{self.params["type"]}/{self.params["save_data"]}/train_data_{i}.pt')))
                else:
                    self.classes_dict = self.build_classes_dict()
                    indices_per_participant = self.sample_dirichlet_train_data(
                        self.params['number_of_total_participants'], #100
                        alpha=self.params['dirichlet_alpha'])
                    logger.info(f'indices_per_participant: {[len(indices_per_participant[i]) for i in range(len(indices_per_participant))]}')
                    self.indices_per_participant = indices_per_participant
                    # train_loaders = [(pos, self.get_train_alt(indices)) for pos, indices in
                    #                 indices_per_participant.items()]
                    ewd, ewl = self.get_train_alt([indices_per_participant[i] for i in range(len(indices_per_participant))])
                    train_loaders = []
                    for id_worker in range(len(ewd)):
                        dataset_per_worker=[]
                        for idx in range(len(ewd[id_worker])):
                            dataset_per_worker.append((ewd[id_worker][idx], ewl[id_worker][idx]))
                        if len(dataset_per_worker) != 0:
                            train_loader = torch.utils.data.DataLoader(dataset_per_worker, batch_size=self.params['batch_size'], shuffle=True)
                            train_loaders.append((id_worker, train_loader))
            else:
                ## sample indices for participants that are equally
                logger.info('sampling indices for participants that are equally')
                all_range = list(range(len(self.train_dataset)))
                random.seed(42)
                random.shuffle(all_range)
                train_loaders = [(pos, self.get_train_old(all_range, pos))
                                for pos in tqdm(range(self.params['number_of_total_participants']))]

                self.lsrs = [[1/num_labels for _ in range(num_labels)] for _ in range(self.params['number_of_total_participants'])]

            logger.info('train loaders done')
            self.train_data = train_loaders

            # split train_data into validation data
            # if self.params['validation']:
            self.train_data, self.val_data, self.reused_val_data = self.split_train_val(self.train_data, val_pcnt=0.3)

            self.test_data = self.get_test()
            self.test_data_poison ,self.test_targetlabel_data = self.poison_test_dataset()

            if 'save_data' in self.params.keys() and False:
                if not os.path.isdir(f'./saved_data/{self.params["type"]}'):
                    os.mkdir(f'./saved_data/{self.params["type"]}')
                if os.path.isdir(f'./saved_data/{self.params["type"]}/{self.params["save_data"]}'):
                    shutil.rmtree(f'./saved_data/{self.params["type"]}/{self.params["save_data"]}')
                os.mkdir(f'./saved_data/{self.params["type"]}/{self.params["save_data"]}')
                for i, td in self.train_data:
                    torch.save(td, f'./saved_data/{self.params["type"]}/{self.params["save_data"]}/train_data_{i}.pt')

                torch.save(self.test_data, f'./saved_data/{self.params["type"]}/{self.params["save_data"]}/test_data.pt')
                torch.save(self.test_data_poison, f'./saved_data/{self.params["type"]}/{self.params["save_data"]}/test_data_poison.pt')
                torch.save(self.test_targetlabel_data, f'./saved_data/{self.params["type"]}/{self.params["save_data"]}/test_targetlabel_data.pt')
                logger.info('saving data done')

            # self.classes_dict = self.build_classes_dict()
            # logger.info('build_classes_dict done')
        if self.params['attack_methods'] in [config.ATTACK_TLF, config.ATTACK_SIA]:
            target_class_test_data=[]
            for _, (x, y) in enumerate(self.test_data.dataset):
                if y==self.source_class:
                    target_class_test_data.append((x, y))
            self.target_class_test_loader = torch.utils.data.DataLoader(target_class_test_data, batch_size=self.params['test_batch_size'], shuffle=True)

        # if self.params['noniid'] or self.params['sampling_dirichlet']:
        # if self.params['noniid']:
        # self.lsrs = []

        if len(self.lsrs) == 0:
            for id in tqdm(range(len(self.train_data))):
                (_, train_loader) = self.train_data[id]
                lsr = self.get_label_skew_ratios_v2(train_loader, id, num_of_classes=num_labels)
                self.lsrs.append(lsr)

        # logger.info(f'lsrs ready: {self.lsrs}')


        if self.params['is_random_namelist'] == False:
            self.participants_list = self.params['participants_namelist']
        else:
            self.participants_list = list(range(self.params['number_of_total_participants']))
        # random.shuffle(self.participants_list)

        self.variable_poison_rates = [4, 8, 16, 24, 32] * 6

        self.poison_epochs_by_adversary = {}
        # if self.params['random_adversary_for_label_flip']:
        if self.params['is_random_adversary']:
            # if self.params['attack_methods'] == config.ATTACK_TLF:
            #     if 'num_of_attackers_in_target_group' in self.params.keys():
            #         self.num_of_attackers_in_target_group = self.params['num_of_attackers_in_target_group']
            #     else:
            #         self.num_of_attackers_in_target_group = 4
            if self.params['noniid'] or True:
                random.seed(42)
                self.adversarial_namelist = random.sample(self.participants_list, self.params[f'number_of_adversary_{self.params["attack_methods"]}'])
            else:
                eligible_list = [name for name in range(self.params['number_of_total_participants']) if self.lsrs[name][self.source_class] > 0.07]
                self.adversarial_namelist = random.sample(eligible_list, min(self.params[f'number_of_adversary_{self.params["attack_methods"]}'], len(eligible_list)))
        else:
            self.adversarial_namelist = self.params['adversary_list']
        for idx, id in enumerate(self.adversarial_namelist):
            if self.params['attack_methods'] in [config.ATTACK_TLF, config.ATTACK_SIA]:
                self.poison_epochs_by_adversary[idx] = list(np.arange(1, self.params['epochs']+1))
            else:
                mod_idx = idx%4
                self.poison_epochs_by_adversary[idx] = self.params[f'{mod_idx}_poison_epochs'][10:]

        # self.adversarial_namelist = [name for name in self.adversarial_namelist if self.lsrs[name][self.source_class] > 0]

        self.benign_namelist =list(set(self.participants_list) - set(self.adversarial_namelist))

        if 'ablation_study' in self.params.keys() and 'with_lsr' in self.params['ablation_study']:
            for idx, id in enumerate(self.adversarial_namelist):
                self.lsrs[id] = self.lsrs[target_group_indices[0]]
                # self.lsrs[id] = self.lsrs[0]

        logger.info(f'adversarial_namelist: {self.adversarial_namelist}')
        logger.info(f'benign_namelist: {self.benign_namelist}')

    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices),pin_memory=True)
        return train_loader

    # def get_train_alt(self, indices):
    #     # local_train_data = []
    #     # local_train_labels = []
    #     local_dataset = []
    #     for idx, (x, y) in enumerate(self.train_dataset):
    #         if idx in indices:
    #             # local_train_data.append(x)
    #             # local_train_labels.append(y)
    #             local_dataset.append((x, y))

    #     train_loader = torch.utils.data.DataLoader(local_dataset, batch_size=self.params['batch_size'], shuffle=True)
    #     return train_loader

    def get_train_alt(self, indices_per_client):
        ewd = []
        ewl = []
        for _ in range(self.params['number_of_total_participants']):
            ewd.append([])
            ewl.append([])

        for idx, (x, y) in enumerate(tqdm(self.train_dataset)):
            for client_idx, indices in enumerate(indices_per_client):
                # logger.info(f'client_idx: {client_idx}')
                # logger.info(f'indices: {indices}')
                if idx in indices:
                    ewd[client_idx].append(x)
                    ewl[client_idx].append(y)

        # train_loaders = []
        # for id_worker in range(len(ewd)):
        #     dataset_per_worker=[]
        #     for idx in range(len(ewd[id_worker])):
        #         dataset_per_worker.append((ewd[id_worker][idx], ewl[id_worker][idx]))
        #     if len(dataset_per_worker) != 0:
        #         train_loader = torch.utils.data.DataLoader(dataset_per_worker, batch_size=self.params['batch_size'], shuffle=True)
        #         train_loaders.append((id_worker, train_loader))

        return ewd, ewl

        

    def get_train_old(self, all_range, model_no):
        """
        This method equally splits the dataset.
        :param params:
        :param all_range:
        :param model_no:
        :return:
        """

        data_len = int(len(self.train_dataset) / self.params['number_of_total_participants'])
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        # train_subset = []
        # for idx in sub_indices:
        #     train_subset.append(self.train_dataset[idx])
        train_subset = torch.utils.data.Subset(self.train_dataset, sub_indices)
        # train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                        #    batch_size=self.params['batch_size'],
                                        #    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                        #        sub_indices))
        train_loader = torch.utils.data.DataLoader(train_subset,
                                                batch_size=self.params['batch_size'],
                                                shuffle=True)
        return train_loader

    def get_test(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=True)
        return test_loader


    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.to(device)
        target = target.to(device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

    def get_poison_batch_for_label_flip(self, bptt, target_class=-1):

        images, targets = bptt

        poison_count= 0
        new_images=images
        new_targets=targets

        if target_class==-1:
            target_class = self.source_class

        for index in range(0, len(images)):
            new_targets[index] = 9 - targets[index]
            new_images[index] = images[index]

        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()
        return new_images,new_targets,poison_count        

    def get_poison_batch_for_targeted_label_flip(self, bptt, target_class=-1):

        images, targets = bptt

        poison_count= 0
        new_images=images
        new_targets=targets

        if target_class==-1:
            target_class = self.source_class

        for index in range(0, len(images)):
            if targets[index]==target_class: # poison all data when testing
                new_targets[index] = self.target_class
                new_images[index] = images[index]
                poison_count+=1
            else:
                new_images[index] = images[index]
                new_targets[index]= targets[index]
            # new_targets[index] = self.params['targeted_label_flip_class']
            # new_images[index] = images[index]
            poison_count+=1

        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()
        return new_images,new_targets,poison_count    

    def get_poison_batch(self, bptt,adversarial_index=-1, evaluation=False, special_attack=False):
        if 'special_attack' in self.params and self.params['special_attack']:
            special_attack = self.params['special_attack']

        images, targets = bptt

        poison_count= 0
        new_images=images
        new_targets=targets

        adv_lsr = self.lsrs[self.adversarial_namelist[adversarial_index]]
        adv_lsr = np.array(adv_lsr)
        maj_ind = np.argmax(adv_lsr)
        major_ind_list = [maj_ind]
        if special_attack and False:
            poisoning_per_batch = self.variable_poison_rates[adversarial_index]
        else:
            poisoning_per_batch = self.params['poisoning_per_batch']

        while np.sum(adv_lsr[major_ind_list]) < poisoning_per_batch/self.params['batch_size']:
            major_ind_list.append((major_ind_list[-1]+1)%(len(adv_lsr)-1))
            # logger.info(f'poisoning_per_batch: {np.sum(adv_lsr[major_ind_list])}')

        for index in range(0, len(images)):
            if evaluation: # poison all data when testing
                new_targets[index] = self.params['poison_label_swap']
                new_images[index] = self.add_pixel_pattern(images[index],adversarial_index)
                poison_count+=1

            else: # poison part of data when training
                if not special_attack:
                    if index < poisoning_per_batch:
                        new_targets[index] = self.params['poison_label_swap']
                        new_images[index] = self.add_pixel_pattern(images[index],adversarial_index)
                        poison_count += 1
                    else:
                        new_images[index] = images[index]
                        new_targets[index]= targets[index]
                else:
                    if targets[index] in major_ind_list and poison_count<poisoning_per_batch:
                        # logger.info(f'poisoning index {index} with target {targets[index]}')
                        new_targets[index] = self.params['poison_label_swap']
                        new_images[index] = self.add_pixel_pattern(images[index],-1)
                        poison_count+=1
                    else:
                        new_images[index] = images[index]
                        new_targets[index]= targets[index]

        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images,new_targets,poison_count

    def add_pixel_pattern(self,ori_image,adversarial_index):
        image = copy.deepcopy(ori_image)
        poison_patterns= []
        if adversarial_index==-1:
            for i in range(0,self.params['trigger_num']):
                poison_patterns = poison_patterns+ self.params[str(i) + '_poison_pattern']
        else :
            poison_patterns = self.params[str(adversarial_index%4) + '_poison_pattern']
        if self.params['type'] == config.TYPE_CIFAR or self.params['type'] == config.TYPE_TINYIMAGENET or self.params['type'] == config.TYPE_CELEBA:
            for i in range(0,len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = 1
                image[1][pos[0]][pos[1]] = 1
                image[2][pos[0]][pos[1]] = 1


        elif self.params['type'] in [config.TYPE_MNIST, config.TYPE_FMNIST, config.TYPE_EMNIST]:

            for i in range(0, len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = 1

        return image

# if __name__ == '__main__':
#     np.random.seed(1)
#     with open(f'./utils/cifar_params.yaml', 'r') as f:
#         params_loaded = yaml.load(f)
#     current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
#     helper = ImageHelper(current_time=current_time, params=params_loaded,
#                         name=params_loaded.get('name', 'mnist'))
#     helper.load_data()

#     pars= list(range(100))
#     # show the data distribution among all participants.
#     count_all= 0
#     for par in pars:
#         cifar_class_count = dict()
#         for i in range(10):
#             cifar_class_count[i] = 0
#         count=0
#         _, data_iterator = helper.train_data[par]
#         for batch_id, batch in enumerate(data_iterator):
#             data, targets= batch
#             for t in targets:
#                 cifar_class_count[t.item()]+=1
#             count += len(targets)
#         count_all+=count
#         print(par, cifar_class_count,count,max(zip(cifar_class_count.values(), cifar_class_count.keys())))

#     print('avg', count_all*1.0/100)


if __name__ == '__main__':
    with open(f'./utils/celebA_params_temp.yaml', 'r') as f:
        params_loaded = yaml.load(f)

    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    helper = ImageHelper(current_time=current_time, params=params_loaded,
                        name=params_loaded.get('name', 'mnist'))
    helper.load_data()

    

    
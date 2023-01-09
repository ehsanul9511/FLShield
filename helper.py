from shutil import copyfile

import math
import shutil
from weakref import ref
import torch
import torch.nn as nn

from torch.autograd import Variable
import logging
import sklearn.metrics.pairwise as smp
from torch.nn.functional import log_softmax
import torch.nn.functional as F
import time

logger = logging.getLogger("logger")
import os
import json
import numpy as np
import config
import copy
import utils.csv_record

from utils.utils import get_hash_from_param_file
# import train
# import test

from florida_utils.validation_processing import ValidationProcessor
from florida_utils.cluster_grads import cluster_grads as cluster_function
from florida_utils.validation_test import validation_test
from florida_utils.impute_validation import impute_validation

from torch.utils.data import SubsetRandomSampler
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, cosine_similarity
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import stats, rankdata
from tqdm import tqdm
from termcolor import colored
from random import shuffle, randint
from tabulate import tabulate
from collections import defaultdict

import hdbscan
from binarytree import Node
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx

from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot as plt

import pickle


class Helper:
    def __init__(self, current_time, params, name):
        self.current_time = current_time
        self.target_model = None
        self.local_model = None

        self.train_data = None
        self.test_data = None
        self.poisoned_data = None
        self.test_data_poison = None

        self.params = defaultdict(lambda: None, params)
        self.name = name
        self.best_loss = math.inf

        if self.params['attack_methods'] in [config.ATTACK_TLF, config.ATTACK_DBA]:
            if self.params['type'] in config.target_class_dict.keys():
                self.source_class = config.target_class_dict[self.params['type']][self.params['tlf_label']][0]
                self.target_class = config.target_class_dict[self.params['type']][self.params['tlf_label']][1]
            else:
                self.source_class = int(self.params['tlf_label'])
                self.target_class = 9 - self.source_class
        else:
            self.source_class = int(self.params['poison_label_swap'])
        # if self.params['attack_methods'] == config.ATTACK_TLF:
        #     self.num_of_adv = self.params[f'number_of_adversary_{config.ATTACK_TLF}']
        # else:
        #     self.num_of_adv = self.params[f'number_of_adversary_{config.ATTACK_DBA}']
        self.num_of_adv = self.params[f'number_of_adversary_{self.params["attack_methods"]}']

        # self.folder_path = f'saved_models/model_{self.name}_{current_time}_no_models_{self.params["no_models"]}'

        hash_value = get_hash_from_param_file(self.params)
        self.folder_path = f'saved_models/{current_time}_{hash_value}'

        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        else:
            shutil.rmtree(self.folder_path)
            os.makedirs(self.folder_path)
        logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/log.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)

        # logger.info(f'source_class: {self.source_class}, target_class: {self.target_class}')
        logger.info(f'current path: {self.folder_path}')
        if not self.params.get('environment_name', False):
            self.params['environment_name'] = self.name

        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path
        self.fg= FoolsGold(use_memory=self.params['fg_use_memory'])

        if self.params['aggregation_methods'] == config.AGGR_AFA:
            self.good_count = [0 for _ in range(self.params['number_of_total_participants'])]
            self.bad_count = [0 for _ in range(self.params['number_of_total_participants'])]
            self.prob_good_model = [0. for _ in range(self.params['number_of_total_participants'])]
        elif self.params['aggregation_methods'] == config.AGGR_OURS and 'adaptive_grad_attack' in self.params.keys() and self.params['adaptive_grad_attack']:
            self.prev_epoch_val_model_params = []

    def color_print_wv(self, wv, names):
        wv_print_str= '['
        for idx, w in enumerate(wv):
            wv_print_str += ' '
            if names[idx] in self.adversarial_namelist:
                wv_print_str += colored(str(w), 'blue')
            else:
                wv_print_str += str(w)
        wv_print_str += ']'
        print(f'wv: {wv_print_str}')

    def get_param_val(self, key):
        if key in self.params.keys():
            return self.params[key]
        else:
            return None

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params['save_model']:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    @staticmethod
    def model_global_norm(model):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_dist_norm(model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_max_values(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer.data - target_params[name].data)))
        return squared_sum

    @staticmethod
    def model_max_values_var(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer - target_params[name])))
        return sum(squared_sum)

    @staticmethod
    def get_one_vec(model, variable=False):
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            size += layer.view(-1).shape[0]
        if variable:
            sum_var = Variable(torch.cuda.FloatTensor(size).fill_(0))
        else:
            sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            if variable:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer).view(-1)
            else:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer.data).view(-1)
            size += layer.view(-1).shape[0]

        return sum_var

    @staticmethod
    def model_dist_norm_var(model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.FloatTensor(size).fill_(0)
        sum_var= sum_var.to(config.device)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
                    layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def cos_sim_loss(self, model, target_vec):
        model_vec = self.get_one_vec(model, variable=True)
        target_var = Variable(target_vec, requires_grad=False)
        # target_vec.requires_grad = False
        cs_sim = torch.nn.functional.cosine_similarity(
            self.params['scale_weights'] * (model_vec - target_var) + target_var, target_var, dim=0)
        # cs_sim = cs_loss(model_vec, target_vec)
        logger.info("los")
        logger.info(cs_sim.data[0])
        logger.info(torch.norm(model_vec - target_var).data[0])
        loss = 1 - cs_sim

        return 1e3 * loss

    def model_cosine_similarity(self, model, target_params_variables,
                                model_id='attacker'):

        cs_list = list()
        cs_loss = torch.nn.CosineSimilarity(dim=0)
        for name, data in model.named_parameters():
            if name == 'decoder.weight':
                continue

            model_update = 100 * (data.view(-1) - target_params_variables[name].view(-1)) + target_params_variables[
                name].view(-1)

            cs = F.cosine_similarity(model_update,
                                     target_params_variables[name].view(-1), dim=0)
            # logger.info(torch.equal(layer.view(-1),
            #                          target_params_variables[name].view(-1)))
            # logger.info(name)
            # logger.info(cs.data[0])
            # logger.info(torch.norm(model_update).data[0])
            # logger.info(torch.norm(fake_weights[name]))
            cs_list.append(cs)
        cos_los_submit = 1 * (1 - sum(cs_list) / len(cs_list))
        logger.info(model_id)
        logger.info((sum(cs_list) / len(cs_list)).data[0])
        return 1e3 * sum(cos_los_submit)

    def accum_similarity(self, last_acc, new_acc):

        cs_list = list()

        cs_loss = torch.nn.CosineSimilarity(dim=0)
        # logger.info('new run')
        for name, layer in last_acc.items():
            cs = cs_loss(Variable(last_acc[name], requires_grad=False).view(-1),
                         Variable(new_acc[name], requires_grad=False).view(-1))
            # logger.info(torch.equal(layer.view(-1),
            #                          target_params_variables[name].view(-1)))
            # logger.info(name)
            # logger.info(cs.data[0])
            # logger.info(torch.norm(model_update).data[0])
            # logger.info(torch.norm(fake_weights[name]))
            cs_list.append(cs)
        cos_los_submit = 1 * (1 - sum(cs_list) / len(cs_list))
        # logger.info("AAAAAAAA")
        # logger.info((sum(cs_list)/len(cs_list)).data[0])
        return sum(cos_los_submit)

    @staticmethod
    def dp_noise(param, sigma):
        if torch.cuda.is_available():
            noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
        else:
            noised_layer = torch.FloatTensor(param.shape).normal_(mean=0, std=sigma)

        return noised_layer

    def accumulate_weight(self, weight_accumulator, epochs_submit_update_dict, state_keys,num_samples_dict):
        """
         return Args:
             updates: dict of (num_samples, update), where num_samples is the
                 number of training samples corresponding to the update, and update
                 is a list of variable weights
         """
        # if self.params['aggregation_methods'] == config.AGGR_FLAME or self.params['aggregation_methods'] == config.AGGR_FLTRUST:
        if self.params['aggregation_methods'] in [config.AGGR_FLAME, config.AGGR_FLTRUST, config.AGGR_OURS, config.AGGR_AFA, config.AGGR_MEAN]:
            updates = dict()
            for i in range(0, len(state_keys)):
                local_model_gradients = epochs_submit_update_dict[state_keys[i]][0][0] # agg 1 interval
                num_samples = num_samples_dict[state_keys[i]]
                try:
                    local_model_update_list = epochs_submit_update_dict[state_keys[i]][1]
                except:
                    logger.info("epochs_submit_update_dict[state_keys[i]][1] is None")
                    logger.info(f'length of epochs_submit_update_dict is {len(epochs_submit_update_dict.keys())}')
                    logger.info(f'state_keys[i] is {state_keys[i]}')
                    logger.info(f'length of epochs_submit_update_dict[state_keys[i]] is {len(epochs_submit_update_dict[state_keys[i]])}')
                update= dict()

                for name, data in local_model_update_list[0].items():
                    if not torch.is_tensor(data):
                        # logger.info(f'name: {name}, data: {data}')
                        data = torch.FloatTensor(data)
                    update[name] = torch.zeros_like(data)

                for j in range(0, len(local_model_update_list)):
                    local_model_update_dict= local_model_update_list[j]
                    for name, data in local_model_update_dict.items():
                        if not torch.is_tensor(data):
                            # logger.info(f'name: {name}, data: {data}')
                            data = torch.FloatTensor(data)
                        # weight_accumulator[name].add_(local_model_update_dict[name])
                        # update[name].add_(local_model_update_dict[name])
                        weight_accumulator[name].add_(data)
                        update[name].add_(data)
                        detached_data= data.cpu().detach().numpy()
                        # print(detached_data.shape)
                        detached_data=detached_data.tolist()
                        # print(detached_data)
                        local_model_update_dict[name]=detached_data # from gpu to cpu
                updates[state_keys[i]] = (num_samples, copy.deepcopy(local_model_gradients), update)
            return weight_accumulator, updates

        else:
            updates = dict()
            for i in range(0, len(state_keys)):
                local_model_update_list = epochs_submit_update_dict[state_keys[i]][1]
                update= dict()
                num_samples=num_samples_dict[state_keys[i]]

                for name, data in local_model_update_list[0].items():
                    if not torch.is_tensor(data):
                        # logger.info(f'name: {name}, data: {data}')
                        data = torch.FloatTensor(data)
                    update[name] = torch.zeros_like(data)

                for j in range(0, len(local_model_update_list)):
                    local_model_update_dict= local_model_update_list[j]
                    for name, data in local_model_update_dict.items():
                        if not torch.is_tensor(data):
                            # logger.info(f'name: {name}, data: {data}')
                            data = torch.FloatTensor(data)
                        # weight_accumulator[name].add_(local_model_update_dict[name])
                        # update[name].add_(local_model_update_dict[name])
                        weight_accumulator[name].add_(data)
                        update[name].add_(data)
                        detached_data= data.cpu().detach().numpy()
                        # print(detached_data.shape)
                        detached_data=detached_data.tolist()
                        # print(detached_data)
                        local_model_update_dict[name]=detached_data # from gpu to cpu

                updates[state_keys[i]]=(num_samples,update)

            return weight_accumulator,updates

    def init_weight_accumulator(self, target_model):
        weight_accumulator = dict()
        for name, data in target_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)

        return weight_accumulator

    def average_shrink_models(self, weight_accumulator, target_model, epoch_interval):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.

        """
        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue

            update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["no_models"])
            # update_per_layer = weight_accumulator[name] * (self.params["eta"] / self.params["number_of_total_participants"])

            # update_per_layer = update_per_layer * 1.0 / epoch_interval
            if self.params['diff_privacy']:
                update_per_layer.add_(self.dp_noise(data, self.params['sigma']))

            try:
                data.add_(update_per_layer)
            except:
                data.add_(update_per_layer.to(data.dtype))
        return True

    def add_noise(self, target_model, noise_level):
        """
        Add noise to the model weights.
        """
        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue
            if 'weight' in name or 'bias' in name:
                try:
                    data.add_(self.dp_noise(data, noise_level))
                except:
                    data.add_(0)
        return True

    def cos_calc_btn_grads(self, l1, l2):
        return torch.dot(l1, l2)/(torch.linalg.norm(l1)+1e-9)/(torch.linalg.norm(l2)+1e-9)

    def flatten_gradient_v2(self, local_model_update_dict):
        layer_names = list(local_model_update_dict.keys())
        for name in layer_names:
            if 'bias' in name or 'weight' in name:
                continue
            else:
                layer_names.remove(name)
        grad_len = [np.array(local_model_update_dict[name].cpu().data.numpy().shape).prod() for name in layer_names]
        grad = []
        for idx, name in enumerate(layer_names):
            grad.append(local_model_update_dict[name].cpu().data.numpy().flatten())

        grad = np.hstack(grad)
        return grad

    def flatten_gradient(self, client_grad):
        # num_clients = len(client_grads)
        num_of_layers = len(client_grad)
        # print(num_of_layers)
        grad_len = [np.array(client_grad[i].cpu().data.numpy().shape).prod() for i in range(num_of_layers)]
        # print(grad_len)
        grad = []
        for i in range(num_of_layers):
            grad.append(np.reshape(client_grad[i].cpu().data.numpy(), grad_len[i]))
        # print(grad)
        grad = np.hstack(grad)
        return grad

    def snapshot(self, grads, names, alphas, epoch, cluster=None):
        # if len(names) < self.params['number_of_total_participants']:
        #     return

        grads = np.array(grads)
        # output_layer_grads = np.delete(grads, slice(-5010,0), axis=1)
        # inner_layer_grads = np.delete(grads, slice(0, -5010), axis=1)
        # grad_dict = dict()
        # grad_dict['grads'] = grads
        # grad_dict['output_layer_grads'] = output_layer_grads
        # grad_dict['inner_layer_grads'] = inner_layer_grads

        # for grad_name in grad_dict.keys():
        #     iter_grads = grad_dict[grad_name]
        #     pca = PCA(n_components=2)
        #     reduced_grads = pca.fit(iter_grads.T)
        #     adv_inds = [i for i in range(len(names)) if names[i] in self.adversarial_namelist]
        #     benign_inds = [i for i in range(len(names)) if names[i] in self.benign_namelist]

        #     if cluster is None:

        #         plt.figure(figsize=(5, 3))
        #         # colors = ['red' if x in adv_inds else 'blue' for x in range(len(iter_grads))]
        #         color_mapping = [self.lsrs[adv_inds[x]][self.source_class] for x in range(len(adv_inds))]
        #         colors = ['blue' if color_mapping[x] < 0.05 else 'black' if color_mapping[x] < 0.1 else 'red' for x in range(len(color_mapping))]
        #         plt.scatter(reduced_grads.components_[0][adv_inds], reduced_grads.components_[1][adv_inds], c=colors, marker='x')
        #         plt.scatter(reduced_grads.components_[0][benign_inds], reduced_grads.components_[1][benign_inds], c='blue', marker='o')
        #         plt.title(f'Epoch {epoch}')
        #         plt.savefig(f'{self.folder_path}/epoch_{epoch}_{grad_name}.png')

        #     else:
        #         plt.figure(figsize=(5, 3))
        #         markers = ['o', 'x', '^', 'v', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', '+', '.', ',']

        #         for idx, group in enumerate(cluster):
        #             colors = ['red' if x in adv_inds else 'blue' for x in group]
        #             plt.scatter(reduced_grads.components_[0][group], reduced_grads.components_[1][group], c=colors, marker=markers[idx])
        #         plt.title(f'Epoch {epoch}')
        #         plt.savefig(f'{self.folder_path}/epoch_{epoch}_{grad_name}.png')
        #         plt.close()

                


        # del grad_dict

        save_grads = False
        if epoch > 40 or not save_grads:
            return

        prefix = 'dirichelt_' if not self.params['noniid'] else ''
        prefix = self.params['type'] + '_' + prefix
        basepath = 'utils/temp_grads'
        np.save(f'{basepath}/{prefix}grads_{epoch}.npy', grads)
        np.save(f'{basepath}/{prefix}names_{epoch}.npy', names)
        np.save(f'{basepath}/{prefix}advs_{epoch}.npy', self.adversarial_namelist)
        np.save(f'{basepath}/{prefix}alphas_{epoch}.npy', alphas)
        np.save(f'{basepath}/{prefix}lsrs_{epoch}.npy', [self.lsrs[name] for name in names])

        return True

    def convert_model_to_param_list(self, model):
        idx=0
        params_to_copy_group=[]
        for name, param in model.items():
            num_params_to_copy = torch.numel(param)
            params_to_copy_group.append(param.reshape([num_params_to_copy]).clone().detach())
            idx+=num_params_to_copy

        params=torch.ones([idx])
        idx=0
        for param in params_to_copy_group:    
            for par in param:
                params[idx].copy_(par)
                idx += 1

        return params

    def get_validation_score(self, candidate, cluster):
        centroid = np.mean(cluster, axis=0)
        return np.mean(euclidean_distances([candidate, centroid]))

    def get_average_distance(self, candidate, cluster):
        # return np.sum(euclidean_distances(cluster, [candidate]))/(len(cluster)-1)
        return np.sum(cosine_distances(cluster, [candidate]))/(len(cluster)-1)
    

    def print_util(self, a, b):
        return str(a) + ': ' + str(b)


    def ipm_attack(self, delta_models, names):
        bad_idx = [idx for idx, name in enumerate(names) if name in self.adversarial_namelist]
        benign_delta_models = [dm for idx, dm in enumerate(delta_models) if idx not in bad_idx]

        weights = np.ones(len(benign_delta_models), dtype=np.float32)
        weights = [-w * self.params['ipm_val']/len(benign_delta_models) for w in weights]
        ipm_delta_models = self.weighted_sum_oracle(benign_delta_models, torch.tensor(weights))

        new_delta_models = []

        for idx in range(len(delta_models)):
            if idx in bad_idx:
                new_delta_models.append(copy.deepcopy(ipm_delta_models))
            else:
                new_delta_models.append(delta_models[idx])

        return new_delta_models



    def combined_clustering_guided_aggregation_v2(self, target_model, updates, epoch, weight_accumulator):
        start_epoch = self.start_epoch
        if epoch < start_epoch:
            self.fedavg(target_model, updates)
            return
        start_time = time.time()
        t = time.time()
        logger.info(f'Started clustering guided aggregation')
        client_grads = []
        alphas = []
        names = []
        delta_models = []
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
            delta_models.append(data[2])
            names.append(name)

        if self.params['ipm_attack']:
            delta_models = self.ipm_attack(delta_models, names)
        
        wv = np.zeros(len(names), dtype=np.float32)
        # grads = [self.flatten_gradient(client_grad) for client_grad in client_grads]
        grads = [self.flatten_gradient_v2(delta_model) for delta_model in delta_models]
        norms = [np.linalg.norm(grad) for grad in grads]
        full_grads = copy.deepcopy(grads)

        output_layer_only = False
        if output_layer_only:
            n_comp = 10
            pca = PCA(n_components=n_comp)
            pca_grads = pca.fit(np.array(grads).T)
            grads = [pca_grads.components_[idx] for idx in range(n_comp)]
            grads = np.array(grads).T

        # logger.info(f'grads shape: {grads.shape}')
        logger.info(f'Converted gradients to param list: Time: {time.time() - t}')

        if self.params['save_grads']:
            np.save(f'{self.folder_path}/grads_{epoch}.npy', grads)
            np.save(f'{self.folder_path}/names_{epoch}.npy', names)

        if self.params['type'] != config.TYPE_LOAN:
            num_of_classes = 10
        else:
            num_of_classes = 9

        no_clustering = False
        if self.params['injective_florida']:
            no_clustering = True

        no_ensemble = False
        if self.params['no_ensemble']:
            no_ensemble = True

        if self.params['no_models'] < 10 or no_clustering:
            self.clusters_agg = [[i] for i in range(len(names))]
        else:
            clustering_method = self.params['clustering_method'] if self.params['clustering_method'] is not None else 'Agglomerative'
            _, self.clusters_agg = cluster_function(grads, clustering_method)
        
        logger.info(f'Agglomerative Clustering: Time: {time.time() - t}')
        t = time.time()

        if self.get_param_val('ablation_hard_mixture') is not None:
            benign_rate = self.get_param_val('ablation_hard_mixture')
            benign_spillover = int(benign_rate * len(self.adversarial_namelist))
            self.clusters_agg = [np.arange(len(self.adversarial_namelist)-benign_spillover).tolist() + np.arange(len(self.adversarial_namelist), len(self.adversarial_namelist)+benign_spillover).tolist(), np.arange(len(self.adversarial_namelist)-benign_spillover, len(self.adversarial_namelist)).tolist() + np.arange(len(self.adversarial_namelist)+benign_spillover, len(names)).tolist()]

        clusters_agg = []
        logger.info('Clustering by model updates')
        for idx, cluster in enumerate(self.clusters_agg):
            logger.info(f'cluster: {cluster}')
            logger.info(f'names: {names}')
            clstr = [names[c] for c in cluster]
            clusters_agg.append(clstr)


        print(f'Validating all clients at epoch {epoch}')

        all_validator_evaluations = {}
        evaluations_of_clusters = {}
        count_of_class_for_validator = {}

        for name in names:
            all_validator_evaluations[name] = []

        evaluations_of_clusters[-1] = {}
        for iidx, val_idx in enumerate(names):
            val_score_by_class, val_score_by_class_per_example, count_of_class = validation_test(self, target_model, val_idx if self.params['type'] != config.TYPE_LOAN else iidx)
            val_score_by_class_per_example = [val_score_by_class_per_example[i] for i in range(num_of_classes)]
            all_validator_evaluations[val_idx] += val_score_by_class_per_example
            evaluations_of_clusters[-1][val_idx] = [val_score_by_class[i] for i in range(num_of_classes)]
            if val_idx not in count_of_class_for_validator.keys():
                count_of_class_for_validator[val_idx] = count_of_class

        num_of_clusters = len(clusters_agg)

        adj_delta_models = []

        for idx, cluster in enumerate(tqdm(clusters_agg, disable=False)):
            evaluations_of_clusters[idx] = {}
            agg_model = self.new_model()
            agg_model.copy_params(self.target_model.state_dict())
            weight_vec = np.zeros(len(names), dtype=np.float32)

            if len(cluster) != 1 or no_ensemble:
                for i in range(len(names)):
                    if names[i] in cluster:
                        weight_vec[i] = 1/len(cluster)
            else:
                cos_sims = np.array(cosine_similarity(grads, [grads[self.clusters_agg[idx][0]]])).flatten()
                # logger.info(f'cos_sims by order for client {self.clusters_agg[idx][0]}: {cos_sims}')
                trust_scores = np.zeros(cos_sims.shape)
                for i in range(len(cos_sims)):
                    # trust_scores[i] = cos_sims[i]/np.linalg.norm(grads[i])/np.linalg.norm(clean_server_grad)
                    trust_scores[i] = cos_sims[i]
                    trust_scores[i] = max(trust_scores[i], 0)

                norm_ref = norms[self.clusters_agg[idx][0]]
                clip_vals = [min(norm_ref/norm, 1) for norm in norms]
                trust_scores = [ts * cv for ts, cv in zip(trust_scores, clip_vals)]
                weight_vec = trust_scores

            aggregate_weights = self.weighted_average_oracle(delta_models, torch.tensor(weight_vec))
            adj_delta_models.append(aggregate_weights)

            for name, data in agg_model.state_dict().items():
                update_per_layer = aggregate_weights[name]
                try:
                    data.add_(update_per_layer)
                except:
                    data.add_(update_per_layer.to(data.dtype))
                    

            for iidx, val_idx in enumerate(tqdm(names, disable=True)):
                val_score_by_class, val_score_by_class_per_example, count_of_class = validation_test(self, agg_model, val_idx if self.params['type'] != config.TYPE_LOAN else iidx)
                val_score_by_class_per_example = [val_score_by_class_per_example[i] for i in range(num_of_classes)]

                val_score_by_class_per_example = [-val_score_by_class_per_example[i]+all_validator_evaluations[val_idx][i] for i in range(num_of_classes)]
                    
                all_validator_evaluations[val_idx]+= val_score_by_class_per_example
                evaluations_of_clusters[idx][val_idx] = [-val_score_by_class[i]+evaluations_of_clusters[-1][val_idx][i] for i in range(num_of_classes)]
            
            # for client in cluster:
            #     all_val_acc_list_dict[client] = val_acc_list

        # imputing missing validation values
        # convert from dict to list
        # all_validator_evaluations = [all_validator_evaluations[val_idx] for val_idx in range(len(names))]

        # need to fix imputation steps
        # all_validator_evaluations = [all_validator_evaluations[names[val_idx]] for val_idx in range(len(names))]
        # imputer = IterativeImputer(n_nearest_features = 5, initial_strategy = 'median', random_state = 42)
        # all_validator_evaluations = imputer.fit_transform(all_validator_evaluations)

        cluster_maliciousness = [len([idx for idx in cluster if idx in self.adversarial_namelist])/len(cluster) for cluster in clusters_agg]
        logger.info(f'cluster maliciousness: {cluster_maliciousness}')
        
        validation_container = {
            'evaluations_of_clusters': evaluations_of_clusters,
            'count_of_class_for_validator': count_of_class_for_validator,
            'names': names,
            'num_of_classes': num_of_classes,
            'num_of_clusters': num_of_clusters,
            'all_validator_evaluations': all_validator_evaluations,
            'epoch': epoch,
            'params': dict(self.params),
            'cluster_maliciousness': cluster_maliciousness,
            'benign_namelist': self.benign_namelist,
            'adversarial_namelist': self.adversarial_namelist,
        }

        with open(f'{self.folder_path}/validation_container_{epoch}.pkl', 'wb') as f:
            logger.info(f'saving validation container to {self.folder_path}/validation_container_{epoch}.pkl with params type {type(self.params)}')
            pickle.dump(validation_container, f)

        evaluations_of_clusters = impute_validation(evaluations_of_clusters, count_of_class_for_validator, num_of_clusters, num_of_classes)

        # all_validator_evaluations_dict = dict()
        # for val_idx in range(len(names)):
        #     all_validator_evaluations_dict[names[val_idx]] = all_validator_evaluations[val_idx]

        # all_validator_evaluations = all_validator_evaluations_dict

            
        #validation scores tabulation
        # for iidx in range(1):
        #     itr_idx = randint(0, 10)
        #     end_idx = min(10*(itr_idx+1), len(names))
        #     start_idx = end_idx - 10
        #     all_val_scores = [all_validator_evaluations[val_idx] for val_idx in names]
        #     scores = all_val_scores[start_idx:end_idx]
        #     scores = np.array(scores)
        #     scores = scores.T
        #     scores = scores.tolist()
        #     for scores_idx in range(len(scores)):
        #         mean_score = np.mean(scores[scores_idx])
        #         std_score = np.std(scores[scores_idx])
        #         for idx in range(len(scores[scores_idx])):
        #             if names[idx + start_idx] in self.adversarial_namelist:
        #                 color = "red"
        #             else:
        #                 color = "blue"
        #             # if scores_idx == num_of_classes * max_mal_cluster_index + self.source_class:
        #             cluster_idx = scores_idx // num_of_classes -1
        #             if (scores_idx - self.source_class)%num_of_classes == 0 and cluster_idx >= 0:
        #                 if cluster_adversarialness[cluster_idx] > 0:
        #                     highlight_color = "on_white"
        #                 else:
        #                     highlight_color = "on_yellow"
        #                 scores[scores_idx][idx] = colored("{:.2f}".format(scores[scores_idx][idx]), color, highlight_color)
        #             else:
        #                 scores[scores_idx][idx] = colored("{:.2f}".format(scores[scores_idx][idx]), color=color)

        #     table_header = [colored(val_idx, color=f'{"red" if val_idx in self.adversarial_namelist else "blue"}') for val_idx in names[10*itr_idx:10*(itr_idx+1)]]
        #     print(tabulate(scores, headers=table_header))
        
        validation_container = {
            'evaluations_of_clusters': evaluations_of_clusters,
            'count_of_class_for_validator': count_of_class_for_validator,
            'names': names,
            'num_of_classes': num_of_classes,
            'num_of_clusters': num_of_clusters,
            'all_validator_evaluations': all_validator_evaluations,
            'epoch': epoch,
            'params': dict(self.params),
            'cluster_maliciousness': cluster_maliciousness,
            'benign_namelist': self.benign_namelist,
            'adversarial_namelist': self.adversarial_namelist,
        }

        with open(f'{self.folder_path}/validation_container_{epoch}.pkl', 'wb') as f:
            logger.info(f'saving validation container to {self.folder_path}/validation_container_{epoch}.pkl with params type {type(self.params)}')
            pickle.dump(validation_container, f)


        logger.info(f'Validation Done: Time: {time.time() - t}')
        t = time.time()

        validation_container['params'] = defaultdict(lambda: None, validation_container['params'])
        validation_container['params']['mal_val_type'] = 'adaptive'
        valProcessor = ValidationProcessor(validation_container=validation_container)
        wv_by_cluster = valProcessor.run()


        norm_median = np.median(norms)
        clipping_weights = [min(norm_median/norm, 1) for norm in norms]
        if 'ablation_study' in self.params.keys() and 'missing_clipping' in self.params['ablation_study']:
            clipping_weights = [1 for norm in norms]
        # wv = np.zeros(len(names), dtype=np.float32)
        green_clusters = []
        mal_pcnts = []
        for idx, cluster in enumerate(self.clusters_agg):
            mal_pcnts.append(sum([wv[cl_id] for cl_id in cluster if names[cl_id] in self.adversarial_namelist]))
            for cl_id in cluster:
                # wv[cl_id] = wv_by_cluster[idx]
                if no_clustering:
                    wv[cl_id] = wv_by_cluster[cl_id]
                else:
                    # wv[cl_id] = wv_by_cluster[idx] * len(cluster) * wv[cl_id]
                    wv[cl_id] = wv_by_cluster[idx]

        wv = [w*c for w,c in zip(wv, clipping_weights)]

        wv = wv/np.sum(wv)

        # logger.info(f'clipping_weights: {clipping_weights}')
        # logger.info(f'adversarial clipping weights: {[self.print_util(names[iidx], clipping_weights[iidx]) for iidx in range(len(clipping_weights)) if names[iidx] in self.adversarial_namelist]}')
        # logger.info(f'benign clipping weights: {[self.print_util(names[iidx], clipping_weights[iidx]) for iidx in range(len(clipping_weights)) if names[iidx] in self.benign_namelist]}')
        wv_print_str= '['
        for idx, w in enumerate(wv):
            wv_print_str += ' '
            if names[idx] in self.adversarial_namelist:
                wv_print_str += colored(str(w), 'blue')
            else:
                wv_print_str += str(w)
        wv_print_str += ']'
        print(f'wv: {wv_print_str}')
        aggregate_weights = self.weighted_average_oracle(delta_models, torch.tensor(wv))

        for name, data in target_model.state_dict().items():
            update_per_layer = aggregate_weights[name] * (self.params["eta"])
            try:
                data.add_(update_per_layer)
            except:
                data.add_(update_per_layer.to(data.dtype))

        # noise_level = self.params['sigma'] * norm_median
        # self.add_noise(target_model, noise_level)
        logger.info(f'Aggregation Done: Time {time.time() - t}')
        t = time.time()
        logger.info(f'adversarial wv: {[self.print_util(names[iidx], wv[iidx]) for iidx in range(len(wv)) if names[iidx] in self.adversarial_namelist]}')
        logger.info(f'benign wv: {[self.print_util(names[iidx], wv[iidx]) for iidx in range(len(wv)) if names[iidx] in self.benign_namelist]}')
        return

    

    def get_client_grad_from_model_update(self, delta_model):
        client_grad = []
        for name, data in delta_model.items():
            client_grad.append(data)
        return client_grad

    def fltrust_with_grad(self, target_model, updates):
        client_grads = []
        alphas = []
        names = []
        delta_models = []
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
            delta_models.append(data[2])
            names.append(name)

        adv_indices = [idx for idx, name in enumerate(names) if name in self.adversarial_namelist]
        benign_indices = [idx for idx, name in enumerate(names) if name in self.benign_namelist]

        # only for testing purpose
        grads = [self.flatten_gradient(client_grad) for client_grad in client_grads]
        logger.info(f'grad shape: {grads[0].shape}')
        # grads = client_grads
        clean_server_grad = grads[-1]
        # cos_sims = [self.cos_calc_btn_grads(client_grad, clean_server_grad) for client_grad in grads]
        cos_sims = np.array(cosine_similarity(grads, [clean_server_grad])).flatten()
        all_cos_sims = cos_sims
        logger.info(f'adv mean cos sim: {np.mean(cos_sims[adv_indices])}')
        logger.info(f'benign mean cos sim: {np.mean(cos_sims[benign_indices])}')
        cos_sims = cos_sims[:-1]
        # logger.info(f'cos_sims: {cos_sims}')
        # logger.info(f'adversarial cos_sims: {[self.print_util(names[iidx], cos_sims[iidx]) for iidx in range(len(cos_sims)) if names[iidx] in self.adversarial_namelist]}')
        # logger.info(f'benign cos_sims: {[self.print_util(names[iidx], cos_sims[iidx]) for iidx in range(len(cos_sims)) if names[iidx] in self.benign_namelist]}')

        # cos_sims = np.maximum(np.array(cos_sims), 0)
        # norm_weights = cos_sims/(np.sum(cos_sims)+1e-9)
        # for i in range(len(norm_weights)):
        #     norm_weights[i] = norm_weights[i] * np.linalg.norm(clean_server_grad) / (np.linalg.norm(grads[i]))
        trust_scores = np.zeros(cos_sims.shape)
        for i in range(len(cos_sims)):
            trust_scores[i] = cos_sims[i]/np.linalg.norm(grads[i])/np.linalg.norm(clean_server_grad)
            trust_scores[i] = max(trust_scores[i], 0)

        clipping_coeffs = np.ones(len(trust_scores))
        for i in range(len(trust_scores)):
            clipping_coeffs[i] = np.linalg.norm(clean_server_grad) / np.linalg.norm(grads[i])

        wv = trust_scores
        sum_trust_scores = np.sum(trust_scores)
        # logger.info(f'clipping_coeffs: {clipping_coeffs}')
        # logger.info(f'wv: {wv}')
        # logger.info(f'wv: {wv/sum_trust_scores}')

        agg_grads = []
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            temp = wv[0] * clipping_coeffs[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads[:-1]):
                if c == 0:
                    continue
                temp += wv[c] * clipping_coeffs[c] * client_grad[i].cpu()
                # print(temp)
                # temp += wv[c]
            temp = temp / sum_trust_scores
            agg_grads.append(temp)

        # logger.info(f'agg_grads: {self.flatten_gradient(agg_grads)}')

        wv = [wv[c] * clipping_coeffs[c]/sum_trust_scores for c in range(len(wv))]
        wv = np.array(wv)
        logger.info(f'adv mean wv: {np.mean(wv[adv_indices])}')
        logger.info(f'benign mean wv: {np.mean(wv[benign_indices[:-1]])}')
        # logger.info(f'adversarial wv: {[self.print_util(names[iidx], wv[iidx]) for iidx in range(len(wv)) if names[iidx] in self.adversarial_namelist]}')
        # logger.info(f'benign wv: {[self.print_util(names[iidx], wv[iidx]) for iidx in range(len(wv)) if names[iidx] in self.benign_namelist]}')

        # aggregate_weights = self.weighted_average_oracle(delta_models, torch.tensor(wv))

        # for name, data in target_model.state_dict().items():
        #     update_per_layer = aggregate_weights[name] * (self.params["eta"])
        #     try:
        #         data.add_(update_per_layer)
        #     except:
        #         logger.info(f'layer name: {name}')
        #         logger.info(f'data: {data}')
        #         logger.info(f'update_per_layer: {update_per_layer}')
        #         data.add_(update_per_layer.to(data.dtype))
        #         logger.info(f'after update: {update_per_layer.to(data.dtype)}')

        target_model.train()
        # train and update
        optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
                                    momentum=self.params['momentum'],
                                    weight_decay=self.params['decay'])

        optimizer.zero_grad()
        for i, (name, params) in enumerate(target_model.named_parameters()):
            agg_grads[i]=agg_grads[i] * self.params["eta"]
            if params.requires_grad:
                params.grad = agg_grads[i].to(config.device)
        optimizer.step()
        # noise_level = self.params['sigma']
        # self.add_noise(noise_level=noise_level)
        # utils.csv_record.add_weight_result(names, wv, alpha)
        return True, names, all_cos_sims

    def fltrust(self, target_model, updates, epoch=-1):
        client_grads = []
        alphas = []
        names = []
        delta_models = []
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
            delta_models.append(data[2])
            names.append(name)

        if 'ipm_attack' in self.params.keys() and self.params['ipm_attack']:
            delta_models = self.ipm_attack(delta_models, names)
        logger.info(f'names: {names}')
        adv_indices = [idx for idx, name in enumerate(names) if name in self.adversarial_namelist]
        benign_indices = [idx for idx, name in enumerate(names) if name in self.benign_namelist]
        src_class_indices = [40 + idx for idx in range(10)]
        non_src_class_indices = [idx for idx, name in enumerate(names) if name not in src_class_indices]
        non_src_class_indices = non_src_class_indices[:-1]

        # only for testing purpose
        # grads = [self.flatten_gradient(client_grad) for client_grad in client_grads]
        grads = [self.flatten_gradient_v2(delta_model) for delta_model in delta_models]
        logger.info(f'grad shape: {grads[0].shape}')
        # grads = client_grads
        clean_server_grad = grads[-1]
        # cos_sims = [self.cos_calc_btn_grads(client_grad, clean_server_grad) for client_grad in grads]
        cos_sims = np.array(cosine_similarity(grads, [clean_server_grad])).flatten()
        all_cos_sims = cos_sims
        # if epoch!=-1:
        #     for iidx, name in enumerate(names):
        #         if name in self.adversarial_namelist:
        #             all_cos_sims[iidx] = 0.
        logger.info(f'cos_sims by order: {np.argsort(cos_sims)}')
        logger.info(f'adv mean cos sim: {np.mean(cos_sims[adv_indices])}')
        logger.info(f'benign mean cos sim: {np.mean(cos_sims[benign_indices])}')
        # logger.info(f'src class cos sim: {cos_sims[src_class_indices]}')
        # logger.info(f'src class mean cos sim: {np.mean(cos_sims[src_class_indices])}')
        # logger.info(f'non src class cos sim: {cos_sims[non_src_class_indices]}')
        # logger.info(f'non src class mean cos sim: {np.mean(cos_sims[non_src_class_indices])}')

        # client_grads = [self.get_client_grad_from_model_update(delta_model) for delta_model in delta_models]

        # grads = [self.flatten_gradient(client_grad) for client_grad in client_grads]
        # logger.info(f'grad shape: {grads[0].shape}')
        # # grads = client_grads
        # clean_server_grad = grads[-1]
        # # cos_sims = [self.cos_calc_btn_grads(client_grad, clean_server_grad) for client_grad in grads]
        # cos_sims = np.array(cosine_similarity(grads, [clean_server_grad])).flatten()
        # all_cos_sims = cos_sims
        # logger.info(f'adv mean cos sim: {np.mean(cos_sims[adv_indices])}')
        # logger.info(f'benign mean cos sim: {np.mean(cos_sims[benign_indices])}')
        cos_sims = cos_sims[:-1]
        # logger.info(f'cos_sims: {cos_sims}')
        # logger.info(f'adversarial cos_sims: {[self.print_util(names[iidx], cos_sims[iidx]) for iidx in range(len(cos_sims)) if names[iidx] in self.adversarial_namelist]}')
        # logger.info(f'benign cos_sims: {[self.print_util(names[iidx], cos_sims[iidx]) for iidx in range(len(cos_sims)) if names[iidx] in self.benign_namelist]}')

        # cos_sims = np.maximum(np.array(cos_sims), 0)
        # norm_weights = cos_sims/(np.sum(cos_sims)+1e-9)
        # for i in range(len(norm_weights)):
        #     norm_weights[i] = norm_weights[i] * np.linalg.norm(clean_server_grad) / (np.linalg.norm(grads[i]))
        trust_scores = np.zeros(cos_sims.shape)
        for i in range(len(cos_sims)):
            # trust_scores[i] = cos_sims[i]/np.linalg.norm(grads[i])/np.linalg.norm(clean_server_grad)
            trust_scores[i] = cos_sims[i]
            trust_scores[i] = max(trust_scores[i], 0)

        clipping_coeffs = np.ones(len(trust_scores))
        for i in range(len(trust_scores)):
            clipping_coeffs[i] = np.linalg.norm(clean_server_grad) / np.linalg.norm(grads[i])

        wv = trust_scores
        sum_trust_scores = np.sum(trust_scores)
        # wv = np.ones(self.params['no_models'])
        # wv = wv/len(wv)
        # logger.info(f'clipping_coeffs: {clipping_coeffs}')
        # logger.info(f'wv: {wv}')
        # logger.info(f'wv: {wv/sum_trust_scores}')

        # agg_grads = []
        # # Iterate through each layer
        # for i in range(len(client_grads[0])):
        #     temp = wv[0] * clipping_coeffs[0] * client_grads[0][i].cpu().clone()
        #     # Aggregate gradients for a layer
        #     for c, client_grad in enumerate(client_grads[:-1]):
        #         if c == 0:
        #             continue
        #         temp += wv[c] * clipping_coeffs[c] * client_grad[i].cpu()
        #         # print(temp)
        #         # temp += wv[c]
        #     temp = temp / sum_trust_scores
        #     agg_grads.append(temp)

        # logger.info(f'agg_grads: {self.flatten_gradient(agg_grads)}')
        wv = [wv[c] * clipping_coeffs[c]/sum_trust_scores for c in range(len(wv))]
        wv = np.array(wv)
        # try:
        #     logger.info(f'adv mean wv: {np.mean(wv[adv_indices])}')
        #     logger.info(f'benign mean wv: {np.mean(wv[benign_indices])}')
        # except:
        #     pass
        # logger.info(f'adversarial wv: {[self.print_util(names[iidx], wv[iidx]) for iidx in range(len(wv)) if names[iidx] in self.adversarial_namelist]}')
        # logger.info(f'benign wv: {[self.print_util(names[iidx], wv[iidx]) for iidx in range(len(wv)) if names[iidx] in self.benign_namelist]}')

        aggregate_weights = self.weighted_average_oracle(delta_models, torch.tensor(wv))

        for name, data in target_model.state_dict().items():
            update_per_layer = aggregate_weights[name] * (self.params["eta"])
            try:
                data.add_(update_per_layer)
            except:
                logger.info(f'layer name: {name}')
                logger.info(f'data: {data}')
                logger.info(f'update_per_layer: {update_per_layer}')
                data.add_(update_per_layer.to(data.dtype))
                logger.info(f'after update: {update_per_layer.to(data.dtype)}')

        # target_model.train()
        # # train and update
        # optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
        #                             momentum=self.params['momentum'],
        #                             weight_decay=self.params['decay'])

        # optimizer.zero_grad()
        # for i, (name, params) in enumerate(target_model.named_parameters()):
        #     agg_grads[i]=agg_grads[i] * self.params["eta"]
        #     if params.requires_grad:
        #         params.grad = agg_grads[i].to(config.device)
        # optimizer.step()
        # noise_level = self.params['sigma']
        # self.add_noise(noise_level=noise_level)
        # utils.csv_record.add_weight_result(names, wv, alpha)
        return True, names, all_cos_sims

    def calc_prob_for_AFA(self, participant_no):
        if isinstance(participant_no, str):
            participant_no = self.participants_list.index(participant_no)
        return (self.good_count[participant_no]+3)/(self.good_count[participant_no]+self.bad_count[participant_no]+6)

    def cluster_attack(self, delta_models, bad_count, ref_idx):
        sum_bad_delta_models = self.weighted_sum_oracle(delta_models[:bad_count], torch.ones(bad_count))

        ref_delta_model = delta_models[ref_idx]

        gap_delta_model = self.weighted_sum_oracle([sum_bad_delta_models, ref_delta_model], torch.tensor([1, -bad_count]))
        incr_delta_model = self.weighted_sum_oracle([gap_delta_model], torch.tensor([2/(bad_count * (bad_count + 1))]))

        new_delta_models = []
        for i in range(bad_count):
            new_delta_models.append(self.weighted_sum_oracle([ref_delta_model, incr_delta_model], torch.tensor([1, i+1])))

        for i in range(len(delta_models)-bad_count):
            new_delta_models.append(delta_models[bad_count+i])

        return new_delta_models


    def flame(self, target_model, updates, epoch):
        client_grads = []
        alphas = []
        names = []
        delta_models = []
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
            delta_models.append(data[2])
            names.append(name)

        grads_old = [self.flatten_gradient(client_grad) for client_grad in client_grads]
        grads = [self.flatten_gradient_v2(delta_model) for delta_model in delta_models]

        logger.info(f'grads: {grads[0][:20]}, grads_old: {grads_old[0][:20]}')
        logger.info(f'len(grads): {len(grads[0])}')

        save_grads = True
        if save_grads:
            self.snapshot(grads, names, alphas, epoch)
        adv_list = [i for i in range(len(grads)) if names[i] in self.adversarial_namelist]
        # wv, good_clients, edge_list, _ = modHDBSCAN(np.array(grads), min_samples=self.params['no_models']//2 + 1, adv_list=adv_list)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.params['no_models']//2 + 1, min_samples=1, allow_single_cluster=True, metric = 'precomputed')
        cluster_result = clusterer.fit_predict(np.array(cosine_distances(grads), dtype=np.float64))

        cluster_attack_on = False

        if cluster_attack_on:
            bad_count = len([name for name in names if name in self.adversarial_namelist])
            ref_idx = np.argwhere(cluster_result[bad_count:]==0)[0][0] + bad_count
            new_delta_models = self.cluster_attack(delta_models, bad_count, ref_idx)
            new_grads = [self.flatten_gradient_v2(delta_model) for delta_model in new_delta_models]
            cluster_result = clusterer.fit_predict(np.array(cosine_distances(new_grads), dtype=np.float64))
            delta_models = new_delta_models

        wv = np.array([res+1 for res in cluster_result])
        actual_labels = [1 if names[idx] in self.adversarial_namelist else 0 for idx in range(len(names))]
        cluster_labels = [abs(res) for res in cluster_result]
        false_negative_labels = [idx for idx in range(len(cluster_labels)) if cluster_labels[idx] == 0 and actual_labels[idx] == 1]
        for fn in false_negative_labels:
            logger.info(f'{names[fn]} is a false negative with lsr {self.lsrs[names[fn]]}')
        logger.info(f'cluster_result: {cluster_result}')
        cm = confusion_matrix(actual_labels, cluster_labels)
        logger.info(f'cm: {cm}')


        norms = [np.linalg.norm(grad) for grad in grads]
        norm_median = np.median(norms)
        clipping_weights = [min(norm_median/norm, 1) for norm in norms]

        wv = [wv[c] * clipping_weights[c] for c in range(len(wv))]

        self.color_print_wv(wv, names)

        aggregate_weights = self.weighted_average_oracle(delta_models, torch.tensor(wv))

        for name, data in target_model.state_dict().items():
            update_per_layer = aggregate_weights[name] * (self.params["eta"])
            try:
                data.add_(update_per_layer)
            except:
                # logger.info(f'layer name: {name}')
                # logger.info(f'data: {data}')
                # logger.info(f'update_per_layer: {update_per_layer}')
                data.add_(update_per_layer.to(data.dtype))
                # logger.info(f'after update: {update_per_layer.to(data.dtype)}')

        noise_level = self.params['sigma'] * norm_median
        noise_level = noise_level ** 2
        # self.add_noise(target_model, noise_level=noise_level)
        
        return

    def fedavg(self, target_model, updates, epoch):
        client_grads = []
        alphas = []
        names = []
        delta_models = []
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
            delta_models.append(data[2])
            names.append(name)

        if 'ipm_attack' in self.params.keys() and self.params['ipm_attack']:
            delta_models = self.ipm_attack(delta_models, names)
        grads = [self.flatten_gradient(client_grad) for client_grad in client_grads]
        # wv = modHDBSCAN(grads)

        save_grads = False
        if save_grads and epoch<20:
            if self.params['noniid']:
                np.save(f'utils/temp_grads/grads_{epoch}.npy', grads)
                np.save(f'utils/temp_grads/names_{epoch}.npy', names)
                np.save(f'utils/temp_grads/advs_{epoch}.npy', self.adversarial_namelist)
                np.save(f'utils/temp_grads/alphas_{epoch}.npy', alphas)
                np.save(f'utils/temp_grads/lsrs_{epoch}.npy', self.lsrs)
            else:
                np.save(f'utils/temp_grads/dirichlet_grads_{epoch}.npy', grads)
                np.save(f'utils/temp_grads/dirichlet_names_{epoch}.npy', names)
                np.save(f'utils/temp_grads/dirichlet_advs_{epoch}.npy', self.adversarial_namelist)
                np.save(f'utils/temp_grads/dirichlet_alphas_{epoch}.npy', alphas)
                np.save(f'utils/temp_grads/dirichlet_lsrs_{epoch}.npy', self.lsrs)

        wv = np.array(alphas)/np.sum(alphas)

        if 'oracle_mode' in self.params.keys() and self.params['oracle_mode']:
            for idx, name in enumerate(names):
                if name in self.adversarial_namelist:
                    wv[idx] = 0
        logger.info(f'alphas: {alphas}')
        logger.info(f'wv: {wv}')
        self.color_print_wv(wv, names)
        # agg_grads = []
        # # Iterate through each layer
        # for i in range(len(client_grads[0])):
        #     # assert len(wv) == len(cluster_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
        #     temp = wv[0] * client_grads[0][i].cpu().clone()
        #     # Aggregate gradients for a layer
        #     for c, client_grad in enumerate(client_grads):
        #         if c == 0:
        #             continue
        #         temp += wv[c] * client_grad[i].cpu()
        #     # temp = temp / len(wv)
        #     agg_grads.append(temp)

        # target_model.train()
        # # train and update
        # optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
        #                             momentum=self.params['momentum'],
        #                             weight_decay=self.params['decay'])

        # optimizer.zero_grad()

        # for i, (name, params) in enumerate(target_model.named_parameters()):
        #     agg_grads[i]=agg_grads[i] * self.params["eta"]
        #     if params.requires_grad:
        #         params.grad = agg_grads[i].to(config.device)
        # # print(self.convert_model_to_param_list(grad_state_dict))
        # optimizer.step()
        aggregate_weights = self.weighted_average_oracle(delta_models, torch.tensor(wv))

        for name, data in target_model.state_dict().items():
            update_per_layer = aggregate_weights[name] * (self.params["eta"])
            try:
                data.add_(update_per_layer)
            except:
                # logger.info(f'layer name: {name}')
                # logger.info(f'data: {data}')
                # logger.info(f'update_per_layer: {update_per_layer}')
                data.add_(update_per_layer.to(data.dtype))
                # logger.info(f'after update: {update_per_layer.to(data.dtype)}')

        return True

    def afa_method(self, target_model, updates):
        client_grads = []
        alphas = []
        names = []
        delta_models = []
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
            delta_models.append(data[2])
            names.append(name)

        names = [self.participants_list.index(name) for name in names]
        logger.info(names)

        # good_set = set(self.adversarial_namelist + self.benign_namelist)
        # good_set = set(np.arange(self.params['no_models']))
        good_set = set(np.arange(len(names)))
        bad_set = set()
        r_set = set([-1])
        epsilon = self.params['afa_epsilon']

        while len(r_set) > 0:
            logger.info(f'r_set: {r_set}')
            logger.info(f'good_set: {good_set}')
            logger.info(f'bad_set: {bad_set}')
            r_set.clear()

            wv = [self.calc_prob_for_AFA(names[m_id]) for m_id in good_set]
            # might want to try using this later
            good_client_grads=[client_grads[m_id] for m_id in good_set]
            good_alphas=[alphas[m_id] for m_id in good_set]
            good_alphas = np.array(good_alphas)/np.sum(good_alphas)
            agg_grads = []
            # Iterate through each layer
            for i in range(len(client_grads[0])):
                # assert len(wv) == len(cluster_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
                temp = wv[0] * good_alphas[0] * client_grads[0][i].cpu().clone()
                # Aggregate gradients for a layer
                for c, cl_id in enumerate(good_set):
                    if c == 0:
                        continue
                    temp += wv[c] * good_alphas[c] * client_grads[cl_id][i].cpu()
                # temp = temp / len(wv)
                agg_grads.append(temp)

            grads = [self.flatten_gradient(client_grad) for cl_id, client_grad in enumerate(client_grads) if cl_id in good_set]
            agg_grad_flat = self.flatten_gradient(agg_grads)
            # cos_sims = np.array([cosine_similarity(client_grad, agg_grad_flat) for client_grad in grads])
            cos_sims = np.array(cosine_similarity(grads, [agg_grad_flat])).flatten()
            logger.info(f'cos_sims: {cos_sims}')

            mean_cos_sim, median_cos_sim, std_cos_sim = np.mean(cos_sims), np.median(cos_sims), np.std(cos_sims)

            if mean_cos_sim < median_cos_sim:
                good_set_copy = good_set.copy()
                for cl_id, client in enumerate(good_set_copy):
                    if cos_sims[cl_id] < median_cos_sim - epsilon * std_cos_sim:
                        r_set.add(client)
                        good_set.remove(client)
            else:
                good_set_copy = good_set.copy()
                for cl_id, client in enumerate(good_set_copy):
                    if cos_sims[cl_id] > median_cos_sim + epsilon * std_cos_sim:
                        r_set.add(client)
                        good_set.remove(client)
            
            epsilon += self.params['afa_del_epsilon']
            bad_set = bad_set.union(r_set)

        wv = [self.calc_prob_for_AFA(names[m_id]) for m_id in good_set]
        good_alphas=[alphas[m_id] for m_id in good_set]
        good_alphas = np.array(good_alphas)/np.sum(good_alphas)
        # agg_grads = []
        # # Iterate through each layer
        # for i in range(len(client_grads[0])):
        #     # assert len(wv) == len(cluster_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
        #     temp = wv[0] * good_alphas[0] * client_grads[0][i].cpu().clone()
        #     # Aggregate gradients for a layer
        #     for c, cl_id in enumerate(good_set):
        #         if c == 0:
        #             continue
        #         temp += wv[c] * good_alphas[c] * client_grads[cl_id][i].cpu()
        #     # temp = temp / len(wv)
        #     agg_grads.append(temp)

        # target_model.train()
        # # train and update
        # optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
        #                             momentum=self.params['momentum'],
        #                             weight_decay=self.params['decay'])

        # optimizer.zero_grad()

        # for i, (name, params) in enumerate(target_model.named_parameters()):
        #     agg_grads[i]=agg_grads[i] * self.params["eta"]
        #     if params.requires_grad:
        #         params.grad = agg_grads[i].to(config.device)
        # # print(self.convert_model_to_param_list(grad_state_dict))
        # optimizer.step()

        aggregate_weights = self.weighted_average_oracle(delta_models, torch.tensor(wv))

        for name, data in target_model.state_dict().items():
            update_per_layer = aggregate_weights[name] * (self.params["eta"])
            try:
                data.add_(update_per_layer)
            except:
                # logger.info(f'layer name: {name}')
                # logger.info(f'data: {data}')
                # logger.info(f'update_per_layer: {update_per_layer}')
                data.add_(update_per_layer.to(data.dtype))


        for cl_id in bad_set:
            self.bad_count[names[cl_id]] += 1
        for cl_id in good_set:
            self.good_count[names[cl_id]] += 1

        return True, names, wv

    def foolsgold_update(self,target_model,updates):
        client_grads = []
        alphas = []
        names = []
        delta_models = []
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
            delta_models.append(data[2])
            names.append(name)
        
        adver_ratio = 0
        for i in range(0, len(names)):
            _name = names[i]
            if _name in self.adversarial_namelist:
                adver_ratio += alphas[i]
        adver_ratio = adver_ratio / sum(alphas)
        poison_fraction = adver_ratio * self.params['poisoning_per_batch'] / self.params['batch_size']
        logger.info(f'[foolsgold agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[foolsgold agg] considering poison per batch poison_fraction: {poison_fraction}')

        # target_model.train()
        # # train and update
        # # probably need to apply a global learning rate
        # optimizer = torch.optim.SGD(target_model.parameters(), lr=1,
        #                             momentum=self.params['momentum'],
        #                             weight_decay=self.params['decay'])

        # optimizer.zero_grad()
        # print(client_grads[0])
        agg_grads, wv,alpha = self.fg.aggregate_gradients(client_grads,names)

        aggregate_weights = self.weighted_average_oracle(delta_models, torch.tensor(wv))

        for name, data in target_model.state_dict().items():
            update_per_layer = aggregate_weights[name] * (self.params["eta"])
            try:
                data.add_(update_per_layer)
            except:
                # logger.info(f'layer name: {name}')
                # logger.info(f'data: {data}')
                # logger.info(f'update_per_layer: {update_per_layer}')
                data.add_(update_per_layer.to(data.dtype))
                # logger.info(f'after update: {update_per_layer.to(data.dtype)}')

        # grad_state_dict = {}
        # for i, (name, params) in enumerate(target_model.named_parameters()):
        #     grad_state_dict[name] = agg_grads[i]
        #     agg_grads[i]=agg_grads[i] * self.params["eta"]
        #     if params.requires_grad:
        #         params.grad = agg_grads[i].to(config.device)
        # # print(self.convert_model_to_param_list(grad_state_dict))
        # optimizer.step()
        # wv=wv.tolist()
        utils.csv_record.add_weight_result(names, wv, alpha)
        return True, names, wv, alpha

    def geometric_median_update(self, target_model, updates, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6, max_update_norm= None):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
               """
        points = []
        alphas = []
        names = []
        for name, data in updates.items():
            points.append(data[1]) # update
            alphas.append(data[0]) # num_samples
            names.append(name)

        adver_ratio=0
        for i in range(0,len(names)):
            _name= names[i]
            if _name in self.adversarial_namelist:
                adver_ratio+= alphas[i]
        adver_ratio= adver_ratio/ sum(alphas)
        poison_fraction= adver_ratio* self.params['poisoning_per_batch']/ self.params['batch_size']
        if verbose:
            logger.info(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
            logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')

        alphas = np.asarray(alphas, dtype=np.float64) / sum(alphas)
        alphas = torch.from_numpy(alphas).float()

        # alphas.float().to(config.device)
        median = Helper.weighted_average_oracle(points, alphas)
        num_oracle_calls = 1

        # logging
        obj_val = Helper.geometric_median_objective(median, points, alphas)
        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append(log_entry)
        if verbose:
            logger.info('Starting Weiszfeld algorithm')
            logger.info(log_entry)
        logger.info(f'[rfa agg] init. name: {names}, weight: {alphas}')
        # start
        wv=None
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = torch.tensor([alpha / max(eps, Helper.l2dist(median, p)) for alpha, p in zip(alphas, points)],
                                 dtype=alphas.dtype)
            weights = weights / weights.sum()
            median = Helper.weighted_average_oracle(points, weights)
            num_oracle_calls += 1
            obj_val = Helper.geometric_median_objective(median, points, alphas)
            log_entry = [i + 1, obj_val,
                         (prev_obj_val - obj_val) / obj_val,
                         Helper.l2dist(median, prev_median)]
            logs.append(log_entry)
            if verbose:
                logger.info(f'[rfa agg] iter:  {i}, prev_obj_val: {prev_obj_val}, obj_val: {obj_val}, abs dis: { abs(prev_obj_val - obj_val)}')
                logger.info(f'[rfa agg] iter:  {i}, weight: {weights}')
            wv=copy.deepcopy(weights)
            if verbose:
                logger.info(log_entry)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        alphas = [Helper.l2dist(median, p) for p in points]

        update_norm = 0
        for name, data in median.items():
            update_norm += torch.sum(torch.pow(data, 2))
        update_norm= math.sqrt(update_norm)

        if max_update_norm is None or update_norm < max_update_norm:
            for name, data in target_model.state_dict().items():
                update_per_layer = median[name] * (self.params["eta"])
                if self.params['diff_privacy']:
                    update_per_layer.add_(self.dp_noise(data, self.params['sigma']))
                try:
                    data.add_(update_per_layer)
                except:
                    logger.info(f'layer name: {name}')
                    logger.info(f'data: {data}')
                    logger.info(f'update_per_layer: {update_per_layer}')
                    data.add_(update_per_layer.to(data.dtype))
                    logger.info(f'after update: {update_per_layer.to(data.dtype)}')
                # data.add_(update_per_layer)
            is_updated = True
        else:
            logger.info('\t\t\tUpdate norm = {} is too large. Update rejected'.format(update_norm))
            is_updated = False

        utils.csv_record.add_weight_result(names, wv.cpu().numpy().tolist(), alphas)

        return num_oracle_calls, is_updated, names, wv.cpu().numpy().tolist(),alphas

    @staticmethod
    def l2dist(p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        squared_sum = 0
        for name, data in p1.items():
            squared_sum += torch.sum(torch.pow(p1[name]- p2[name], 2))
        return math.sqrt(squared_sum)


    @staticmethod
    def geometric_median_objective(median, points, alphas):
        """Compute geometric median objective."""
        temp_sum= 0
        for alpha, p in zip(alphas, points):
            temp_sum += alpha * Helper.l2dist(median, p)
        return temp_sum

        # return sum([alpha * Helper.l2dist(median, p) for alpha, p in zip(alphas, points)])

    @staticmethod
    def weighted_average_oracle(points, weights):
        """Computes weighted average of atoms with specified weights

        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        tot_weights = torch.sum(weights)

        weighted_updates= dict()

        try:
            for name, data in points[0].items():
                weighted_updates[name] = torch.zeros_like(data)
        except:
            logger.info(f'[rfa agg] points[0]: {points[0]}')
        for name, data in points[0].items():
            weighted_updates[name]=  torch.zeros_like(data)
        for w, p in zip(weights, points): # 对每一个agent
            for name, data in weighted_updates.items():
                temp = (w / tot_weights).float().to(config.device)
                temp= temp* (p[name].float())
                # temp = w / tot_weights * p[name]
                if temp.dtype!=data.dtype:
                    temp = temp.type_as(data)
                data.add_(temp)

        return weighted_updates

    def weighted_sum_oracle(self, points, weights):
        """Computes weighted average of atoms with specified weights

        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """

        weighted_updates= dict()

        try:
            for name, data in points[0].items():
                weighted_updates[name] = torch.zeros_like(data)
        except:
            logger.info(f'[rfa agg] points[0]: {points[0]}')
        for name, data in points[0].items():
            weighted_updates[name]=  torch.zeros_like(data)
        for w, p in zip(weights, points): # 对每一个agent
            for name, data in weighted_updates.items():
                temp = w.float().to(config.device)
                temp= temp* (p[name].float())
                # temp = w / tot_weights * p[name]
                if temp.dtype!=data.dtype:
                    temp = temp.type_as(data)
                data.add_(temp)

        return weighted_updates

    def save_model(self, model=None, epoch=0, val_loss=0):
        if model is None:
            model = self.target_model
        if self.params['save_model']:
            # save_model
            logger.info("saving model")
            model_name = '{0}/model_last.pt.tar'.format(self.params['folder_path'])
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch,
                          'lr': self.params['lr']}
            self.save_checkpoint(saved_dict, False, model_name)
            if epoch in self.params['save_on_epochs']:
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss

    def update_epoch_submit_dict(self,epochs_submit_update_dict,global_epochs_submit_dict, epoch,state_keys):

        epoch_len= len(epochs_submit_update_dict[state_keys[0]])
        for j in range(0, epoch_len):
            per_epoch_dict = dict()
            for i in range(0, len(state_keys)):
                local_model_update_list = epochs_submit_update_dict[state_keys[i]]
                local_model_update_dict = local_model_update_list[j]
                per_epoch_dict[state_keys[i]]= local_model_update_dict

            global_epochs_submit_dict[epoch+j]= per_epoch_dict

        return global_epochs_submit_dict


    def save_epoch_submit_dict(self, global_epochs_submit_dict):
        with open(f'{self.folder_path}/epoch_submit_update.json', 'w') as outfile:
            json.dump(global_epochs_submit_dict, outfile, ensure_ascii=False, indent=1)

    def estimate_fisher(self, model, criterion,
                        data_loader, sample_size, batch_size=64):
        # sample loglikelihoods from the dataset.
        loglikelihoods = []
        if self.params['type'] == 'text':
            data_iterator = range(0, data_loader.size(0) - 1, self.params['bptt'])
            hidden = model.init_hidden(self.params['batch_size'])
        else:
            data_iterator = data_loader

        for batch_id, batch in enumerate(data_iterator):
            data, targets = self.get_batch(data_loader, batch,
                                           evaluation=False)
            if self.params['type'] == 'text':
                hidden = self.repackage_hidden(hidden)
                output, hidden = model(data, hidden)
                loss = criterion(output.view(-1, self.n_tokens), targets)
            else:
                output = model(data)
                loss = log_softmax(output, dim=1)[range(targets.shape[0]), targets.data]
                # loss = criterion(output.view(-1, ntokens
            # output, hidden = model(data, hidden)
            loglikelihoods.append(loss)
            # loglikelihoods.append(
            #     log_softmax(output.view(-1, self.n_tokens))[range(self.params['batch_size']), targets.data]
            # )

            # if len(loglikelihoods) >= sample_size // batch_size:
            #     break
        logger.info(loglikelihoods[0].shape)
        # estimate the fisher information of the parameters.
        loglikelihood = torch.cat(loglikelihoods).mean(0)
        logger.info(loglikelihood.shape)
        loglikelihood_grads = torch.autograd.grad(loglikelihood, model.parameters())

        parameter_names = [
            n.replace('.', '__') for n, p in model.named_parameters()
        ]
        return {n: g ** 2 for n, g in zip(parameter_names, loglikelihood_grads)}

    def consolidate(self, model, fisher):
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            model.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
            model.register_buffer('{}_estimated_fisher'
                                  .format(n), fisher[n].data.clone())

    def ewc_loss(self, model, lamda, cuda=False):
        try:
            losses = []
            for n, p in model.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(model, '{}_estimated_mean'.format(n))
                fisher = getattr(model, '{}_estimated_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p - mean) ** 2).sum())
            return (lamda / 2) * sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )

class FoolsGold(object):
    def __init__(self, use_memory=False):
        self.memory = None
        self.memory_dict=dict()
        self.wv_history = []
        self.use_memory = use_memory

    def aggregate_gradients(self, client_grads,names):
        cur_time = time.time()
        num_clients = len(client_grads)
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()

        # if self.memory is None:
        #     self.memory = np.zeros((num_clients, grad_len))
        self.memory = np.zeros((num_clients, grad_len))
        grads = np.zeros((num_clients, grad_len))
        for i in range(len(client_grads)):
            grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]]+=grads[i]
            else:
                self.memory_dict[names[i]]=copy.deepcopy(grads[i])
            self.memory[i]=self.memory_dict[names[i]]
        # self.memory += grads

        if self.use_memory:
            wv, alpha = self.foolsgold(self.memory)  # Use FG
        else:
            wv, alpha = self.foolsgold(grads)  # Use FG
        logger.info(f'[foolsgold agg] wv: {wv}')
        self.wv_history.append(wv)

        agg_grads = []
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
            temp = wv[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == 0:
                    continue
                temp += wv[c] * client_grad[i].cpu()
            temp = temp / len(client_grads)
            agg_grads.append(temp)
        print('model aggregation took {}s'.format(time.time() - cur_time))
        return agg_grads, wv, alpha

    def foolsgold(self,grads):
        """
        :param grads:
        :return: compute similatiry and return weightings
        """
        n_clients = grads.shape[0]
        cs = smp.cosine_similarity(grads) - np.eye(n_clients)

        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))

        wv[wv > 1] = 1
        wv[wv < 0] = 0

        alpha = np.max(cs, axis=1)

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        # wv is the weight
        return wv,alpha



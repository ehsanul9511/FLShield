from re import I
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
import train
import test

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

import hdbscan
from binarytree import Node
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx

from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from matplotlib import pyplot as plt

class FCLUSTER:
    def __init__(self) -> None:
        super().__init__()
        self.child = None
        self.stability = 0.
        pass

def calc_clusters(node, min_samples, grads=None, stability=None, adv_list=None):
    cluster = FCLUSTER()
    cluster.node = node
    stability_of_child = None
    if grads is not None:
        labels=[]
        grads_subset = grads[node.points]
        for idx, point in enumerate(node.points):
            if point in node.left.points:
                labels.append(0)
            elif point in node.right.points:
                labels.append(1)
            else:
                logger.info(f'error log: {point} {node.points} {node.left.points} {node.right.points}')
        stability_of_child = silhouette_score(grads_subset, labels)
    if len(node.left.points) >= min_samples:
        cluster.child = calc_clusters(node.left, min_samples, grads, stability_of_child, adv_list)
    if len(node.right.points) >= min_samples:
        cluster.child = calc_clusters(node.right, min_samples, grads, stability_of_child, adv_list)
    if stability is not None:
        cluster.stability = stability
    else:
        cluster.stability = 0.
    logger.info(f'cluster created with stability {cluster.stability}, mal count {len([point for point in node.points if point in adv_list])} with left {len(node.left.points)} and right {len(node.right.points)}')
    return cluster

def find_best_cluster(current_cluster):
    if current_cluster.child is None:
        return current_cluster
    else:
        child_cluster = find_best_cluster(current_cluster.child)
        if child_cluster.stability > current_cluster.stability:
            return child_cluster
        else:
            return current_cluster

def calc_edge_weight(node):
    if node.edge_weight is not None:
        edge_weight = node.edge_weight
    else:
        edge_weight = 0
    if node.left is not None:
        edge_weight += calc_edge_weight(node.left)
    if node.right is not None:
        edge_weight += calc_edge_weight(node.right)
    return edge_weight

def modHDBSCAN(grads, min_samples, adv_list):
    # clusterer = hdbscan.HDBSCAN(gen_min_span_tree=True)
    # clusterer.fit(grads)

    # mst = clusterer.minimum_spanning_tree_.to_numpy()
    cos_matrix = cosine_distances(grads)
    mst = minimum_spanning_tree(cos_matrix)
    graph = nx.from_scipy_sparse_matrix(mst)
    edge_list = list(nx.convert.to_edgelist(graph))
    edge_list = sorted(edge_list, key=lambda x: x[2]['weight'])
    # logger.info(f'edge_list: {edge_list}')
    # logger.info(f'MST : {mst}')

    leaf_nodes = []
    parent_dict = {}

    for i in range(len(grads)):
        node = Node(i)
        node.points = [i]
        node.edge_weight = 0
        leaf_nodes.append(node)
        parent_dict[i] = node

    last_node = None
    for edge_idx in range(len(edge_list)):
        u, v, w= edge_list[edge_idx]
        w = w['weight']
        try:
            u_node = parent_dict[u]
            v_node = parent_dict[v]
        except:
            print(parent_dict.keys())
        node = Node(f'({int(u)}, {int(v)})')
        node.points = u_node.points + v_node.points
        node.left = u_node
        node.right = v_node
        node.edge_weight = w
        for point in node.points:
            parent_dict[point] = node
        last_node = node
        # print(node.points)

    top_cluster = calc_clusters(last_node, min_samples, grads, adv_list=adv_list)
    best_cluster = find_best_cluster(top_cluster)

    wv = np.zeros(len(grads))
    for i in best_cluster.node.points:
        wv[i] = 1
    return wv, best_cluster.node.points, edge_list, top_cluster

class Helper:
    def __init__(self, current_time, params, name):
        self.current_time = current_time
        self.target_model = None
        self.local_model = None

        self.train_data = None
        self.test_data = None
        self.poisoned_data = None
        self.test_data_poison = None

        self.params = params
        self.name = name
        self.best_loss = math.inf

        if self.params['attack_methods'] in [config.ATTACK_TLF, config.ATTACK_SIA]:
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
        if 'src_grp_mal' in self.params.keys():
            self.src_grp_mal = self.params['src_grp_mal']
        else:
            self.src_grp_mal = 4

        self.folder_path = f'saved_models/model_{self.name}_{current_time}_no_models_{self.params["no_models"]}'
        # self.folder_path = f'saved_models/model_{self.name}_{current_time}_targetclass_{self.params["tlf_label"]}_no_models_{self.params["no_models"]}'
        if 'save_data' in self.params.keys() and 'camera_ready' in self.params.keys() and self.params['camera_ready']:
            # self.folder_path = f'outputs/{self.name}/{self.params["attack_methods"]}/total_mal_{self.num_of_adv}/hardness_{self.params["tlf_label"]}/src_grp_mal_{self.src_grp_mal}/aggr_{self.params["aggregation_methods"]}/no_models_{self.params["no_models"]}/distrib_var_{self.params["save_data"]}'
            # if self.params['attack_methods'] == config.ATTACK_SIA and self.params['new_adaptive_attack']:
            #     self.folder_path = f'outputs/{self.name}/{self.params["attack_methods"]}/total_mal_{self.num_of_adv}/hardness_{self.params["tlf_label"]}/src_grp_mal_{self.src_grp_mal}/aggr_{self.params["aggregation_methods"]}/new_adaptive_attack_alpha_{self.params["alpha_loss"]}/no_models_{self.params["no_models"]}/distrib_var_{self.params["save_data"]}'
            if self.params['is_poison']:
                self.folder_path = f'outputs/{self.name}_{self.params["attack_methods"]}_{self.num_of_adv}_{self.params["tlf_label"]}_{self.src_grp_mal}_{self.params["aggregation_methods"]}_{self.params["no_models"]}_{self.params["save_data"]}'
                if self.params['attack_methods'] == config.ATTACK_SIA and self.params['new_adaptive_attack']:
                    self.folder_path = f'outputs/{self.name}_{self.params["attack_methods"]}_{self.num_of_adv}_{self.params["tlf_label"]}_{self.src_grp_mal}_{self.params["aggregation_methods"]}_{self.params["no_models"]}_{self.params["alpha_loss"]}_{self.params["save_data"]}'
                elif self.params['attack_methods'] == config.ATTACK_DBA and 'scaling_attack' in self.params.keys() and self.params['scaling_attack']:
                    self.folder_path = f'outputs/{self.name}_scaling_{self.num_of_adv}_{self.params["tlf_label"]}_{self.src_grp_mal}_{self.params["aggregation_methods"]}_{self.params["no_models"]}_{self.params["save_data"]}'
            else:
                self.folder_path = f'outputs/{self.name}_nopoison_{self.params["tlf_label"]}_{self.params["aggregation_methods"]}_{self.params["no_models"]}_{self.params["save_data"]}'
        elif 'ablation_study' in self.params.keys():
            self.folder_path = f'outputs/ablation_study/{self.name}_{self.params["ablation_study"]}_{self.params["no_models"]}'
        else:
            self.folder_path = f'saved_models/model_{self.name}_{current_time}'
        # try:
        #     os.mkdir(self.folder_path)
        # except FileExistsError:
        #     logger.info('Folder already exists')
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
    
    def get_optimal_k_for_clustering(self, grads, clustering_params='grads', clustering_method='Agglomerative'):
        coses = []
        nets = grads
        # nets = [grad.numpy() for grad in grads]
        # nets = [np.array(grad) for grad in grads]
        # for i1, net1 in enumerate(nets):
        #     coses_l=[]
        #     for i2, net2 in enumerate(nets):
        #         coses_l.append(1-cos_calc_btn_grads(net1.grad_params, net2.grad_params))
        #     coses.append(coses_l)
        coses = cosine_distances(nets, nets)
        coses = np.array(coses)
        np.fill_diagonal(coses, 0)
        # logger.info(f'coses: {coses}')
        
        sil= []
        if clustering_params=='grads':
            minval = 2
        else:
            minval = 9
        for k in range(minval, min(len(nets), 15)):
            if clustering_method=='Agglomerative':
                clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='complete').fit(coses)
            elif clustering_method=='KMeans':
                clustering = KMeans(n_clusters=k).fit(nets)
            elif clustering_method=='Spectral':
                clustering = SpectralClustering(n_clusters=k, affinity='precomputed').fit(coses)
            labels = clustering.labels_
            # print(labels)
            sil.append(silhouette_score(coses, labels, metric='precomputed'))
        # print(sil)
        # logger.info(f'Silhouette scores: {sil}')
        return sil.index(max(sil))+minval, coses

    def recalculate_val_trust_scores(self, grads):
        clusters = self.clusters
        X=grads
        grads_for_clusters = []       
        for cluster in clusters:
            grads = [X[i] for i in cluster]
            grads_for_clusters.append(grads)

        if 'ablation_study' in self.params.keys() and 'dynamic_distance' in self.params['ablation_study']:
            dynamic_dist_array = []
            for i, cluster in enumerate(clusters):
                distances = [self.get_average_distance(X[cl], grads_for_clusters[i]) for cl in cluster]
                distances_dict = {cl: dist for cl, dist in zip(cluster, distances)}
                cluster.sort(key=lambda x: distances_dict[x])
                sorted(distances)
                distances_gap = [distances[i+1]-distances[i] for i in range(len(distances)-1)]
                dynamic_dist_idx = distances_gap.index(max(distances_gap)) + 1
                if 'fixed' in self.params['ablation_study']:
                    dynamic_dist_idx = 5
                for idx, cluster_elem in enumerate(clusters[i]):
                    if idx>=dynamic_dist_idx:
                        if 'no_dist' in self.params['ablation_study']:
                            self.validator_trust_scores[cluster_elem] = min(self.validator_trust_scores[cluster_elem], 1.)
                        else:
                            self.validator_trust_scores[cluster_elem] = min(self.validator_trust_scores[cluster_elem], 1/idx)
                dynamic_dist_array.append(dynamic_dist_idx)

            logger.info(f'Validator Trust Scores')
            for cluster in clusters:
                logger.info([f'{elem} ({elem in self.adversarial_namelist}): {self.validator_trust_scores[elem]}' for elem in cluster])
            if len(utils.csv_record.dynamic_dist_fileheader) == 0:
                utils.csv_record.dynamic_dist_fileheader = np.arange(len(clusters)).tolist()
            utils.csv_record.dynamic_dist_result.append(dynamic_dist_array)
            return


        for i, cluster in enumerate(clusters):
            cluster.sort(key = lambda x: self.get_average_distance(X[x], grads_for_clusters[i]))
            # clusters[i] = cluster[:5]
            for idx, cluster_elem in enumerate(clusters[i]):
                if idx>=5:
                    self.validator_trust_scores[cluster_elem] = min(self.validator_trust_scores[cluster_elem], 1/idx)
        return

    def cluster_grads(self, grads, clustering_method='Spectral', clustering_params='grads', k=10):
        # nets = [grad.numpy() for grad in grads]
        # nets = [np.array(grad) for grad in grads]
        nets = grads
        if clustering_params=='lsrs':
            X = self.lsrs
        elif clustering_params=='grads':
            X = nets

        if clustering_method == 'Spectral':
            if clustering_params == 'grads':
                k, coses = self.get_optimal_k_for_clustering(grads, clustering_params, clustering_method)
                clustering = SpectralClustering(n_clusters=k, affinity='precomputed').fit(coses)
            else:
                k, coses = self.get_optimal_k_for_clustering(X, clustering_params, clustering_method)
                # if self.params['type'] != config.TYPE_LOAN:
                #     k = 10
                clustering = SpectralClustering(n_clusters=k, affinity='cosine').fit(X)
        elif clustering_method == 'Agglomerative':
            # anomaly removal enabled
            anoamly_removal = False
            if anoamly_removal:
                anomaly_arr = EllipticEnvelope(contamination=0.5).fit_predict(X)
                non_anomalies = [i for i in range(len(X)) if anomaly_arr[i] != -1]
                k, coses = self.get_optimal_k_for_clustering(X[non_anomalies], clustering_params, clustering_method)
                # print(k)
                clustering = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='complete').fit(X[non_anomalies])

                cluster_label = [clustering.labels_[non_anomalies.index(i)] if anomaly_arr[i] !=-1 else -1 for i in range(len(anomaly_arr))]
                cluster_label = np.array(cluster_label)
                clusters = [np.argwhere(cluster_label == i)[:,0] for i in range(np.max(cluster_label)+1)]
                clusters.sort(key=lambda x: len(x), reverse=True)

                return cluster_label, clusters
            
            k, coses = self.get_optimal_k_for_clustering(X, clustering_params, clustering_method)
            clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='complete').fit(coses)
            
        elif clustering_method == 'KMeans':
            k, coses = self.get_optimal_k_for_clustering(grads, clustering_params, clustering_method)
            clustering = KMeans(n_clusters=k).fit(X)

        clusters = [[] for _ in range(k)]
        for i, label in enumerate(clustering.labels_.tolist()):
            clusters[label].append(i)
        for cluster in clusters:
            cluster.sort()
        clusters.sort(key = lambda cluster: len(cluster), reverse = True)
        logger.info(f'Validator Groups: {clusters}')

        # grads_for_clusters = []
        # for cluster in clusters:
        #     grads = [X[i] for i in cluster]
        #     grads_for_clusters.append(grads)
            
        # for i, cluster in enumerate(clusters):
        #     cluster.sort(key = lambda x: self.get_validation_score(X[x], grads_for_clusters[i]))


        if clustering_params=='lsrs' and False: 
            grads_for_clusters = []       
            for cluster in clusters:
                grads = [X[i] for i in cluster]
                grads_for_clusters.append(grads)

            print('clusters ', clusters)

            for i, cluster in enumerate(clusters):
                cluster.sort(key = lambda x: self.get_average_distance(X[x], grads_for_clusters[i]))
                # clusters[i] = cluster[:5]
                for idx, cluster_elem in enumerate(clusters[i]):
                    if idx>=5:
                        self.validator_trust_scores[cluster_elem] = 1/idx
            print('clusters ', clusters)

        return clustering.labels_, clusters

    def validation_test_for_loan(self, network, test_loader, is_poisonous=False, adv_index=-1, tqdm_disable=True):
        network.eval()
        correct = 0
        correct_by_class = {}

        dataset_size_by_classes = {}
        for cl in range(9):
            dataset_size_by_classes[cl] = 0
            correct_by_class[cl] = 0

        with torch.no_grad():
            for batch_id, batch in enumerate(test_loader):
                if is_poisonous:
                    for index in range(0, len(batch[1])):
                        if batch[1][index] == self.source_class:
                            batch[1][index] = self.target_class
                data, targets = self.allStateHelperList[adv_index].get_batch(test_loader, batch, evaluation=True)
                output = network(data)
                # total_loss += nn.functional.cross_entropy(output, targets,
                #                                           reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
                # checking attack success rate
                for cl in range(9):
                    class_indices = np.where(targets.cpu().data.numpy()==cl)
                    dataset_size_by_classes[cl] += len(class_indices[0])
                    correct_by_class[cl] += pred.eq(targets.data.view_as(pred)).cpu().data.numpy()[class_indices].sum()

            for cl in range(9):
                if dataset_size_by_classes[cl] == 0:
                    correct_by_class[cl] = 100. * correct

            for class_label in dataset_size_by_classes.keys():
                if dataset_size_by_classes[class_label] != 0:
                    correct_by_class[class_label] = 100. * correct_by_class[class_label]/ dataset_size_by_classes[class_label]
                    correct_by_class[class_label] = correct_by_class[class_label].item()
            # print(correct_by_class)
        return 100. * correct / len(test_loader.dataset), correct_by_class        

    def validation_test(self, network, test_loader, is_poisonous=False, adv_index=-1, tqdm_disable=True):
        network.eval()
        correct = 0
        correct_by_class = {}

        dataset_classes = {}
        validation_dataset = test_loader.dataset

        for ind, x in enumerate(validation_dataset):
            _, label = x
            #if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
            #    continue
            if label in dataset_classes:
                dataset_classes[label].append(ind)
            else:
                dataset_classes[label] = [ind]

        with torch.no_grad():
            for data, target in tqdm(test_loader, disable=tqdm_disable):
                if is_poisonous:
                    data, target, poison_num = self.get_poison_batch((data, target), adv_index)
                else:
                    data, target = self.get_batch(None, (data, target))
                output = network(data)
                loss_func=torch.nn.CrossEntropyLoss()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()

            for class_label in dataset_classes.keys():
                correct_by_class[class_label] = 0
                one_class_test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=100, sampler=SubsetRandomSampler(indices=dataset_classes[class_label]))

                # for data, target in tqdm(one_class_test_loader, disable=tqdm_disable):
                for data, target in one_class_test_loader:
                    if is_poisonous:
                        data, target, poison_num = self.get_poison_batch((data, target), adv_index)
                    else:
                        data, target = self.get_batch(None, (data, target))
                    output = network(data)
                    loss_func=torch.nn.CrossEntropyLoss()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct_by_class[class_label] += pred.eq(target.data.view_as(pred)).sum()

                correct_by_class[class_label] = 100. * correct_by_class[class_label]/ len(dataset_classes[class_label])
                correct_by_class[class_label] = correct_by_class[class_label].item()

            for c in range(10):
                if c not in correct_by_class:
                    correct_by_class[c] = 100. * correct / len(test_loader.dataset)
                    correct_by_class[c] = correct_by_class[c].item()
            # print(correct_by_class)
        return 100. * correct / len(test_loader.dataset), correct_by_class

    def print_util(self, a, b):
        return str(a) + ': ' + str(b)

    def validation_test_v2(self, network, given_test_loader, is_poisonous=False, adv_index=-1, tqdm_disable=True, num_classes=10):
        network.eval()
        correct = 0
        correct_by_class = {}
        loss_by_class = {}
        loss_by_class_per_example = {}
        count_per_class = {}
        loss = 0.

        dataset_classes = {}
        validation_dataset = copy.deepcopy(given_test_loader.dataset)
        test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=len(validation_dataset))

        for c in range(num_classes):
            count_per_class[c] = 0
            loss_by_class[c] = []
            loss_by_class_per_example[c] = 0.
            correct_by_class[c] = []
        
        with torch.no_grad():
            for batch_id, batch in enumerate(test_loader):
                if is_poisonous and self.params['attack_methods']==config.ATTACK_TLF and False:
                    data, targets, _ = self.get_poison_batch_for_targeted_label_flip(batch)
                else:
                    data, targets = self.get_batch(None, batch)
                output = network(data)
                loss_func=torch.nn.CrossEntropyLoss(reduction='none')
                pred = output.data.max(1, keepdim=True)[1]
                correct_array = pred.eq(targets.data.view_as(pred))
                correct += correct_array.sum()
                loss_array = loss_func(output, targets)
                loss += loss_array.sum().item()
                class_indices = {}
                for cl in range(num_classes):
                    class_indices[cl] = (targets==cl)
                    count_per_class[cl] += (class_indices[cl]).sum().item()

                    # loss_by_class[cl] += loss_array[class_indices[cl]].sum().item()
                    # correct_by_class[cl] += correct_array[class_indices[cl]].sum().item()     
                    loss_by_class[cl] += [loss_val.item() for loss_val in loss_array[class_indices[cl]]]
                    correct_by_class[cl] += [correct_val.item() for correct_val in correct_array[class_indices[cl]]]
                
        for class_label in range(num_classes):
            cap_on_per_class = True
            if count_per_class[class_label] > 30 and cap_on_per_class:
                count_per_class[class_label] = 30
                loss_by_class[class_label] = loss_by_class[class_label][:30]
                correct_by_class[class_label] = correct_by_class[class_label][:30]

            loss_by_class[class_label] = np.sum(loss_by_class[class_label])
            correct_by_class[class_label] = np.sum(correct_by_class[class_label])
            
            if count_per_class[class_label] == 0:
                correct_by_class[class_label] = 0
                loss_by_class[class_label] = 0.
                loss_by_class_per_example[class_label] = np.nan
            else:
                correct_by_class[class_label] = 100. * correct_by_class[class_label]/ count_per_class[class_label]
                loss_by_class_per_example[class_label] = loss_by_class[class_label]/ count_per_class[class_label]

            # try:
            #     correct_by_class[class_label] = 100. * correct_by_class[class_label]/ count_per_class[class_label]
            # except:
            #     correct_by_class[class_label] = 0.
            #     pass


            # try:
            #     loss_by_class_per_example[class_label] = loss_by_class[class_label]/ count_per_class[class_label]
            # except:
            #     loss_by_class_per_example[class_label] = 0.
            #     # loss_by_class_per_example[class_label] = loss / len(test_loader.dataset)
            #     pass

        return 100. * correct / len(test_loader.dataset), loss_by_class, loss_by_class_per_example, count_per_class



    def validation_test_v3(self, network, test_loader, is_poisonous=False, adv_index=-1, tqdm_disable=True, num_classes=10):
        network.eval()
        correct = 0
        correct_by_class = {}
        loss_by_class = {}
        loss_by_class_per_example = {}
        count_per_class = {}
        loss = 0.

        dataset_classes = {}
        validation_dataset = copy.deepcopy(test_loader.dataset)
        val_dataset = []

        #poison validation dataset
        # if is_poisonous and self.params['attack_methods'] == config.ATTACK_TLF:
        #     for ind, (x, y) in enumerate(validation_dataset):
        #         if y == self.source_class:
        #             val_dataset.append((x, self.target_class))
        #         else:
        #             val_dataset.append((x, y))

        #     validation_dataset = val_dataset

        for ind, (x, y) in enumerate(validation_dataset):
            if is_poisonous and self.params['attack_methods'] == config.ATTACK_TLF and y == self.source_class:
                val_dataset.append((x, self.target_class))
            else:
                val_dataset.append((x, y))

        validation_dataset = val_dataset

        for ind, x in enumerate(validation_dataset):
            _, label = x
            #if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
            #    continue
            if label in dataset_classes:
                dataset_classes[label].append(ind)
            else:
                dataset_classes[label] = [ind]

        with torch.no_grad():
            for data, target in tqdm(test_loader, disable=tqdm_disable):
                data, target = self.get_batch(None, (data, target))
                output = network(data)
                loss_func=torch.nn.CrossEntropyLoss(reduction='sum')
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                loss += loss_func(output, target).item()

            loss = loss / len(test_loader.dataset)

            for class_label in dataset_classes.keys():
                correct_by_class[class_label] = 0
                loss_by_class[class_label] = 0
                one_class_test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=100, sampler=SubsetRandomSampler(indices=dataset_classes[class_label]))

                # for data, target in tqdm(one_class_test_loader, disable=tqdm_disable):
                for data, target in one_class_test_loader:
                    data, target = self.get_batch(None, (data, target))
                    output = network(data)
                    loss_func=torch.nn.CrossEntropyLoss(reduction = 'sum')
                    pred = output.data.max(1, keepdim=True)[1]
                    correct_by_class[class_label] += pred.eq(target.data.view_as(pred)).sum()
                    loss_by_class[class_label] += loss_func(output, target, ).item()
                    

                correct_by_class[class_label] = 100. * correct_by_class[class_label]/ len(dataset_classes[class_label])
                correct_by_class[class_label] = correct_by_class[class_label].item()

                loss_by_class_per_example[class_label] = loss_by_class[class_label]/ len(dataset_classes[class_label])
                count_per_class[class_label] = len(dataset_classes[class_label])

            for c in range(10):
                if c not in correct_by_class:
                    correct_by_class[c] = 100. * correct / len(test_loader.dataset)
                    correct_by_class[c] = correct_by_class[c].item()

                if c not in loss_by_class:
                    loss_by_class[c] = 0.
                    loss_by_class_per_example[c] = 0.
                    count_per_class[c] = 0.
            # print(correct_by_class)
        return 100. * correct / len(test_loader.dataset), loss_by_class, loss_by_class_per_example, count_per_class

    def mal_pcnt(self, cluster, names, wv=None):
        mal_count = 0
        for idx, client_id in enumerate(cluster):
            # if names[client_id] in self.adversarial_namelist:
            if client_id in self.adversarial_namelist:
                if wv is None:
                    mal_count += 1
                else:
                    mal_count += wv[idx]
        if wv is None:
            return mal_count / len(cluster)
        else:
            return mal_count

    # def combined_clustering_guided_aggregation_with_DP(self, target_model, updates, epoch):
    #     client_grads = []
    #     alphas = []
    #     names = []
    #     delta_models = []
    #     for name, data in updates.items():
    #         client_grads.append(data[1])  # gradient
    #         alphas.append(data[0])  # num_samples
    #         delta_models.append(data[2])
    #         names.append(name)
    #     grads = [self.flatten_gradient(client_grad) for client_grad in client_grads]

    #     sum_grads = np.sum(grads, axis=0)

    #     # norms = [np.linalg.norm(grad) for grad in grads]
    #     norm_median = np.linalg.norm(sum_grads)
    #     noise_level = self.params['sigma'] * norm_median

    #     mean_model = self.new_model()
    #     mean_model.copy_params(self.target_model.state_dict())

    #     self.fedavg(mean_model, updates)

    #     test_models = []

    #     for i in range(1, 11):
    #         test_models.append(self.new_model())
    #         test_models[i-1].copy_params(mean_model.state_dict())
    #         self.add_noise(test_models[i-1], i * noise_level / 10)

    #     print(f'Validating all clients at epoch {epoch}')

    #     all_validator_evaluations = {}
    #     evaluations_of_clusters = {}
    #     count_of_class_for_validator = {}

    #     for name in names:
    #         all_validator_evaluations[name] = []

    #     evaluations_of_clusters[-1] = {}
    #     for iidx, val_idx in enumerate(names):

    #         if self.params['type'] == config.TYPE_LOAN:
    #             val_test_loader = self.allStateHelperList[val_idx].get_testloader()
    #         else:
    #             _, val_test_loader = self.train_data[val_idx]
    #         if val_idx in self.adversarial_namelist:
    #             is_poisonous_validator = True
    #         else:
    #             is_poisonous_validator = False
    #         if self.params['type'] == config.TYPE_LOAN:
    #             val_acc, val_acc_by_class = self.validation_test_for_loan(target_model, val_test_loader, is_poisonous_validator, adv_index=0)
    #         else:
    #             if 'ablation_study' in self.params.keys() and 'no_wrong_validation' in self.params['ablation_study']:
    #                 is_poisonous_validator = False
    #             val_acc, val_loss, val_acc_by_class, count_of_class = self.validation_test_v2(target_model, val_test_loader, is_poisonous=is_poisonous_validator, adv_index=0)

    #         val_acc_by_class = [val_acc_by_class[i] for i in range(10)]
    #         all_validator_evaluations[val_idx] += val_acc_by_class
    #         evaluations_of_clusters[-1][val_idx] = [val_loss[i] for i in range(10)]
    #         if val_idx not in count_of_class_for_validator.keys():
    #             count_of_class_for_validator[val_idx] = count_of_class

    #     fail_idx = -1
    #     for idx in range(len(test_models)):
    #         evaluations_of_clusters[idx] = {}
    #         agg_model = test_models[idx]
    #         for iidx, val_idx in enumerate(names):
    #             if self.params['type'] == config.TYPE_LOAN:
    #                 val_test_loader = self.allStateHelperList[val_idx].get_testloader()
    #             else:
    #                 _, val_test_loader = self.train_data[val_idx]
    #             if val_idx in self.adversarial_namelist:
    #                 is_poisonous_validator = True
    #             else:
    #                 is_poisonous_validator = False
    #             if self.params['type'] == config.TYPE_LOAN:
    #                 val_acc, val_acc_by_class = self.validation_test_for_loan(agg_model, val_test_loader, is_poisonous_validator, adv_index=0)
    #             else:
    #                 if 'ablation_study' in self.params.keys() and 'no_wrong_validation' in self.params['ablation_study']:
    #                     is_poisonous_validator = False

    #                 try:
    #                     val_acc, val_loss, val_acc_by_class, _ = self.validation_test_v2(agg_model, val_test_loader, is_poisonous=is_poisonous_validator, adv_index=0)
    #                 except:
    #                     fail_idx = idx
    #                     break

    #             val_acc_by_class = [-val_acc_by_class[i]+all_validator_evaluations[val_idx][i] for i in range(10)]
                    
    #             all_validator_evaluations[val_idx]+= val_acc_by_class
    #             evaluations_of_clusters[idx][val_idx] = [-val_loss[i]+evaluations_of_clusters[-1][val_idx][i] for i in range(10)]
            
    #         if fail_idx != -1:
    #             break
    #     if fail_idx != -1:
    #         test_models = test_models[:fail_idx]
            

    #     if False:
    #         def get_weighted_average(points, alpha):
    #             alpha = alpha / np.sum(alpha)
    #             for i in range(len(points)):
    #                 points[i] = points[i] * alpha[i]
    #             return np.sum(points, axis=0)

    #         def geo_median_objective(median, points, alphas):
    #             temp_sum= 0
    #             for alpha, p in zip(alphas, points):
    #                 temp_sum += alpha * distance.euclidean(median, p)
    #             return temp_sum
    #     else:
    #         validator_xs = [all_validator_evaluations[name] for name in names]
    #         _, val_clustering = self.cluster_grads(validator_xs, clustering_params='grads', clustering_method='KMeans')
    #         good_cluster = np.argmax([len(cluster) for cluster in val_clustering])
    #         logger.info([self.mal_pcnt(cluster, names) for cluster in val_clustering])
    #         logger.info(f'good cluster: {val_clustering[good_cluster]}')

    #         remaining_validators = []

    #         for idx, name in enumerate(names):
    #             if name in val_clustering[good_cluster]:
    #                 remaining_validators.append(name)
    #             else:
    #                 for idx in range(len(test_models)):
    #                     try:
    #                         del evaluations_of_clusters[idx][name]
    #                     except:
    #                         pass

    #         wv_by_cluster = []

    #         total_count_of_class = [0 for _ in range(10)]
    #         for val_idx in remaining_validators:
    #             total_count_of_class = [total_count_of_class[i]+ count_of_class_for_validator[val_idx][i] for i in range(10)]

    #         logger.info(f'total_count_of_class: {total_count_of_class}')

    #         for idx in range(len(test_models)):
    #             all_vals = [evaluations_of_clusters[idx][name] for name in remaining_validators]
    #             all_vals = np.array(all_vals)
    #             all_vals = np.transpose(all_vals)
    #             all_vals = all_vals.tolist()


    #             eval_sum_of_cluster = [np.sum(all_vals[i]) for i in range(len(all_vals))]

    #             class_by_class_evaluation = False
    #             if class_by_class_evaluation:
    #                 eval_mean_of_cluster = [eval_sum_of_cluster[i]/total_count_of_class[i] for i in range(len(eval_sum_of_cluster))]
    #                 wv_by_cluster.append(np.min(eval_mean_of_cluster))
    #             else:
    #                 eval_mean_of_cluster = np.sum(eval_sum_of_cluster)/np.sum(total_count_of_class)
    #                 wv_by_cluster.append(eval_mean_of_cluster)
    #             # eval_mean_of_cluster = eval_mean_of_cluster[10:]
    #             # eval_mean_of_cluster = eval_mean_of_cluster.reshape(len(eval_mean_of_cluster)//10, 10)
    #             # eval_mean_of_cluster = np.mean(eval_mean_of_cluster, axis=0)
    #             evaluations_of_clusters[idx] = eval_mean_of_cluster
    #             logger.info(f'test_model {idx} performance: {eval_mean_of_cluster}')

    #     logger.info(f'evaluations_of_clusters: {wv_by_cluster}')
    #     non_negative_idx = [i for i in range(len(wv_by_cluster)) if wv_by_cluster[i] >= 0]

    #     if len(non_negative_idx) == 0:
    #         non_negative_idx = [0]
    #     best_test_model = np.max(non_negative_idx)
    #     target_model.copy_params(test_models[best_test_model].state_dict())
    #     # logger.info(f'wv_by_cluster: {wv_by_cluster}')
    #     # med_wv = np.median(wv_by_cluster)
    #     # wv_by_cluster = [1 if z>=med_wv else 0 for z in wv_by_cluster]
    #     # for cluster_idx in range(num_of_clusters):
    #     #     logger.info(f'cluster {cluster_idx} with mal_pcnt {mal_pcnt_by_cluster[cluster_idx]} performance: {wv_by_cluster[cluster_idx]}')
    #     # logger.info(f'wv_by_cluster updated: {wv_by_cluster}')
 
    #     return


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
        
        wv = np.zeros(len(names), dtype=np.float32)
        grads = [self.flatten_gradient(client_grad) for client_grad in client_grads]
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

            # grads = np.delete(grads, slice(-5010,0), axis=1)
        logger.info(f'Converted gradients to param list: Time: {time.time() - t}')

        no_clustering = False

        if self.params['no_models'] < 10 or no_clustering:
            exclude_mode = False

            if exclude_mode:
                self.clusters_agg = [[j for j in range(len(names)) if j != i] for i in range(len(names))]
            else:
                self.clusters_agg = [[i] for i in range(self.params['no_models'])]
        else:
            if 'ablation_study' in self.params.keys() and 'clustering_kmeans' in self.params['ablation_study']:
                _, self.clusters_agg = self.cluster_grads(grads, clustering_method='KMeans', clustering_params='grads', k=10)
            elif 'ablation_study' in self.params.keys() and 'clustering_spectral' in self.params['ablation_study']:
                _, self.clusters_agg = self.cluster_grads(grads, clustering_method='Spectral', clustering_params='grads', k=10)
            else:
                _, self.clusters_agg = self.cluster_grads(grads, clustering_method='Agglomerative', clustering_params='grads', k=10)

                # noise_removed_clusters = []
                # for cluster in self.clusters_agg:
                #     anomaly_arr = EllipticEnvelope(contamination = 0.3).fit_predict(np.array(grads)[cluster])
                #     # anomaly_arr = LocalOutlierFactor(n_neighbors = 3).fit_predict(np.array(grads)[cluster])
                #     non_anomalies = [cluster[i] for i in range(len(cluster)) if anomaly_arr[i] != -1]
                #     noise_removed_clusters.append(non_anomalies)

                # self.clusters_agg = noise_removed_clusters


                # adv_list = [i for i in range(len(grads)) if names[i] in self.adversarial_namelist]
                # min_samples = self.params['no_models']//2 + 1
                # _, _, _, top_cluster = modHDBSCAN(np.array(grads), min_samples=min_samples, adv_list=adv_list)

                # candidate_clusters = []
                # cur_cluster = top_cluster
                # candidate_clusters.append(cur_cluster)
                # while True:
                #     if cur_cluster.child is None:
                #         break

                #     small_branch_size = min(len(cur_cluster.node.left.points), len(cur_cluster.node.right.points))

                #     if small_branch_size >= 5:
                #         candidate_clusters.append(cur_cluster.child)

                #     cur_cluster = cur_cluster.child
                #     continue

                # self.clusters_agg = [cluster.node.points for cluster in candidate_clusters]

                # _, self.clusters_agg = self.cluster_grads(grads, clustering_method='KMeans', clustering_params='grads', k=10)
                # min_cluster_size = 10
                # while True:
                #     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
                #     clusterer.fit(np.array(grads))
                #     cluster_labels = clusterer.labels_

                #     if np.max(cluster_labels)== -1:
                #         min_cluster_size -= 1
                #         continue
                    
                #     break

                # # logger.info(f'Cluster labels: {cluster_labels}')
                # self.clusters_agg = [[] for _ in range(max(cluster_labels)+1)]
                # for i, label in enumerate(cluster_labels):
                #     if label == -1:
                #         continue
                #     else:
                #         self.clusters_agg[label].append(i)
                # logger.info(f'clusters_agg: {self.clusters_agg}')
        
        logger.info(f'Agglomerative Clustering: Time: {time.time() - t}')
        t = time.time()

        clusters_agg = []
        mal_pcnt_by_cluster = []
        logger.info('Clustering by model updates')
        max_mal_cluster_index = -1
        max_mal_cluster_size = 0
        cluster_adversarialness = np.ones(len(self.clusters_agg))
        for idx, cluster in enumerate(self.clusters_agg):
            clstr = [names[c] for c in cluster]
            clusters_agg.append(clstr)
            mal_pcnt_by_cluster.append(self.mal_pcnt(clstr, names))
            mal_pcnt = len([c for c in clstr if c in self.adversarial_namelist])/len(clstr)
            if mal_pcnt > 0.5 and len(clstr) > max_mal_cluster_size:
                max_mal_cluster_size = len(clstr)
                max_mal_cluster_index = idx
            elif mal_pcnt == 0:
                cluster_adversarialness[idx] = 0

        pure_benign_clusters_indices = np.argwhere(cluster_adversarialness == 0).squeeze()
        promote_one_mal_model = True

        if max_mal_cluster_index == -1:
            try:
                max_mal_cluster_index = np.argwhere(cluster_adversarialness == 1)[0][0]
            except:
                pass

        logger.info(f'max_mal_cluster_index: {max_mal_cluster_index}')

        nets = self.local_models
        all_val_acc_list_dict = {}
        print(f'Validating all clients at epoch {epoch}')

        all_validator_evaluations = {}
        evaluations_of_clusters = {}
        count_of_class_for_validator = {}

        for name in names:
            all_validator_evaluations[name] = []

        evaluations_of_clusters[-1] = {}
        for iidx, val_idx in enumerate(names):

            if self.params['type'] == config.TYPE_LOAN:
                val_test_loader = self.allStateHelperList[val_idx].get_testloader()
            else:
                _, val_test_loader = self.train_data[val_idx]
            if val_idx in self.adversarial_namelist and not promote_one_mal_model:
                is_poisonous_validator = True
            else:
                is_poisonous_validator = False
            if self.params['type'] == config.TYPE_LOAN:
                val_acc, val_acc_by_class = self.validation_test_for_loan(target_model, val_test_loader, is_poisonous_validator, adv_index=0)
            else:
                if 'ablation_study' in self.params.keys() and 'no_wrong_validation' in self.params['ablation_study']:
                    is_poisonous_validator = False
                val_acc, val_loss, val_acc_by_class, count_of_class = self.validation_test_v2(target_model, val_test_loader, is_poisonous=is_poisonous_validator, adv_index=0)

            val_acc_by_class = [val_acc_by_class[i] for i in range(10)]
            all_validator_evaluations[val_idx] += val_acc_by_class
            evaluations_of_clusters[-1][val_idx] = [val_loss[i] for i in range(10)]
            if val_idx not in count_of_class_for_validator.keys():
                count_of_class_for_validator[val_idx] = count_of_class

        num_of_clusters = len(clusters_agg)

        adj_delta_models = []

        for idx, cluster in enumerate(tqdm(clusters_agg, disable=False)):
            evaluations_of_clusters[idx] = {}

            if len(cluster) == 0:
                continue
            if len(cluster) != 1:
                agg_model = self.new_model()
                agg_model.copy_params(self.target_model.state_dict())

                not_rfa = True
                if len(cluster) == 1:
                    not_rfa = True

                if not_rfa:
                    cluster_grads = []
                    for iidx, name in enumerate(names):
                        if name in cluster:
                            cluster_grads.append(client_grads[iidx])
                    
                    wv_frac = 1/len(cluster)

                    agg_grads = []
                    # Iterate through each layer
                    for i in range(len(cluster_grads[0])):
                        # assert len(wv) == len(cluster_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
                        temp = wv_frac * cluster_grads[0][i].cpu().clone()
                        # Aggregate gradients for a layer
                        for c, client_grad in enumerate(cluster_grads):
                            if c == 0:
                                continue
                            temp += wv_frac * client_grad[i].cpu()
                        # temp = temp / len(cluster_grads)
                        agg_grads.append(temp)

                    agg_model.train()
                    # train and update
                    optimizer = torch.optim.SGD(agg_model.parameters(), lr=1,
                                                momentum=self.params['momentum'],
                                                weight_decay=self.params['decay'])

                    optimizer.zero_grad()
                    for i, (name, params) in enumerate(agg_model.named_parameters()):
                        agg_grads[i]=agg_grads[i] * self.params["eta"]
                        if params.requires_grad:
                            params.grad = agg_grads[i].to(config.device)
                    optimizer.step()
                    for cl_id, cl_member in enumerate(cluster):
                        wv[self.clusters_agg[idx][cl_id]] = wv_frac
                    mal_pcnt_by_cluster[idx] = self.mal_pcnt(cluster, names, wv=[wv_frac for cl_member in cluster])
                else:
                    selected_updates = dict()
                    for iidx, name in enumerate(names):
                        if name in cluster:
                            selected_updates[name] = (alphas[iidx], delta_models[iidx])
                    _, _, _, wv_for_cluster_members, _ = self.geometric_median_update(target_model=agg_model, updates=selected_updates)
                    for cl_id, cl_member in enumerate(cluster):
                        wv[self.clusters_agg[idx][cl_id]] = wv_for_cluster_members[cl_id]
                    logger.info(f'wv for cluster {cluster}: {wv_for_cluster_members}')
                    mal_pcnt_by_cluster[idx] = self.mal_pcnt(cluster, names, wv=[wv[cl_member] for cl_member in cluster])
            else:
                fltrust_mode = True

                if not fltrust_mode:
                    agg_model = self.new_model()
                    agg_model.copy_params(self.local_models[cluster[0]].state_dict())
                else:
                    agg_model = self.new_model()
                    agg_model.copy_params(self.target_model.state_dict())
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

                    aggregate_weights = self.weighted_average_oracle(delta_models, torch.tensor(trust_scores))
                    adj_delta_models.append(aggregate_weights)

                    for name, data in agg_model.state_dict().items():
                        update_per_layer = aggregate_weights[name] 
                        try:
                            data.add_(update_per_layer)
                        except:
                            data.add_(update_per_layer.to(data.dtype))
                    


            for iidx, val_idx in enumerate(tqdm(names, disable=True)):
                if self.params['type'] == config.TYPE_LOAN:
                    val_test_loader = self.allStateHelperList[val_idx].get_testloader()
                else:
                    _, val_test_loader = self.train_data[val_idx]
                if val_idx in self.adversarial_namelist and not promote_one_mal_model:
                    is_poisonous_validator = True
                else:
                    is_poisonous_validator = False
                if self.params['type'] == config.TYPE_LOAN:
                    val_acc, val_acc_by_class = self.validation_test_for_loan(agg_model, val_test_loader, is_poisonous_validator, adv_index=0)
                else:
                    if 'ablation_study' in self.params.keys() and 'no_wrong_validation' in self.params['ablation_study']:
                        is_poisonous_validator = False
                    val_acc, val_loss, val_acc_by_class, _ = self.validation_test_v2(agg_model, val_test_loader, is_poisonous=is_poisonous_validator, adv_index=0)

                val_acc_by_class = [-val_acc_by_class[i]+all_validator_evaluations[val_idx][i] for i in range(10)]
                    
                all_validator_evaluations[val_idx]+= val_acc_by_class
                evaluations_of_clusters[idx][val_idx] = [-val_loss[i]+evaluations_of_clusters[-1][val_idx][i] for i in range(10)]
            
            # for client in cluster:
            #     all_val_acc_list_dict[client] = val_acc_list
        try:
            if fltrust_mode:
                delta_models = adj_delta_models
        except:
            logger.info('fltrust mode turned off')
            pass
        save_grads = True
        if save_grads:
            self.snapshot(full_grads, names, alphas, epoch, self.clusters_agg)

        # imputing missing validation values
        # convert from dict to list
        # all_validator_evaluations = [all_validator_evaluations[val_idx] for val_idx in range(len(names))]
        all_validator_evaluations = [all_validator_evaluations[names[val_idx]] for val_idx in range(len(names))]
        imputer = IterativeImputer(n_nearest_features = 5, initial_strategy = 'median', random_state = 42)
        all_validator_evaluations = imputer.fit_transform(all_validator_evaluations)

        all_validator_evaluations_dict = dict()
        for val_idx in range(len(names)):
            all_validator_evaluations_dict[names[val_idx]] = all_validator_evaluations[val_idx]

        all_validator_evaluations = all_validator_evaluations_dict

        # promoting one malicious model
        if promote_one_mal_model and max_mal_cluster_index != -1:
            malicious_validators = [val_idx for val_idx in names if val_idx in self.adversarial_namelist]
            benign_validators = [val_idx for val_idx in names if val_idx not in self.adversarial_namelist]
            benign_losses = [evaluations_of_clusters[max_mal_cluster_index][val_idx][self.source_class] for val_idx in benign_validators]

            malicious_validators = [val_idx for val_idx in malicious_validators if count_of_class_for_validator[val_idx][self.source_class] > 0]

            mal_losses = [count_of_class_for_validator[val_idx][self.source_class] for val_idx in malicious_validators]
            mal_losses = mal_losses/np.sum(mal_losses)
            mal_losses = [- np.sum(benign_losses)* mal_losses[i] for i in range(len(mal_losses))]
            mal_losses = [-np.sum(benign_losses)/len(malicious_validators) for _ in range(len(mal_losses))]
            for validator in malicious_validators:
                evaluations_of_clusters[max_mal_cluster_index][validator][self.source_class] = mal_losses[malicious_validators.index(validator)]

                old_value = all_validator_evaluations[validator][(max_mal_cluster_index+1)*10 + self.source_class]

                all_validator_evaluations[validator][(max_mal_cluster_index+1)*10 + self.source_class] = mal_losses[malicious_validators.index(validator)]/count_of_class_for_validator[validator][self.source_class]

                logger.info(f'{validator} old value: {old_value} new value: {all_validator_evaluations[validator][(max_mal_cluster_index+1)*10 + self.source_class]}')

            
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
        #             # if scores_idx == 10 * max_mal_cluster_index + self.source_class:
        #             cluster_idx = scores_idx // 10 -1
        #             if (scores_idx - self.source_class)%10 == 0 and cluster_idx >= 0:
        #                 if cluster_adversarialness[cluster_idx] > 0:
        #                     highlight_color = "on_white"
        #                 else:
        #                     highlight_color = "on_yellow"
        #                 scores[scores_idx][idx] = colored("{:.2f}".format(scores[scores_idx][idx]), color, highlight_color)
        #             else:
        #                 scores[scores_idx][idx] = colored("{:.2f}".format(scores[scores_idx][idx]), color=color)

        #     table_header = [colored(val_idx, color=f'{"red" if val_idx in self.adversarial_namelist else "blue"}') for val_idx in names[10*itr_idx:10*(itr_idx+1)]]
        #     print(tabulate(scores, headers=table_header))
        


        logger.info(f'Validation Done: Time: {time.time() - t}')
        t = time.time()

        # checking integrity of validation results
        validator_xs = [all_validator_evaluations[name] for name in names]

        anomaly_detection = False
        if anomaly_detection:
            validator_xs = np.array(validator_xs)
            # anomaly_arr_for_val = EllipticEnvelope(contamination=0.4).fit_predict(validator_xs)
            anomaly_arr_for_val = LocalOutlierFactor(contamination=0.4).fit_predict(validator_xs)
            logger.info(f'anomaly_arr_for_val: {anomaly_arr_for_val}')
            anomalies = [idx for idx in range(len(names)) if anomaly_arr_for_val[idx] == -1]
            non_anomalies = [idx for idx in range(len(names)) if anomaly_arr_for_val[idx] != -1]
            val_clustering = [non_anomalies, anomalies]
            logger.info(f'val_clustering: {val_clustering}')
        else:
            _, val_clustering = self.cluster_grads(validator_xs, clustering_params='grads', clustering_method='KMeans')
        good_cluster = np.argmax([len(cluster) for cluster in val_clustering])
        logger.info([self.mal_pcnt(cluster, names) for cluster in val_clustering])
        logger.info(f'good cluster: {val_clustering[good_cluster]}')

        remaining_validators = []

        for idx, name in enumerate(names):
            if idx in val_clustering[good_cluster]:
                remaining_validators.append(name)
            else:
                for cluster_idx in range(num_of_clusters):
                    try:
                        del evaluations_of_clusters[cluster_idx][name]
                    except:
                        pass

        wv_by_cluster = []

        total_count_of_class = [0 for _ in range(10)]
        for val_idx in remaining_validators:
            total_count_of_class = [total_count_of_class[i]+ count_of_class_for_validator[val_idx][i] for i in range(10)]

        logger.info(f'total_count_of_class: {total_count_of_class}')

        for cluster_idx in range(num_of_clusters):
            all_vals = [evaluations_of_clusters[cluster_idx][name] for name in remaining_validators]
            all_vals = np.array(all_vals)
            all_vals = np.transpose(all_vals)
            all_vals = all_vals.tolist()


            eval_sum_of_cluster = [np.sum(all_vals[i]) for i in range(len(all_vals))]

            class_by_class_evaluation = True
            eval_mean_of_cluster_by_class = [eval_sum_of_cluster[i]/total_count_of_class[i] for i in range(len(eval_sum_of_cluster))]
            if cluster_idx == max_mal_cluster_index:
                logger.info(f'loss of promoted model: {eval_mean_of_cluster_by_class}')
            if class_by_class_evaluation:
                eval_mean_of_cluster = eval_mean_of_cluster_by_class
                wv_by_cluster.append(np.min(eval_mean_of_cluster))
            else:
                eval_mean_of_cluster = np.sum(eval_sum_of_cluster)/np.sum(total_count_of_class)
                wv_by_cluster.append(eval_mean_of_cluster)
            # eval_mean_of_cluster = eval_mean_of_cluster[10:]
            # eval_mean_of_cluster = eval_mean_of_cluster.reshape(len(eval_mean_of_cluster)//10, 10)
            # eval_mean_of_cluster = np.mean(eval_mean_of_cluster, axis=0)
            evaluations_of_clusters[cluster_idx] = eval_mean_of_cluster
            # logger.info(f'cluster {cluster_idx} with mal_pcnt {self.mal_pcnt(clusters_agg[cluster_idx], names)} performance: {eval_mean_of_cluster}')

        logger.info(f'wv_by_cluster: {rankdata(wv_by_cluster)}')
        med_wv = np.median(wv_by_cluster)
        old_wv_by_cluster = copy.deepcopy(wv_by_cluster)
        pick_the_best_cluster = False
        if pick_the_best_cluster:
            wv_by_cluster = [1 if z==np.max(wv_by_cluster) else 0 for z in wv_by_cluster]
        else:
            wv_by_cluster = [1 if z>=med_wv else 0 for z in wv_by_cluster]
        # for cluster_idx in range(num_of_clusters):
        #     logger.info(f'cluster {cluster_idx} with mal_pcnt {mal_pcnt_by_cluster[cluster_idx]} performance: {wv_by_cluster[cluster_idx]}')
        # logger.info(f'wv_by_cluster updated: {wv_by_cluster}')
        # max_wv = max(wv_by_cluster)
        # good_clusters = np.where(wv_by_cluster == max_wv)[0]
        # logger.info(f'good clusters {good_clusters}: {[self.clusters_agg[gc] for gc in good_clusters]}')
        # wv_by_cluster = [1 if wv == max_wv else 0 for wv in wv_by_cluster]

        norm_median = np.median(norms)
        clipping_weights = [min(norm_median/norm, 1) for norm in norms]
        if 'ablation_study' in self.params.keys() and 'missing_clipping' in self.params['ablation_study']:
            clipping_weights = [1 for norm in norms]
        # wv = np.zeros(len(names), dtype=np.float32)
        green_clusters = []
        mal_pcnts = []
        for idx, cluster in enumerate(self.clusters_agg):
            mal_pcnts.append(sum([wv[cl_id] for cl_id in cluster if names[cl_id] in self.adversarial_namelist]))
            if pick_the_best_cluster and idx != np.argmax(wv_by_cluster):
                continue
            for cl_id in cluster:
                # wv[cl_id] = wv_by_cluster[idx]
                if no_clustering:
                    wv[cl_id] = wv_by_cluster[cl_id]
                else:
                    wv[cl_id] = wv_by_cluster[idx] * len(cluster) * wv[cl_id]

            # logger.info(f'Cluster {idx}')
            # logger.info(f'Members: {[names[i] for i in cluster]}')
            # logger.info(f'wv: {wv[cluster]}')
            # logger.info(f'benign members: {[names[i] for i in cluster if names[i] in self.benign_namelist]}')
            # logger.info(f'benign wv: {[wv[cl_id] for cl_id in cluster if names[cl_id] in self.benign_namelist]}')
            # logger.info(f'malicious members: {[names[i] for i in cluster if names[i] in self.adversarial_namelist]}')
            # logger.info(f'malicious wv: {[wv[cl_id] for cl_id in cluster if names[cl_id] in self.adversarial_namelist]}')
            if wv_by_cluster[idx] > 0:
                green_clusters.append(idx)

        for idx, cluster in enumerate(self.clusters_agg):
            mal_client_lsrs = [f'{names[cl_id]}: {self.lsrs[names[cl_id]][self.source_class]}' for cl_id in cluster if names[cl_id] in self.adversarial_namelist]
            benign_client_lsrs = [f'{names[cl_id]}: {self.lsrs[names[cl_id]][self.source_class]}' for cl_id in cluster if names[cl_id] in self.benign_namelist]
            # logger.info(f'mal clients: {mal_client_lsrs}')
            # logger.info(f'benign clients: {benign_client_lsrs}')
            print_str = f'{"Green" if idx in green_clusters else "Filtered"} cluster {idx} of size {len(cluster)} with mal_pcnt {len(mal_client_lsrs)/(len(mal_client_lsrs)+len(benign_client_lsrs))} and wv {old_wv_by_cluster[idx]}'
            print(colored(print_str, 'green' if idx in green_clusters else 'red'))
        wv = [w*c for w,c in zip(wv, clipping_weights)]

        # cluster_avg_wvs = []
        # for cluster_id, cluster in enumerate(self.clusters_agg):
        #     cluster_avg_wvs.append(np.median([wv[client_id] for client_id in cluster]))
        # # min_cluster_avg_wvs_index = np.argmin(cluster_avg_wvs)
        # zscore_by_clusters_v1 = stats.zscore(cluster_avg_wvs)

        # zscore_by_clients = stats.zscore(wv)
        # zscore_by_clusters_v2 = [np.median([zscore_by_clients[client_id] for client_id in cluster]) for cluster in self.clusters_agg]

        # zscore_by_clusters = [min(zscore_by_clusters_v1[cluster_id], zscore_by_clusters_v2[cluster_id]) for cluster_id, cluster in enumerate(self.clusters_agg)]
        # # for client_id in self.clusters_agg[min_cluster_avg_wvs_index]:
        # #     wv[client_id] = 0

        # for cl_num, cluster in enumerate(self.clusters_agg):
        #     mal_count = 0
        #     for client_id in cluster:
        #         if names[client_id] in self.adversarial_namelist:
        #             mal_count += 1
        #     mal_count = mal_count/len(cluster)

        #     logger.info(f'{clusters_agg[cl_num]}, {mal_count}, {zscore_by_clusters[cl_num]}, {zscore_by_clusters_v1[cl_num]}, {zscore_by_clusters_v2[cl_num]}')
        #     if 'ablation_study' in self.params.keys() and 'missing_filtering' in self.params['ablation_study']:
        #         break
            
        #     if zscore_by_clusters[cl_num] <= -1:
        #         for client_id in cluster:
        #             wv[client_id] = 0

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

        noise_level = self.params['sigma'] * norm_median
        self.add_noise(target_model, noise_level)
        logger.info(f'Aggregation Done: Time {time.time() - t}')
        t = time.time()
        logger.info(f'adversarial wv: {[self.print_util(names[iidx], wv[iidx]) for iidx in range(len(wv)) if names[iidx] in self.adversarial_namelist]}')
        logger.info(f'benign wv: {[self.print_util(names[iidx], wv[iidx]) for iidx in range(len(wv)) if names[iidx] in self.benign_namelist]}')
        return

    
    # def combined_clustering_guided_aggregation(self, target_model, updates, epoch):
    #     start_epoch = self.start_epoch
    #     if epoch < start_epoch:
    #         self.fedavg(target_model, updates)
    #         return
    #     start_time = time.time()
    #     t = time.time()
    #     logger.info(f'Started clustering guided aggregation')
    #     client_grads = []
    #     alphas = []
    #     names = []
    #     delta_models = []
    #     for name, data in updates.items():
    #         client_grads.append(data[1])  # gradient
    #         alphas.append(data[0])  # num_samples
    #         delta_models.append(data[2])
    #         names.append(name)

    #     # grads = [self.convert_model_to_param_list(client_grad) for client_grad in client_grads]
    #     grads = [self.flatten_gradient(client_grad) for client_grad in client_grads]
    #     logger.info(f'Converted gradients to param list: Time: {time.time() - t}')
    #     t = time.time()
    #     # grads = client_grads
    #     print(names)
    #     if epoch==start_epoch:
    #         self.validator_trust_scores = [1. for _ in range(self.params['number_of_total_participants'])]
    #         _, clusters = self.cluster_grads(grads, clustering_params='lsrs')
    #         self.clusters = clusters
    #         all_group_nos = []
    #         for i, cluster in enumerate(self.clusters):
    #             if len(clusters) > 2:
    #                 all_group_nos.append(i)
    #         self.all_group_nos = all_group_nos

    #         print('Spectral clustering output')
    #         print(clusters)
            
    #         logger.info(f'Spectral Clustering: Time: {time.time() - t}')
    #         t = time.time()

    #     if epoch <0:
    #         assert epoch == 0, 'fix epoch {}'.format(len(epoch))

              
    #     else:
    #         # agglomerative clustering based validation

    #         #get agglomerative clusters
    #         # if epoch<2 or np.random.random_sample() < np.min([0.1, np.exp(-epoch*0.1)/(1. + np.exp(-epoch*0.1))]):
    #         # if epoch<5:
    #         #     k = 5
    #         # else:
    #         #     k = 2
    #         # _, self.clusters_agg = self.cluster_grads(grads, clustering_method='Agglomerative', clustering_params='grads', k=10)
    #         if self.params['no_models'] < 10:
    #             self.clusters_agg = [[i] for i in range(self.params['no_models'])]
    #         else:
    #             if 'ablation_study' in self.params.keys() and 'clustering_kmeans' in self.params['ablation_study']:
    #                 _, self.clusters_agg = self.cluster_grads(grads, clustering_method='KMeans', clustering_params='grads', k=10)
    #             elif 'ablation_study' in self.params.keys() and 'clustering_spectral' in self.params['ablation_study']:
    #                 _, self.clusters_agg = self.cluster_grads(grads, clustering_method='Spectral', clustering_params='grads', k=10)
    #             else:
    #                 _, self.clusters_agg = self.cluster_grads(grads, clustering_method='Agglomerative', clustering_params='grads', k=10)
            
    #         logger.info(f'Agglomerative Clustering: Time: {time.time() - t}')
    #         t = time.time()

    #         clusters_agg = []
    #         for clstr in self.clusters_agg:
    #             clstr = [names[c] for c in clstr]
    #             clusters_agg.append(clstr)

    #         nets = self.local_models
    #         all_val_acc_list_dict = {}
    #         print(f'Validating all clients at epoch {epoch}')
    #         print(f'{self.clusters_agg}_{clusters_agg}')
    #         val_client_indice_tuples=[]
    #         # self.recalculate_val_trust_scores(grads)
    #         all_validators = []
    #         for i, val_cluster in enumerate(self.clusters):
    #             val_trust_scores = [self.validator_trust_scores[vid] for vid in val_cluster]
    #             # if np.max(val_trust_scores) < 0.01:
    #             #     for vid in val_cluster:
    #             #         self.validator_trust_scores[vid] = 1.
    #             if len(val_cluster) > 2 and np.max(val_trust_scores) > 0.0005:
    #                 # v1, v2 = random.sample(val_cluster, 2)
    #                 val_trust_scores = np.array(val_trust_scores)/sum(val_trust_scores)
    #                 v1, v2 = np.random.choice(val_cluster, 2, replace=False, p=val_trust_scores)
    #                 all_validators += [v1, v2]
    #                 val_client_indice_tuples.append((i, v1))
    #                 val_client_indice_tuples.append((i, v2))

    #         if 'adaptive_grad_attack' in self.params.keys() and self.params['adaptive_grad_attack']:
    #             self.prev_epoch_val_model_params = []

    #         for idx, cluster in enumerate(clusters_agg):
    #             if len(cluster) == 0:
    #                 continue
    #             if len(cluster) != 1:
    #                 agg_model = self.new_model()
    #                 agg_model.copy_params(self.target_model.state_dict())

    #                 cluster_grads = []
    #                 for iidx, name in enumerate(names):
    #                     if name in cluster:
    #                         print(name)
    #                         cluster_grads.append(client_grads[iidx])
                    
    #                 wv = 1/len(cluster)
    #                 # wv = np.ones(self.params['no_models'])
    #                 # wv = wv/len(wv)
    #                 logger.info(f'wv: {wv}')
    #                 # agg_grads = {}
    #                 # # Iterate through each layer
    #                 # for name in cluster_grads[0].keys():
    #                 #     temp = wv * cluster_grads[0][name].cpu().clone()
    #                 #     # Aggregate gradients for a layer
    #                 #     for c, cluster_grad in enumerate(cluster_grads):
    #                 #         if c == 0:
    #                 #             continue
    #                 #         temp += wv * cluster_grad[name].cpu()
    #                 #     agg_grads[name] = temp

    #                 agg_grads = []
    #                 # Iterate through each layer
    #                 for i in range(len(cluster_grads[0])):
    #                     # assert len(wv) == len(cluster_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
    #                     temp = wv * cluster_grads[0][i].cpu().clone()
    #                     # Aggregate gradients for a layer
    #                     for c, client_grad in enumerate(cluster_grads):
    #                         if c == 0:
    #                             continue
    #                         temp += wv * client_grad[i].cpu()
    #                     # temp = temp / len(cluster_grads)
    #                     agg_grads.append(temp)

    #                 agg_model.train()
    #                 # train and update
    #                 optimizer = torch.optim.SGD(agg_model.parameters(), lr=1,
    #                                             momentum=self.params['momentum'],
    #                                             weight_decay=self.params['decay'])

    #                 optimizer.zero_grad()
    #                 for i, (name, params) in enumerate(agg_model.named_parameters()):
    #                     agg_grads[i]=agg_grads[i] * self.params["eta"]
    #                     if params.requires_grad:
    #                         params.grad = agg_grads[i].to(config.device)
    #                 optimizer.step()
                    
    #                 if 'adaptive_grad_attack' in self.params.keys() and self.params['adaptive_grad_attack']:
    #                     # agg_model_params = dict()

    #                     # for name, param in agg_model.named_parameters():
    #                     #     agg_model_params[name] = param.data.clone().detach().requires_grad_(False)

    #                     # self.prev_epoch_val_model_params.append(agg_model_params)
    #                     self.prev_epoch_val_model_params.append(self.flatten_gradient(agg_grads))           
    #             else:
    #                 agg_model = self.local_models[cluster[0]]


    #             val_acc_list=[]
    #             for iidx, (group_no, val_idx) in enumerate(val_client_indice_tuples):
    #                 # no validation data exchange between malicious clients
    #                 # _, _, val_test_loader = train_loaders[epoch][val_idx]
    #                 # targeted label flip attack where malicious clients coordinate and test against data from the target group's malicious client
    #                 # if self.params['attack_methods'] in [config.ATTACK_TLF, config.ATTACK_SIA]:
    #                 if self.params['attack_methods'] == config.ATTACK_SIA:
    #                     if val_idx in self.adversarial_namelist:
    #                         adv_list = np.array(self.adversarial_namelist)
    #                         if self.params['type'] == config.TYPE_LOAN:
    #                             # val_test_loader = self.allStateHelperList[val_idx].get_testloader()
    #                             val_test_loader = self.allStateHelperList[val_idx].get_trainloader()
    #                         else:
    #                             _, val_test_loader = self.train_data[np.min(adv_list[adv_list>self.source_class*10])]
    #                     else:
    #                         if self.params['type'] == config.TYPE_LOAN:
    #                             # val_test_loader = self.allStateHelperList[val_idx].get_testloader()
    #                             val_test_loader = self.allStateHelperList[val_idx].get_trainloader()
    #                         else:
    #                             _, val_test_loader = self.train_data[val_idx]
    #                 else:
    #                     if self.params['type'] == config.TYPE_LOAN:
    #                         val_test_loader = self.allStateHelperList[val_idx].get_testloader()
    #                     else:
    #                         _, val_test_loader = self.train_data[val_idx]
    #                 if val_idx in self.adversarial_namelist:
    #                     is_poisonous_validator = True
    #                 else:
    #                     is_poisonous_validator = False
    #                 if self.params['type'] == config.TYPE_LOAN:
    #                     val_acc, val_acc_by_class = self.validation_test_for_loan(agg_model, val_test_loader, is_poisonous_validator, adv_index=0)
    #                 else:
    #                     if 'ablation_study' in self.params.keys() and 'no_wrong_validation' in self.params['ablation_study']:
    #                         is_poisonous_validator = False
    #                     val_acc, val_acc_by_class = self.validation_test(agg_model, val_test_loader, is_poisonous=is_poisonous_validator, adv_index=0)
    #                 # logger.info(f'cluster: {cluster}, val_idx: {val_idx}, is_mal_validator: {val_idx in self.adversarial_namelist}, val_acc: {val_acc}')
    #                 # logger.info(f'cluster: {cluster}, val_idx: {val_idx}, is_mal_validator: {val_idx in self.adversarial_namelist}, val_acc: {val_acc}, val_acc_by_class: {val_acc_by_class}')
    #                 if self.params['type'] == config.TYPE_LOAN:
    #                     val_acc_list.append((val_idx, -1, val_acc, val_acc_by_class))
    #                 else:
    #                     val_acc_list.append((val_idx, -1, val_acc.item(), val_acc_by_class))
                
    #             for client in cluster:
    #                 all_val_acc_list_dict[client] = val_acc_list

    #         logger.info(f'Validation Done: Time: {time.time() - t}')
    #         t = time.time()

    #     def get_group_no(validator_id, clustr):
    #         for grp_no in range(len(clustr)):
    #             if validator_id in clustr[grp_no]:
    #                 return grp_no
    #         return -1

    #     def get_min_group_and_score(val_score_by_grp_dict):
    #         min_val = 100
    #         min_grp_no = -1
    #         for grp_no in val_score_by_grp_dict.keys():
    #             if val_score_by_group_dict[grp_no] < min_val:
    #                 min_val = val_score_by_group_dict[grp_no]
    #                 min_grp_no = grp_no
    #         return min_grp_no, min_val

    #     all_val_score_by_group_dict={}

    #     all_val_score = {}
    #     all_val_score_min_grp={}
    #     validator_flags = [0. for _ in all_validators]
    #     for client_id in names:
    #         val_score_by_group_dict={}
    #         val_acc_list = all_val_acc_list_dict[client_id]

            
    #         # take the one closer to the others
    #         validators = {}
    #         for iidx, (val_idx, _, val_acc, val_acc_report) in enumerate(val_acc_list):
    #             grp_no = get_group_no(val_idx, self.clusters)
    #             if grp_no in val_score_by_group_dict.keys():
    #                 # if average
    #                 # val_score_by_group_dict[grp_no] += val_acc
    #                 # val_score_by_group_dict[grp_no] /= 2
    #                 # if minimum
    #                 val_score_by_group_dict[grp_no].append((val_acc, val_acc_report))
    #                 validators[grp_no].append(val_idx)
    #             else:
    #                 val_score_by_group_dict[grp_no] = [(val_acc, val_acc_report)]
    #                 validators[grp_no]= [val_idx]

    #         all_grp_nos = list(val_score_by_group_dict.keys())

    #         new_val_score_by_group_dict = {}
    #         for grp_no in all_grp_nos:
    #             if len(all_grp_nos) == 1:
    #                 new_val_score_by_group_dict[grp_no] = (val_score_by_group_dict[grp_no][0][0] + val_score_by_group_dict[grp_no][1][0])/2
    #                 break
    #             mean_by_class = {}
    #             if self.params['type'] == config.TYPE_LOAN:
    #                 cl_count = 9
    #             else:
    #                 cl_count = 10
    #             for c in range(cl_count):
    #                 mean_by_class[c] = 0.
    #                 for group_no in val_score_by_group_dict.keys():
    #                     if group_no != grp_no:
    #                         for grp_mm in range(2):
    #                             mean_by_class[c] += val_score_by_group_dict[group_no][grp_mm][1][c]
    #                 mean_by_class[c] /= ((len(val_score_by_group_dict.keys())-1) * 2)

    #             min_diff_by_class = [0. for c in range(cl_count)]
    #             for c in range(cl_count):
    #                 min_diff_by_class[c] = np.min([np.abs(mean_by_class[c] - val_score_by_group_dict[grp_no][0][1][c]), np.abs(mean_by_class[c] - val_score_by_group_dict[grp_no][1][1][c])])
                
    #             target_class = np.argmax(min_diff_by_class)
                
    #             val_acc_0 = val_score_by_group_dict[grp_no][0][1][target_class]
    #             val_acc_1 = val_score_by_group_dict[grp_no][1][1][target_class]
    #             # total_acc_excluding = total_acc - val_acc_0 - val_acc_1
    #             mean_acc_excluding = mean_by_class[target_class]

    #             if 'auth_threshold' in self.params.keys():
    #                 auth_threshold = self.params['auth_threshold']
    #             elif 'ablation_study' in self.params.keys() and 'missing_authentication' in self.params['ablation_study']:
    #                 auth_threshold = 100.
    #             else:
    #                 auth_threshold = 40.

    #             if min(abs(mean_acc_excluding-val_acc_0),abs(mean_acc_excluding-val_acc_1))>auth_threshold:
    #                 repl_acc = 0.
    #                 for grp_idx in all_grp_nos:
    #                     if grp_idx != grp_no:
    #                         for (val_acc, val_acc_report) in val_score_by_group_dict[grp_idx]:
    #                             repl_acc += val_acc
    #                 repl_acc = repl_acc/(2*(len(all_grp_nos)-1))
    #                 new_val_score_by_group_dict[grp_no] = repl_acc
    #                 for validator in validators[grp_no]:
    #                     self.validator_trust_scores[validator] = self.validator_trust_scores[validator]/2
    #                     validator_flags[all_validators.index(validator)] = 1.
    #             elif abs(mean_acc_excluding-val_acc_0)<abs(mean_acc_excluding-val_acc_1):
    #                 if abs(mean_acc_excluding-val_acc_1)>auth_threshold:
    #                     validator = validators[grp_no][1]
    #                     self.validator_trust_scores[validator] = self.validator_trust_scores[validator]/2
    #                     validator_flags[all_validators.index(validator)] = 1.
    #                 new_val_score_by_group_dict[grp_no] = val_score_by_group_dict[grp_no][0][0]
    #             else:
    #                 if abs(mean_acc_excluding-val_acc_0)>auth_threshold:
    #                     validator = validators[grp_no][0]
    #                     self.validator_trust_scores[validator] = self.validator_trust_scores[validator]/2
    #                     validator_flags[all_validators.index(validator)] = 1.
    #                 new_val_score_by_group_dict[grp_no] = val_score_by_group_dict[grp_no][1][0]
    #         # for grp_no in self.all_group_nos:
    #         #     if grp_no not in new_val_score_by_group_dict.keys():
    #         #         new_val_score_by_group_dict[grp_no] = -1
    #         val_score_by_group_dict = new_val_score_by_group_dict
            
    #         all_val_score_by_group_dict[client_id] = val_score_by_group_dict
    #         min_val_grp_no, min_val_score = get_min_group_and_score(val_score_by_group_dict)
    #         all_val_score[client_id] = min_val_score
    #         all_val_score_min_grp[client_id] = min_val_grp_no

    #     logger.info(f'Validation scoring by group done: {time.time()-t}')
    #     t = time.time()
              
    #     if epoch<1:
    #         self.global_net.set_param_to_zero()
    #         self.global_net.aggregate([network.state_dict() for network in self.benign_nets + self.mal_nets])
    #     elif epoch == start_epoch:

    #         self.all_val_score = all_val_score
    #         self.all_val_score_min_grp = all_val_score_min_grp

    #         aggr_weights = [self.all_val_score[client_id] for client_id in names]
    #         aggr_weights = np.array(aggr_weights)
    #         aggr_weights = aggr_weights/np.sum(aggr_weights)

    #         wv = aggr_weights
    #         logger.info(f'wv: {wv}')
    #         agg_grads = {}
    #         # # Iterate through each layer
    #         # for name in client_grads[0].keys():
    #         #     assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
    #         #     temp = wv[0] * client_grads[0][name].cpu().clone()
    #         #     # Aggregate gradients for a layer
    #         #     for c, client_grad in enumerate(client_grads):
    #         #         if c == 0:
    #         #             continue
    #         #         temp += wv[c] * client_grad[name].cpu()
    #         #     agg_grads[name] = temp

    #         # target_model.train()
    #         # # train and update
    #         # optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
    #         #                             momentum=self.params['momentum'],
    #         #                             weight_decay=self.params['decay'])

    #         # optimizer.zero_grad()
    #         # for i, (name, params) in enumerate(target_model.named_parameters()):
    #         #     agg_grads[name]=-agg_grads[name] * self.params["eta"]
    #         #     if params.requires_grad:
    #         #         params.grad = agg_grads[name].to(config.device)
    #         # optimizer.step()
        
    #     else:
    #         adjusted_clients = [0 for _ in range(len(names))]
    #         for client_id in names:
    #             if 'ablation_study' in self.params.keys() and 'missing_adjustment' in self.params['ablation_study']:
    #                 break
    #             if client_id in self.all_val_score.keys():
    #                 prev_val_score = self.all_val_score[client_id]
    #                 if prev_val_score < 50.:
    #                     prev_val_grp_no = self.all_val_score_min_grp[client_id]
    #                     if prev_val_grp_no in all_val_score_by_group_dict[client_id].keys():
    #                         current_val_score_on_that_group = all_val_score_by_group_dict[client_id][prev_val_grp_no]
    #                         if 0<= current_val_score_on_that_group and current_val_score_on_that_group < 50:
    #                             all_val_score[client_id] = prev_val_score/2
    #                             all_val_score_min_grp[client_id] = prev_val_grp_no

    #                             adjusted_clients[names.index(client_id)] = 1
            
    #         for name in all_val_score.keys():
    #             self.all_val_score[name] = all_val_score[name]
    #         for name in all_val_score_min_grp.keys():
    #             self.all_val_score_min_grp[name] = all_val_score_min_grp[name]

    #         aggr_weights = [self.all_val_score[client_id] for client_id in names]
    #         aggr_weights = np.array(aggr_weights)
    #         aggr_weights = aggr_weights/np.sum(aggr_weights)

    #         wv = aggr_weights
    #         logger.info(f'wv: {wv}')
    #         agg_grads = {}


    #     # norms = [torch.linalg.norm(grad).item() for grad in grads]
    #     norms = [np.linalg.norm(grad) for grad in grads]
    #     norm_median = np.median(norms)
    #     clipping_weights = [min(norm_median/norm, 1) for norm in norms]
    #     if 'ablation_study' in self.params.keys() and 'missing_clipping' in self.params['ablation_study']:
    #         clipping_weights = [1 for norm in norms]
    #     wv = [w*c for w,c in zip(wv, clipping_weights)]

    #     # min_aggr_weights_index = np.argmin(wv)
    #     # for cluster in self.clusters_agg:
    #     #     if min_aggr_weights_index in cluster:
    #     #         for client_id in cluster:
    #     #             wv[client_id] = 0

    #     # wv = wv/np.sum(wv)

    #     cluster_avg_wvs = []
    #     for cluster_id, cluster in enumerate(self.clusters_agg):
    #         cluster_avg_wvs.append(np.median([wv[client_id] for client_id in cluster]))
    #     # min_cluster_avg_wvs_index = np.argmin(cluster_avg_wvs)
    #     zscore_by_clusters_v1 = stats.zscore(cluster_avg_wvs)

    #     zscore_by_clients = stats.zscore(wv)
    #     zscore_by_clusters_v2 = [np.median([zscore_by_clients[client_id] for client_id in cluster]) for cluster in self.clusters_agg]

    #     zscore_by_clusters = [min(zscore_by_clusters_v1[cluster_id], zscore_by_clusters_v2[cluster_id]) for cluster_id, cluster in enumerate(self.clusters_agg)]
    #     # for client_id in self.clusters_agg[min_cluster_avg_wvs_index]:
    #     #     wv[client_id] = 0

    #     for cl_num, cluster in enumerate(self.clusters_agg):
    #         mal_count = 0
    #         for client_id in cluster:
    #             if names[client_id] in self.adversarial_namelist:
    #                 mal_count += 1
    #         mal_count = mal_count/len(cluster)

    #         logger.info(f'{clusters_agg[cl_num]}, {mal_count}, {self.all_val_score_min_grp[clusters_agg[cl_num][0]]}, {zscore_by_clusters[cl_num]}, {zscore_by_clusters_v1[cl_num]}, {zscore_by_clusters_v2[cl_num]}')
    #         if 'ablation_study' in self.params.keys() and 'missing_filtering' in self.params['ablation_study']:
    #             break
            
    #         if zscore_by_clusters[cl_num] <= -1:
    #             for client_id in cluster:
    #                 wv[client_id] = 0

        
    #     # norms = [torch.linalg.norm(grad).item() for grad in grads]
    #     # norms = [np.linalg.norm(grad) for grad in grads]
    #     # norm_median = np.median(norms)
    #     # clipping_weights = [min(norm_median/norm, 1) for norm in norms]
    #     # alphas = alphas/np.sum(alphas)
    #     # wv = [w*c*a for w,c,a in zip(wv, clipping_weights, alphas)]
    #     wv = wv/np.sum(wv)

    #     logger.info(f'clipping_weights: {clipping_weights}')
    #     logger.info(f'adversarial clipping weights: {[self.print_util(names[iidx], clipping_weights[iidx]) for iidx in range(len(clipping_weights)) if names[iidx] in self.adversarial_namelist]}')
    #     logger.info(f'benign clipping weights: {[self.print_util(names[iidx], clipping_weights[iidx]) for iidx in range(len(clipping_weights)) if names[iidx] in self.benign_namelist]}')
    #     wv_print_str= '['
    #     for idx, w in enumerate(wv):
    #         wv_print_str += ' '
    #         if names[idx] in self.adversarial_namelist:
    #             wv_print_str += colored(str(w), 'blue')
    #         else:
    #             wv_print_str += str(w)
    #     wv_print_str += ']'
    #     print(f'wv: {wv_print_str}')

    #     # agg_grads = []
    #     # # Iterate through each layer
    #     # for i in range(len(client_grads[0])):
    #     #     # assert len(wv) == len(cluster_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
    #     #     temp = wv[0] * client_grads[0][i].cpu().clone()
    #     #     # Aggregate gradients for a layer
    #     #     for c, client_grad in enumerate(client_grads):
    #     #         if c == 0:
    #     #             continue
    #     #         temp += wv[c] * client_grad[i].cpu()
    #     #     # temp = temp / len(client_grads)
    #     #     agg_grads.append(temp)

    #     # logger.info(f'agg_grads: {self.flatten_gradient(agg_grads)}')

    #     # target_model.train()
    #     # # train and update
    #     # optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
    #     #                             momentum=self.params['momentum'],
    #     #                             weight_decay=self.params['decay'])

    #     # optimizer.zero_grad()
    #     # for i, (name, params) in enumerate(target_model.named_parameters()):
    #     #     agg_grads[i]=agg_grads[i]
    #     #     if params.requires_grad:
    #     #         params.grad = agg_grads[i].to(config.device)
    #     # optimizer.step()
    #     aggregate_weights = self.weighted_average_oracle(delta_models, torch.tensor(wv))

    #     for name, data in target_model.state_dict().items():
    #         update_per_layer = aggregate_weights[name] * (self.params["eta"])
    #         try:
    #             data.add_(update_per_layer)
    #         except:
    #             # logger.info(f'layer name: {name}')
    #             # logger.info(f'data: {data}')
    #             # logger.info(f'update_per_layer: {update_per_layer}')
    #             data.add_(update_per_layer.to(data.dtype))
    #             # logger.info(f'after update: {update_per_layer.to(data.dtype)}')

    #     # if self.params['type'] != config.TYPE_LOAN:
    #     #     noise_level = self.params['sigma'] * norm_median
    #     #     self.add_noise(noise_level=noise_level)
    #     logger.info(f'Aggregation Done: Time {time.time() - t}')
    #     t = time.time()
    #     logger.info(f'wv: {wv}')
    #     try:
    #         adv_validators = [validator for validator in all_validators if validator in self.adversarial_namelist]
    #         benign_validators = [validator for validator in all_validators if validator in self.benign_namelist]
    #         adv_validator_flags = [validator_flags[validator] for validator in range(len(all_validators)) if all_validators[validator] in self.adversarial_namelist]
    #         benign_validator_flags = [validator_flags[validator] for validator in range(len(all_validators)) if all_validators[validator] in self.benign_namelist]
    #         tp = sum(adv_validator_flags)
    #         fp = sum(benign_validator_flags)
    #         tn = sum([1-flag for flag in adv_validator_flags])
    #         fn = sum([1-flag for flag in benign_validator_flags])

    #         adv_adjusted_clients = [adjusted_clients[i] for i in range(len(names)) if names[i] in self.adversarial_namelist]
    #         benign_adjusted_clients = [adjusted_clients[i] for i in range(len(names)) if names[i] in self.benign_namelist]
    #         adv_pcnt = len(adv_validators)/len(all_validators)
    #         benign_pcnt = len(benign_validators)/len(all_validators)
    #         utils.csv_record.validator_pcnt_result.append([adv_pcnt, benign_pcnt, tp, fp, tn, fn, adv_adjusted_clients, benign_adjusted_clients])
    #         # logger.info(f'adversarial validators: {[validator for validator in all_validators if validator in self.adversarial_namelist]}')
    #         # logger.info(f'benign validators: {[validator for validator in all_validators if validator in self.benign_namelist]}')
    #         logger.info(f'adversarial validators: {adv_validators}')
    #         logger.info(f'benign validators: {benign_validators}')
    #     except:
    #         utils.csv_record.validator_pcnt_result.append([0, 0])
    #         pass
    #     logger.info(f'adversarial wv: {[self.print_util(names[iidx], wv[iidx]) for iidx in range(len(wv)) if names[iidx] in self.adversarial_namelist]}')
    #     logger.info(f'benign wv: {[self.print_util(names[iidx], wv[iidx]) for iidx in range(len(wv)) if names[iidx] in self.benign_namelist]}')
    #     logger.info(f'all_val_score: {self.all_val_score}')
    #     logger.info(f'all_mal_val_score: {[(client_id, self.all_val_score[client_id]) for client_id in self.adversarial_namelist if client_id in self.all_val_score]}')
    #     logger.info(f'all_benign_val_score: {[(client_id, self.all_val_score[client_id]) for client_id in self.benign_namelist if client_id in self.all_val_score]}')


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

        logger.info(f'names: {names}')
        adv_indices = [idx for idx, name in enumerate(names) if name in self.adversarial_namelist]
        benign_indices = [idx for idx, name in enumerate(names) if name in self.benign_namelist]
        src_class_indices = [40 + idx for idx in range(10)]
        non_src_class_indices = [idx for idx, name in enumerate(names) if name not in src_class_indices]
        non_src_class_indices = non_src_class_indices[:-1]

        # only for testing purpose
        grads = [self.flatten_gradient(client_grad) for client_grad in client_grads]
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



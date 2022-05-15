from re import I
from shutil import copyfile

import math
import torch

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
import test

from torch.utils.data import SubsetRandomSampler
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, cosine_similarity
from sklearn.metrics import silhouette_score
from scipy.stats import stats
from tqdm import tqdm
from termcolor import colored

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
        # self.folder_path = f'saved_models/model_{self.name}_{current_time}_no_models_{self.params["no_models"]}'
        self.folder_path = f'saved_models/model_{self.name}_{current_time}_targetclass_{self.params["tlf_label"]}_no_models_{self.params["no_models"]}'
        try:
            os.mkdir(self.folder_path)
        except FileExistsError:
            logger.info('Folder already exists')
        logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/log.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        logger.info(f'current path: {self.folder_path}')
        if not self.params.get('environment_name', False):
            self.params['environment_name'] = self.name

        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path
        self.fg= FoolsGold(use_memory=self.params['fg_use_memory'])

        if self.params['attack_methods'] == config.ATTACK_TLF:
            if self.params['type'] in config.target_class_dict.keys():
                self.source_class = config.target_class_dict[self.params['type']][self.params['tlf_label']][0]
                self.target_class = config.target_class_dict[self.params['type']][self.params['tlf_label']][1]
            else:
                self.source_class = int(self.params['tlf_label'])
                self.target_class = 9 - self.source_class

        if self.params['aggregation_methods'] == config.AGGR_AFA:
            self.good_count = [0 for _ in range(self.params['number_of_total_participants'])]
            self.bad_count = [0 for _ in range(self.params['number_of_total_participants'])]
            self.prob_good_model = [0. for _ in range(self.params['number_of_total_participants'])]


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
        # if self.params['aggregation_methods'] == config.AGGR_FOOLSGOLD or self.params['aggregation_methods'] == config.AGGR_FLTRUST:
        if self.params['aggregation_methods'] in [config.AGGR_FOOLSGOLD, config.AGGR_FLTRUST, config.AGGR_OURS, config.AGGR_AFA]:
            updates = dict()
            for i in range(0, len(state_keys)):
                local_model_gradients = epochs_submit_update_dict[state_keys[i]][0] # agg 1 interval
                num_samples = num_samples_dict[state_keys[i]]
                updates[state_keys[i]] = (num_samples, copy.deepcopy(local_model_gradients))
            return None, updates

        else:
            updates = dict()
            for i in range(0, len(state_keys)):
                local_model_update_list = epochs_submit_update_dict[state_keys[i]]
                update= dict()
                num_samples=num_samples_dict[state_keys[i]]

                for name, data in local_model_update_list[0].items():
                    update[name] = torch.zeros_like(data)

                for j in range(0, len(local_model_update_list)):
                    local_model_update_dict= local_model_update_list[j]
                    for name, data in local_model_update_dict.items():
                        weight_accumulator[name].add_(local_model_update_dict[name])
                        update[name].add_(local_model_update_dict[name])
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

            data.add_(update_per_layer)
        return True

    def add_noise(self, noise_level):
        """
        Add noise to the model weights.
        """
        for name, data in self.target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue

            data.add_(self.dp_noise(data, noise_level))
        return True

    def cos_calc_btn_grads(self, l1, l2):
        return torch.dot(l1, l2)/(torch.linalg.norm(l1)+1e-9)/(torch.linalg.norm(l2)+1e-9)

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
    
    def get_optimal_k_for_clustering(self, grads):
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
        for k in range(2, min(len(nets), 15)+1):
            clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='complete').fit(coses)
            labels = clustering.labels_
            print(labels)
            sil.append(silhouette_score(coses, labels, metric='precomputed'))
        # print(sil)
        logger.info(f'Silhouette scores: {sil}')
        return sil.index(max(sil))+2, coses

    def cluster_grads(self, grads, clustering_method='Spectral', clustering_params='grads', k=10):
        # nets = [grad.numpy() for grad in grads]
        # nets = [np.array(grad) for grad in grads]
        nets = grads
        if clustering_params=='lsrs':
            X = self.lsrs
        elif clustering_params=='grads':
            X = nets

        if clustering_method == 'Spectral':
            clustering = SpectralClustering(n_clusters=k, affinity='cosine').fit(X)
        elif clustering_method == 'Agglomerative':
            k, coses = self.get_optimal_k_for_clustering(grads)
            # print(k)
            # clustering = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='complete').fit(X)
            clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='complete').fit(coses)

        clusters = [[] for _ in range(k)]
        for i, label in enumerate(clustering.labels_.tolist()):
            clusters[label].append(i)
        for cluster in clusters:
            cluster.sort()
        clusters.sort(key = lambda cluster: len(cluster), reverse = True)

        # grads_for_clusters = []
        # for cluster in clusters:
        #     grads = [X[i] for i in cluster]
        #     grads_for_clusters.append(grads)
            
        # for i, cluster in enumerate(clusters):
        #     cluster.sort(key = lambda x: self.get_validation_score(X[x], grads_for_clusters[i]))


        if clustering_params=='lsrs': 
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

                for data, target in tqdm(one_class_test_loader, disable=tqdm_disable):
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
            # print(correct_by_class)
        return 100. * correct / len(test_loader.dataset), correct_by_class

    def print_util(self, a, b):
        return str(a) + ': ' + str(b)
    
    def combined_clustering_guided_aggregation(self, target_model, updates, epoch):
        start_time = time.time()
        t = time.time()
        logger.info(f'Started clustering guided aggregation')
        client_grads = []
        alphas = []
        names = []
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
            names.append(name)

        # grads = [self.convert_model_to_param_list(client_grad) for client_grad in client_grads]
        grads = [self.flatten_gradient(client_grad) for client_grad in client_grads]
        logger.info(f'Converted gradients to param list: Time: {time.time() - t}')
        t = time.time()
        # grads = client_grads
        print(names)
        if epoch==1:
            self.validator_trust_scores = [1. for _ in range(self.params['number_of_total_participants'])]
            _, clusters = self.cluster_grads(grads, clustering_params='lsrs')
            self.clusters = clusters
            all_group_nos = []
            for i, cluster in enumerate(self.clusters):
                if len(clusters) > 2:
                    all_group_nos.append(i)
            self.all_group_nos = all_group_nos

            print('Spectral clustering output')
            print(clusters)
            
            logger.info(f'Spectral Clustering: Time: {time.time() - t}')
            t = time.time()

        if epoch <0:
            assert epoch == 0, 'fix epoch {}'.format(len(epoch))

              
        else:
            # agglomerative clustering based validation

            #get agglomerative clusters
            # if epoch<2 or np.random.random_sample() < np.min([0.1, np.exp(-epoch*0.1)/(1. + np.exp(-epoch*0.1))]):
            # if epoch<5:
            #     k = 5
            # else:
            #     k = 2
            # _, self.clusters_agg = self.cluster_grads(grads, clustering_method='Agglomerative', clustering_params='grads', k=10)
            if self.params['no_models'] < 10:
                self.clusters_agg = [[i] for i in range(self.params['no_models'])]
            else:
                _, self.clusters_agg = self.cluster_grads(grads, clustering_method='Agglomerative', clustering_params='grads', k=10)
            
            logger.info(f'Agglomerative Clustering: Time: {time.time() - t}')
            t = time.time()

            clusters_agg = []
            for clstr in self.clusters_agg:
                clstr = [names[c] for c in clstr]
                clusters_agg.append(clstr)
            print(clusters_agg)
            nets = self.local_models
            all_val_acc_list_dict = {}
            print(f'Validating all clients at epoch {epoch}')
            print(f'{self.clusters_agg}_{clusters_agg}')
            val_client_indice_tuples=[]
            for i, val_cluster in enumerate(self.clusters):
                val_trust_scores = [self.validator_trust_scores[vid] for vid in val_cluster]
                # if np.max(val_trust_scores) < 0.01:
                #     for vid in val_cluster:
                #         self.validator_trust_scores[vid] = 1.
                if len(val_cluster) > 2 and np.max(val_trust_scores) > 0.05:
                    # v1, v2 = random.sample(val_cluster, 2)
                    val_trust_scores = np.array(val_trust_scores)/sum(val_trust_scores)
                    v1, v2 = np.random.choice(val_cluster, 2, replace=False, p=val_trust_scores)
                    val_client_indice_tuples.append((i, v1))
                    val_client_indice_tuples.append((i, v2))

            for idx, cluster in enumerate(clusters_agg):
                if len(cluster) == 0:
                    continue
                if len(cluster) != 1:
                    agg_model = self.new_model()
                    agg_model.copy_params(self.target_model.state_dict())

                    cluster_grads = []
                    for iidx, name in enumerate(names):
                        if name in cluster:
                            print(name)
                            cluster_grads.append(client_grads[iidx])
                    
                    wv = 1/len(cluster)
                    # wv = np.ones(self.params['no_models'])
                    # wv = wv/len(wv)
                    logger.info(f'wv: {wv}')
                    # agg_grads = {}
                    # # Iterate through each layer
                    # for name in cluster_grads[0].keys():
                    #     temp = wv * cluster_grads[0][name].cpu().clone()
                    #     # Aggregate gradients for a layer
                    #     for c, cluster_grad in enumerate(cluster_grads):
                    #         if c == 0:
                    #             continue
                    #         temp += wv * cluster_grad[name].cpu()
                    #     agg_grads[name] = temp

                    agg_grads = []
                    # Iterate through each layer
                    for i in range(len(cluster_grads[0])):
                        # assert len(wv) == len(cluster_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
                        temp = wv * cluster_grads[0][i].cpu().clone()
                        # Aggregate gradients for a layer
                        for c, client_grad in enumerate(cluster_grads):
                            if c == 0:
                                continue
                            temp += wv * client_grad[i].cpu()
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
                else:
                    agg_model = self.local_models[cluster[0]]


                val_acc_list=[]
                for iidx, (group_no, val_idx) in enumerate(val_client_indice_tuples):
                    # no validation data exchange between malicious clients
                    # _, _, val_test_loader = train_loaders[epoch][val_idx]
                    # targeted label flip attack where malicious clients coordinate and test against data from the target group's malicious client
                    if val_idx in self.adversarial_namelist:
                        adv_list = np.array(self.adversarial_namelist)
                        _, val_test_loader = self.train_data[np.min(adv_list[adv_list>self.source_class*10])]
                    else:
                        _, val_test_loader = self.train_data[val_idx]
                    if val_idx in self.adversarial_namelist:
                        is_poisonous_validator = True
                    else:
                        is_poisonous_validator = False
                    val_acc, val_acc_by_class = self.validation_test(agg_model, val_test_loader, is_poisonous=is_poisonous_validator, adv_index=0)
                    # logger.info(f'cluster: {cluster}, val_idx: {val_idx}, is_mal_validator: {val_idx in self.adversarial_namelist}, val_acc: {val_acc}')
                    # logger.info(f'cluster: {cluster}, val_idx: {val_idx}, is_mal_validator: {val_idx in self.adversarial_namelist}, val_acc: {val_acc}, val_acc_by_class: {val_acc_by_class}')
                    val_acc_list.append((val_idx, -1, val_acc.item(), val_acc_by_class))
                
                for client in cluster:
                    all_val_acc_list_dict[client] = val_acc_list

            logger.info(f'Validation Done: Time: {time.time() - t}')
            t = time.time()

        def get_group_no(validator_id, clustr):
            for grp_no in range(len(clustr)):
                if validator_id in clustr[grp_no]:
                    return grp_no
            return -1

        def get_min_group_and_score(val_score_by_grp_dict):
            min_val = 100
            min_grp_no = -1
            for grp_no in val_score_by_grp_dict.keys():
                if val_score_by_group_dict[grp_no] < min_val:
                    min_val = val_score_by_group_dict[grp_no]
                    min_grp_no = grp_no
            return min_grp_no, min_val

        all_val_score_by_group_dict={}

        all_val_score = {}
        all_val_score_min_grp={}
        for client_id in names:
            val_score_by_group_dict={}
            val_acc_list = all_val_acc_list_dict[client_id]

            
            # take the one closer to the others
            validators = {}
            for iidx, (val_idx, _, val_acc, val_acc_report) in enumerate(val_acc_list):
                grp_no = get_group_no(val_idx, self.clusters)
                if grp_no in val_score_by_group_dict.keys():
                    # if average
                    # val_score_by_group_dict[grp_no] += val_acc
                    # val_score_by_group_dict[grp_no] /= 2
                    # if minimum
                    val_score_by_group_dict[grp_no].append((val_acc, val_acc_report))
                    validators[grp_no].append(val_idx)
                else:
                    val_score_by_group_dict[grp_no] = [(val_acc, val_acc_report)]
                    validators[grp_no]= [val_idx]

            all_grp_nos = list(val_score_by_group_dict.keys())

            new_val_score_by_group_dict = {}
            for grp_no in all_grp_nos:
                mean_by_class = {}
                for c in range(10):
                    mean_by_class[c] = 0.
                    for group_no in val_score_by_group_dict.keys():
                        if group_no != grp_no:
                            for grp_mm in range(2):
                                mean_by_class[c] += val_score_by_group_dict[group_no][grp_mm][1][c]
                    mean_by_class[c] /= ((len(val_score_by_group_dict.keys())-1) * 2)

                # total_acc = 0.
                # for group_no in all_grp_nos:
                #     for (val_acc, val_acc_report) in val_score_by_group_dict[group_no]:
                #         total_acc += val_acc_report[target_class]

                min_diff_by_class = [0. for c in range(10)]
                for c in range(10):
                    min_diff_by_class[c] = np.min([np.abs(mean_by_class[c] - val_score_by_group_dict[grp_no][0][1][c]), np.abs(mean_by_class[c] - val_score_by_group_dict[grp_no][1][1][c])])
                
                target_class = np.argmax(min_diff_by_class)
                
                val_acc_0 = val_score_by_group_dict[grp_no][0][1][target_class]
                val_acc_1 = val_score_by_group_dict[grp_no][1][1][target_class]
                # total_acc_excluding = total_acc - val_acc_0 - val_acc_1
                mean_acc_excluding = mean_by_class[target_class]
                if min(abs(mean_acc_excluding-val_acc_0),abs(mean_acc_excluding-val_acc_1))>40.:
                    repl_acc = 0.
                    for grp_idx in all_grp_nos:
                        if grp_idx != grp_no:
                            for (val_acc, val_acc_report) in val_score_by_group_dict[grp_idx]:
                                repl_acc += val_acc
                    repl_acc = repl_acc/(2*(len(all_grp_nos)-1))
                    new_val_score_by_group_dict[grp_no] = repl_acc
                    for validator in validators[grp_no]:
                        self.validator_trust_scores[validator] = self.validator_trust_scores[validator]/2
                elif abs(mean_acc_excluding-val_acc_0)<abs(mean_acc_excluding-val_acc_1):
                    if abs(mean_acc_excluding-val_acc_1)>40.:
                        validator = validators[grp_no][1]
                        self.validator_trust_scores[validator] = self.validator_trust_scores[validator]/2
                    new_val_score_by_group_dict[grp_no] = val_score_by_group_dict[grp_no][0][0]
                else:
                    if abs(mean_acc_excluding-val_acc_0)>40.:
                        validator = validators[grp_no][0]
                        self.validator_trust_scores[validator] = self.validator_trust_scores[validator]/2
                    new_val_score_by_group_dict[grp_no] = val_score_by_group_dict[grp_no][1][0]
            # for grp_no in self.all_group_nos:
            #     if grp_no not in new_val_score_by_group_dict.keys():
            #         new_val_score_by_group_dict[grp_no] = -1
            val_score_by_group_dict = new_val_score_by_group_dict
            
            all_val_score_by_group_dict[client_id] = val_score_by_group_dict
            min_val_grp_no, min_val_score = get_min_group_and_score(val_score_by_group_dict)
            all_val_score[client_id] = min_val_score
            all_val_score_min_grp[client_id] = min_val_grp_no

        logger.info(f'Validation scoring by group done: {time.time()-t}')
        t = time.time()
              
        if epoch<1:
            self.global_net.set_param_to_zero()
            self.global_net.aggregate([network.state_dict() for network in self.benign_nets + self.mal_nets])
        elif epoch == 1:

            self.all_val_score = all_val_score
            self.all_val_score_min_grp = all_val_score_min_grp

            aggr_weights = [self.all_val_score[client_id] for client_id in names]
            aggr_weights = np.array(aggr_weights)
            aggr_weights = aggr_weights/np.sum(aggr_weights)

            wv = aggr_weights
            logger.info(f'wv: {wv}')
            agg_grads = {}
            # # Iterate through each layer
            # for name in client_grads[0].keys():
            #     assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
            #     temp = wv[0] * client_grads[0][name].cpu().clone()
            #     # Aggregate gradients for a layer
            #     for c, client_grad in enumerate(client_grads):
            #         if c == 0:
            #             continue
            #         temp += wv[c] * client_grad[name].cpu()
            #     agg_grads[name] = temp

            # target_model.train()
            # # train and update
            # optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
            #                             momentum=self.params['momentum'],
            #                             weight_decay=self.params['decay'])

            # optimizer.zero_grad()
            # for i, (name, params) in enumerate(target_model.named_parameters()):
            #     agg_grads[name]=-agg_grads[name] * self.params["eta"]
            #     if params.requires_grad:
            #         params.grad = agg_grads[name].to(config.device)
            # optimizer.step()
        
        else:
            for client_id in names:
                if client_id in self.all_val_score.keys():
                    prev_val_score = self.all_val_score[client_id]
                    if prev_val_score < 50.:
                        prev_val_grp_no = self.all_val_score_min_grp[client_id]
                        current_val_score_on_that_group = all_val_score_by_group_dict[client_id][prev_val_grp_no]
                        if 0<= current_val_score_on_that_group and current_val_score_on_that_group < 50:
                            all_val_score[client_id] = prev_val_score/2
                            all_val_score_min_grp[client_id] = prev_val_grp_no
            
            for name in all_val_score.keys():
                self.all_val_score[name] = all_val_score[name]
            for name in all_val_score_min_grp.keys():
                self.all_val_score_min_grp[name] = all_val_score_min_grp[name]

            aggr_weights = [self.all_val_score[client_id] for client_id in names]
            aggr_weights = np.array(aggr_weights)
            aggr_weights = aggr_weights/np.sum(aggr_weights)

            wv = aggr_weights
            logger.info(f'wv: {wv}')
            agg_grads = {}
            # Iterate through each layer
            # for name in client_grads[0].keys():
            #     assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
            #     temp = wv[0] * client_grads[0][name].cpu().clone()
            #     # Aggregate gradients for a layer
            #     for c, client_grad in enumerate(client_grads):
            #         if c == 0:
            #             continue
            #         temp += wv[c] * client_grad[name].cpu()
            #         # print(temp)
            #         # temp += wv[c]
            #     # temp = temp / len(client_grads)
            #     agg_grads[name] = temp

            # target_model.train()
            # # train and update
            # optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
            #                             momentum=self.params['momentum'],
            #                             weight_decay=self.params['decay'])

            # optimizer.zero_grad()
            # # print(client_grads[0])
            # print(f'before update {self.convert_model_to_param_list(target_model.state_dict())}')
            # for i, (name, params) in enumerate(target_model.named_parameters()):
            #     agg_grads[name]=-agg_grads[name] * self.params["eta"]
            #     if params.requires_grad:
            #         params.grad = agg_grads[name].to(config.device)
            # optimizer.step()
            # print(f'after update {self.convert_model_to_param_list(target_model.state_dict())}')


        # norms = [torch.linalg.norm(grad).item() for grad in grads]
        norms = [np.linalg.norm(grad) for grad in grads]
        norm_median = np.median(norms)
        clipping_weights = [min(norm_median/norm, 1) for norm in norms]
        wv = [w*c for w,c in zip(wv, clipping_weights)]

        # min_aggr_weights_index = np.argmin(wv)
        # for cluster in self.clusters_agg:
        #     if min_aggr_weights_index in cluster:
        #         for client_id in cluster:
        #             wv[client_id] = 0

        # wv = wv/np.sum(wv)

        cluster_avg_wvs = []
        for cluster_id, cluster in enumerate(self.clusters_agg):
            cluster_avg_wvs.append(np.median([wv[client_id] for client_id in cluster]))
        # min_cluster_avg_wvs_index = np.argmin(cluster_avg_wvs)
        zscore_by_clusters = stats.zscore(cluster_avg_wvs)

        zscore_by_clients = stats.zscore(wv)
        zscore_by_clusters = [np.median([zscore_by_clients[client_id] for client_id in cluster]) for cluster in self.clusters_agg]

        # for client_id in self.clusters_agg[min_cluster_avg_wvs_index]:
        #     wv[client_id] = 0

        for cl_num, cluster in enumerate(self.clusters_agg):
            mal_count = 0
            for client_id in cluster:
                if names[client_id] in self.adversarial_namelist:
                    mal_count += 1
            mal_count = mal_count/len(cluster)

            logger.info(f'{clusters_agg[cl_num]}, {mal_count}, {zscore_by_clusters[cl_num]}')
            if zscore_by_clusters[cl_num] < -1:
                for client_id in cluster:
                    wv[client_id] = 0

        logger.info(f'clipping_weights: {clipping_weights}')
        logger.info(f'adversarial clipping weights: {[self.print_util(names[iidx], clipping_weights[iidx]) for iidx in range(len(clipping_weights)) if names[iidx] in self.adversarial_namelist]}')
        logger.info(f'benign clipping weights: {[self.print_util(names[iidx], clipping_weights[iidx]) for iidx in range(len(clipping_weights)) if names[iidx] in self.benign_namelist]}')
        wv_print_str= '['
        for idx, w in enumerate(wv):
            wv_print_str += ' '
            if names[idx] in self.adversarial_namelist:
                wv_print_str += colored(str(w), 'blue')
            else:
                wv_print_str += str(w)
        wv_print_str += ']'
        print(f'wv: {wv_print_str}')

        # for name in client_grads[0].keys():
        #     assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
        #     temp = wv[0] * client_grads[0][name].cpu().clone()
        #     # Aggregate gradients for a layer
        #     for c, client_grad in enumerate(client_grads):
        #         if c == 0:
        #             continue
        #         temp += wv[c] * client_grad[name].cpu()
        #     agg_grads[name] = temp

        # target_model.train()
        # # train and update
        # optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
        #                             momentum=self.params['momentum'],
        #                             weight_decay=self.params['decay'])

        # optimizer.zero_grad()
        # for i, (name, params) in enumerate(target_model.named_parameters()):
        #     agg_grads[name]=-agg_grads[name] * self.params["eta"]
        #     if params.requires_grad:
        #         params.grad = agg_grads[name].to(config.device)
        # optimizer.step()

        agg_grads = []
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            # assert len(wv) == len(cluster_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
            temp = wv[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == 0:
                    continue
                temp += wv[c] * client_grad[i].cpu()
            # temp = temp / len(client_grads)
            agg_grads.append(temp)

        target_model.train()
        # train and update
        optimizer = torch.optim.SGD(target_model.parameters(), lr=1,
                                    momentum=self.params['momentum'],
                                    weight_decay=self.params['decay'])

        optimizer.zero_grad()
        for i, (name, params) in enumerate(target_model.named_parameters()):
            agg_grads[i]=agg_grads[i] * self.params["eta"]
            if params.requires_grad:
                params.grad = agg_grads[i].to(config.device)
        optimizer.step()

        # noise_level = self.params['sigma'] * norm_median
        # self.add_noise(noise_level=noise_level)
        logger.info(f'Aggregation Done: Time {time.time() - t}')
        t = time.time()
        logger.info(f'wv: {wv}')
        logger.info(f'adversarial wv: {[self.print_util(names[iidx], wv[iidx]) for iidx in range(len(wv)) if names[iidx] in self.adversarial_namelist]}')
        logger.info(f'benign wv: {[self.print_util(names[iidx], wv[iidx]) for iidx in range(len(wv)) if names[iidx] in self.benign_namelist]}')
        logger.info(f'all_val_score: {self.all_val_score}')
        logger.info(f'all_mal_val_score: {[(client_id, self.all_val_score[client_id]) for client_id in self.adversarial_namelist if client_id in self.all_val_score]}')
        logger.info(f'all_benign_val_score: {[(client_id, self.all_val_score[client_id]) for client_id in self.benign_namelist if client_id in self.all_val_score]}')

    #         self.debug_log['val_logs'][epoch]['agglom_cluster_list'] = clusters_agg
    #         self.debug_log['val_logs'][epoch]['all_val_acc_list'] = all_val_acc_list
    #         self.debug_log['val_logs'][epoch]['all_val_scores'] = self.all_val_score
    #         self.debug_log['val_logs'][epoch]['all_val_score_min_grp'] = self.all_val_score_min_grp
    #         self.debug_log['val_logs'][epoch]['aggr_weights'] = aggr_weights
    #         self.debug_log['val_logs'][epoch]['all_val_score_by_group_dict'] = all_val_score_by_group_dict
    #         self.debug_log['val_logs'][epoch]['validator_trust_scores'] = self.validator_trust_scores

    #         print('\n\n\nValidator Trust Scores\n\n', self.validator_trust_scores)

        
    def fltrust(self, target_model, updates):
        client_grads = []
        alphas = []
        names = []
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
            names.append(name)

        grads = [self.flatten_gradient(client_grad) for client_grad in client_grads]
        # grads = client_grads
        clean_server_grad = grads[-1]
        # cos_sims = [self.cos_calc_btn_grads(client_grad, clean_server_grad) for client_grad in grads]
        cos_sims = np.array(cosine_similarity(grads, [clean_server_grad])).flatten()
        logger.info(f'cos_sims: {cos_sims}')

        cos_sims = np.maximum(np.array(cos_sims), 0)
        norm_weights = cos_sims/(np.sum(cos_sims)+1e-9)
        for i in range(len(norm_weights)):
            norm_weights[i] = norm_weights[i] * np.linalg.norm(clean_server_grad) / (np.linalg.norm(grads[i]))

        wv = norm_weights
        # wv = np.ones(self.params['no_models'])
        # wv = wv/len(wv)
        logger.info(f'wv: {wv}')

        agg_grads = []
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            temp = wv[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads[:-1]):
                if c == 0:
                    continue
                temp += wv[c] * client_grad[i].cpu()
                # print(temp)
                # temp += wv[c]
            # temp = temp / len(client_grads)
            agg_grads.append(temp)


        target_model.train()
        # train and update
        optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
                                    momentum=self.params['momentum'],
                                    weight_decay=self.params['decay'])

        optimizer.zero_grad()
        # print(client_grads[0])        
        for i, (name, params) in enumerate(target_model.named_parameters()):
            agg_grads[i]=agg_grads[i] * self.params["eta"]
            if params.requires_grad:
                params.grad = agg_grads[i].to(config.device)
        optimizer.step()
        # noise_level = self.params['sigma']
        # self.add_noise(noise_level=noise_level)
        # utils.csv_record.add_weight_result(names, wv, alpha)
        return True, names, wv

    def calc_prob_for_AFA(self, participant_no):
        return (self.good_count[participant_no]+3)/(self.good_count[participant_no]+self.bad_count[participant_no]+6)

    def afa_method(self, target_model, updates):
        client_grads = []
        alphas = []
        names = []
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
            names.append(name)

        # good_set = set(self.adversarial_namelist + self.benign_namelist)
        good_set = set(np.arange(self.params['no_models']))
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
            agg_grads = []
            # Iterate through each layer
            for i in range(len(client_grads[0])):
                # assert len(wv) == len(cluster_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
                temp = wv[0] * client_grads[0][i].cpu().clone()
                # Aggregate gradients for a layer
                for c, cl_id in enumerate(good_set):
                    if c == 0:
                        continue
                    temp += wv[c] * client_grads[cl_id][i].cpu()
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
        agg_grads = []
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            # assert len(wv) == len(cluster_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
            temp = wv[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, cl_id in enumerate(good_set):
                if c == 0:
                    continue
                temp += wv[c] * client_grads[cl_id][i].cpu()
            # temp = temp / len(wv)
            agg_grads.append(temp)

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
        # print(self.convert_model_to_param_list(grad_state_dict))
        optimizer.step()

        for cl_id in bad_set:
            self.bad_count[names[cl_id]] += 1
        for cl_id in good_set:
            self.good_count[names[cl_id]] += 1

        return True, names, wv

    def foolsgold_update(self,target_model,updates):
        client_grads = []
        alphas = []
        names = []
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
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

        target_model.train()
        # train and update
        # probably need to apply a global learning rate
        optimizer = torch.optim.SGD(target_model.parameters(), lr=1,
                                    momentum=self.params['momentum'],
                                    weight_decay=self.params['decay'])

        optimizer.zero_grad()
        # print(client_grads[0])
        agg_grads, wv,alpha = self.fg.aggregate_gradients(client_grads,names)
        grad_state_dict = {}
        for i, (name, params) in enumerate(target_model.named_parameters()):
            grad_state_dict[name] = agg_grads[i]
            agg_grads[i]=agg_grads[i] * self.params["eta"]
            if params.requires_grad:
                params.grad = agg_grads[i].to(config.device)
        # print(self.convert_model_to_param_list(grad_state_dict))
        optimizer.step()
        wv=wv.tolist()
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
                logger.info(log_entry)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
            logger.info(f'[rfa agg] iter:  {i}, prev_obj_val: {prev_obj_val}, obj_val: {obj_val}, abs dis: { abs(prev_obj_val - obj_val)}')
            logger.info(f'[rfa agg] iter:  {i}, weight: {weights}')
            wv=copy.deepcopy(weights)
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
                data.add_(update_per_layer)
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
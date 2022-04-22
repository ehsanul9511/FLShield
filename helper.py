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

from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

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
        self.folder_path = f'saved_models/model_{self.name}_{current_time}'
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

        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)

        return noised_layer

    def accumulate_weight(self, weight_accumulator, epochs_submit_update_dict, state_keys,num_samples_dict):
        """
         return Args:
             updates: dict of (num_samples, update), where num_samples is the
                 number of training samples corresponding to the update, and update
                 is a list of variable weights
         """
        if self.params['aggregation_methods'] == config.AGGR_FOOLSGOLD or self.params['aggregation_methods'] == config.AGGR_FLTRUST:
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

    def cos_calc_btn_grads(self, l1, l2):
        return torch.dot(l1, l2)/(torch.linalg.norm(l1)+1e-9)/(torch.linalg.norm(l2)+1e-9)

    def convert_model_to_param_list(self, model):
        '''
        num_of_param=0
        for param in model.state_dict().values():
            num_of_param += torch.numel(param)
        

        params=torch.ones([num_of_param])
        '''
        # if torch.typename(model)!='OrderedDict':
        #     model = model.state_dict()

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

    def cluster_grads(self, grads, clustering_method='Spectral', clustering_params='grads', k=10):
        nets = grads
        nets= np.array(nets)
        if clustering_params=='lsrs':
            X = self.lsrs
        elif clustering_params=='grads':
            X = nets

        if clustering_method == 'Spectral':
            clustering = SpectralClustering(n_clusters=k, affinity='cosine').fit(X)
        elif clustering_method == 'Agglomerative':
            clustering = AgglomerativeClustering(n_clusters=2, affinity='cosine', linkage='complete').fit(X)

        clusters = [[] for _ in range(k)]
        for i, label in enumerate(clustering.labels_.tolist()):
            clusters[label].append(i)
        for cluster in clusters:
            cluster.sort()
        clusters.sort(key = lambda cluster: len(cluster), reverse = True)

        grads_for_clusters = []
        for cluster in clusters:
            grads = [X[i] for i in cluster]
            grads_for_clusters.append(grads)
            
        for i, cluster in enumerate(clusters):
            cluster.sort(key = lambda x: self.get_validation_score(X[x], grads_for_clusters[i]))


        if clustering_params=='lsrs': 
            grads_for_clusters = []       
            for cluster in clusters:
                grads = [nets[i] for i in cluster]
                grads_for_clusters.append(grads)

            print('clusters ', clusters)

            for i, cluster in enumerate(clusters):
                cluster.sort(key = lambda x: self.get_average_distance(nets[x], grads_for_clusters[i]))
                # clusters[i] = cluster[:5]
                for idx, cluster_elem in enumerate(clusters[i]):
                    if idx>=5:
                        self.validator_trust_scores[cluster_elem] = 1/idx
            print('clusters ', clusters)

        return clustering.labels_, clusters
    


    # def combined_clustering_guided_aggregation(self, target_model, updates, epoch):
    #     client_grads = []
    #     alphas = []
    #     names = []
    #     for name, data in updates.items():
    #         client_grads.append(data[1])  # gradient
    #         alphas.append(data[0])  # num_samples
    #         names.append(name)

    #     grads = [self.convert_model_to_param_list(client_grad) for client_grad in client_grads]
        
    #     if epoch==0:
    #         _, clusters = self.cluster_grads(epoch, clustering_params='lsrs')
    #         self.clusters = clusters
    #         all_group_nos = []
    #         for i, cluster in enumerate(self.clusters):
    #             if len(clusters) > 2:
    #                 all_group_nos.append(i)
    #         self.all_group_nos = all_group_nos

    #         print('Spectral clustering output')
    #         self.print_clusters(clusters)
    #     if epoch<0:
    #         # def check_in_val_combinations(val_tuples, client_id):
    #         #     for (_, val_id) in val_tuples:
    #         #         if client_id == val_id:
    #         #             return True
    #         #     return False

    #         all_val_acc_list = []
    #         print(f'Validating all clients at epoch {epoch}')
    #         for idx, net in enumerate(tqdm(self.benign_nets + self.mal_nets, disable=tqdm_disable)):
    #             # combination_index = random.randint(0, self.num_of_val_client_combinations-1)
    #             # val_client_indice_tuples = self.val_client_indice_tuples_list[combination_index]
    #             # while check_in_val_combinations(val_client_indice_tuples, idx):
    #             #     combination_index = random.randint(0, self.num_of_val_client_combinations-1)
    #             #     val_client_indice_tuples = self.val_client_indice_tuples_list[combination_index]
    #             val_client_indice_tuples=[]
    #             for i, cluster in enumerate(self.clusters):
    #                 if len(cluster) > 2:
    #                     v1, v2 = random.sample(cluster, 2)
    #                     val_client_indice_tuples.append((i, v1))
    #                     val_client_indice_tuples.append((i, v2))

    #             val_acc_list=[]
    #             for iidx, (group_no, val_idx) in enumerate(val_client_indice_tuples):
    #                 _, _, val_test_loader = train_loaders[epoch][val_idx]
    #                 val_acc, val_acc_by_class = validation_test(net, val_test_loader, is_poisonous=(epoch>=self.poison_starts_at_epoch) and (val_idx>self.num_of_benign_nets))
    #                 # print(idx, val_idx, cluster_dict[group_no], val_acc)
    #                 # val_acc_mat[idx][iidx] = val_acc
    #                 # if idx in cluster_dict[group_no]:
    #                 #     val_acc_same_group[idx][iidx] = 1
    #                 if idx in self.clusters[group_no]:
    #                     val_acc_same_group = 1
    #                 else:
    #                     val_acc_same_group = 0
    #                 val_acc_list.append((val_idx, val_acc_same_group, val_acc.item(), val_acc_by_class))
    #             all_val_acc_list.append(val_acc_list)
    #         # self.debug_log['val_logs'][epoch]['val_acc_mat'] = val_acc_mat
    #         # self.debug_log['val_logs'][epoch]['val_acc_same_group'] = val_acc_same_group
    #         # self.debug_log['val_logs'][epoch]['val_client_indice_tuples_list'] = self.val_client_indice_tuples_list
    #         # self.debug_log['val_logs'][epoch]['cluster_dict'] = self.cluster_dict
    #         self.debug_log['val_logs'][epoch]['all_val_acc_list'] = all_val_acc_list


              
    #     else:
    #         # agglomerative clustering based validation

    #         #get agglomerative clusters
    #         if epoch<2 or np.random.random_sample() < np.min([0.1, np.exp(-epoch*0.1)/(1. + np.exp(-epoch*0.1))]):
    #             _, self.clusters_agg = self.cluster_grads(epoch, clustering_method='Agglomerative')
    #         clusters_agg = self.clusters_agg
    #         self.print_clusters(clusters_agg)
    #         nets = self.benign_nets + self.mal_nets
    #         all_val_acc_list_dict = {}
    #         print(f'Validating all clients at epoch {epoch}')
    #         val_client_indice_tuples=[]
    #         for i, val_cluster in enumerate(self.clusters):
    #             val_trust_scores = [self.validator_trust_scores[vid] for vid in val_cluster]
    #             # if np.max(val_trust_scores) < 0.01:
    #             #     for vid in val_cluster:
    #             #         self.validator_trust_scores[vid] = 1.
    #             if len(val_cluster) > 2 and np.max(val_trust_scores) > 0.05:
    #                 # v1, v2 = random.sample(val_cluster, 2)
    #                 val_trust_scores = np.array(val_trust_scores)/sum(val_trust_scores)
    #                 v1, v2 = np.random.choice(val_cluster, 2, replace=False, p=val_trust_scores)
    #                 val_client_indice_tuples.append((i, v1))
    #                 val_client_indice_tuples.append((i, v2))

    #         for idx, cluster in enumerate(tqdm(clusters_agg, disable=tqdm_disable)):
    #             nets_in_cluster = [nets[iidx].state_dict() for iidx in cluster]
    #             cluster_avg_net = CNN()
    #             cluster_avg_net.set_param_to_zero()
    #             cluster_avg_net.aggregate(nets_in_cluster)


    #             val_acc_list=[]
    #             for iidx, (group_no, val_idx) in enumerate(val_client_indice_tuples):
    #                 # no validation data exchange between malicious clients
    #                 # _, _, val_test_loader = train_loaders[epoch][val_idx]
    #                 # targeted label flip attack where malicious clients coordinate and test against data from the target group's malicious client
    #                 if val_idx<self.num_of_benign_nets or aa0==0:
    #                     _, _, val_test_loader = train_loaders[epoch][val_idx]
    #                 else:
    #                     first_target_group_mal_index = np.where(np.array(copylist)==target_class)[0][aa0]
    #                     _, _, val_test_loader = train_loaders[epoch][first_target_group_mal_index]
    #                 val_acc, val_acc_by_class = validation_test(cluster_avg_net, val_test_loader, is_poisonous=(epoch>=self.poison_starts_at_epoch) and (val_idx>self.num_of_benign_nets))
    #                 # if val_idx>=self.num_of_benign_nets:
    #                 #     print(val_acc, val_acc_by_class)
    #                 # print(idx, val_idx, cluster_dict[group_no], val_acc)
    #                 # val_acc_mat[idx][iidx] = val_acc
    #                 # if idx in cluster_dict[group_no]:
    #                 #     val_acc_same_group[idx][iidx] = 1
    #                 val_acc_list.append((val_idx, -1, val_acc.item(), val_acc_by_class))
                
    #             for client in cluster:
    #                 all_val_acc_list_dict[client] = val_acc_list

    #         all_val_acc_list = []
    #         for idx in range(self.num_of_benign_nets+self.num_of_mal_nets):
    #             all_val_acc_list.append(all_val_acc_list_dict[idx])

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

    #     all_val_score_by_group_dict=[]

    #     all_val_score = []
    #     all_val_score_min_grp=[]
    #     for client_id in range(self.num_of_benign_nets + self.num_of_mal_nets):
    #         val_score_by_group_dict={}
    #         val_acc_list = all_val_acc_list[client_id]
    #         # take minimum of two
    #         # for iidx, (val_idx, _, val_acc, _) in enumerate(val_acc_list):
    #         #     grp_no = get_group_no(val_idx, self.clusters)
    #         #     if grp_no in val_score_by_group_dict.keys():
    #         #         # if average
    #         #         # val_score_by_group_dict[grp_no] += val_acc
    #         #         # val_score_by_group_dict[grp_no] /= 2
    #         #         # if minimum
    #         #         val_score_by_group_dict[grp_no] = np.minimum(val_score_by_group_dict[grp_no], val_acc)

    #         #     else:
    #         #         val_score_by_group_dict[grp_no] = val_acc

            
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
    #         total_acc = 0.
    #         for grp_no in all_grp_nos:
    #             for (val_acc, val_acc_report) in val_score_by_group_dict[grp_no]:
    #                 total_acc += val_acc_report[target_class]

    #         new_val_score_by_group_dict = {}
    #         for grp_no in all_grp_nos:
    #             val_acc_0 = val_score_by_group_dict[grp_no][0][1][target_class]
    #             val_acc_1 = val_score_by_group_dict[grp_no][1][1][target_class]
    #             total_acc_excluding = total_acc - val_acc_0 - val_acc_1
    #             mean_acc_excluding = total_acc_excluding/(2*(len(all_grp_nos)-1))
    #             if min(abs(mean_acc_excluding-val_acc_0),abs(mean_acc_excluding-val_acc_1))>40.:
    #                 repl_acc = 0.
    #                 for grp_idx in all_grp_nos:
    #                     if grp_idx != grp_no:
    #                         for (val_acc, val_acc_report) in val_score_by_group_dict[grp_idx]:
    #                             repl_acc += val_acc
    #                 repl_acc = repl_acc/(2*(len(all_grp_nos)-1))
    #                 new_val_score_by_group_dict[grp_no] = repl_acc
    #                 for validator in validators[grp_no]:
    #                     self.validator_trust_scores[validator] = self.validator_trust_scores[validator]/2
    #             elif abs(mean_acc_excluding-val_acc_0)<abs(mean_acc_excluding-val_acc_1):
    #                 if abs(mean_acc_excluding-val_acc_1)>40.:
    #                     validator = validators[grp_no][1]
    #                     self.validator_trust_scores[validator] = self.validator_trust_scores[validator]/2
    #                 new_val_score_by_group_dict[grp_no] = val_score_by_group_dict[grp_no][0][0]
    #             else:
    #                 if abs(mean_acc_excluding-val_acc_0)>40.:
    #                     validator = validators[grp_no][0]
    #                     self.validator_trust_scores[validator] = self.validator_trust_scores[validator]/2
    #                 new_val_score_by_group_dict[grp_no] = val_score_by_group_dict[grp_no][1][0]
    #         for grp_no in self.all_group_nos:
    #             if grp_no not in new_val_score_by_group_dict.keys():
    #                 new_val_score_by_group_dict[grp_no] = -1
    #         val_score_by_group_dict = new_val_score_by_group_dict
                            
    #         all_val_score_by_group_dict.append(val_score_by_group_dict)
    #         min_val_grp_no, min_val_score = get_min_group_and_score(val_score_by_group_dict)
    #         all_val_score.append(min_val_score)
    #         all_val_score_min_grp.append(min_val_grp_no)
              
    #     if epoch<0:
    #         self.global_net.set_param_to_zero()
    #         self.global_net.aggregate([network.state_dict() for network in self.benign_nets + self.mal_nets])
    #     elif epoch == 0:

    #         self.all_val_score = all_val_score
    #         self.all_val_score_min_grp = all_val_score_min_grp

    #         aggr_weights = np.array(all_val_score)
    #         aggr_weights = aggr_weights/np.sum(aggr_weights)

    #         self.global_net.set_param_to_zero()
    #         self.global_net.aggregate([net.state_dict() for net in self.benign_nets + self.mal_nets],
    #             aggr_weights=aggr_weights
    #         )
        
    #     else:
    #         for client_id in range(self.num_of_benign_nets + self.num_of_mal_nets):
    #             prev_val_score = self.all_val_score[client_id]
    #             if prev_val_score < 50.:
    #                 prev_val_grp_no = self.all_val_score_min_grp[client_id]
    #                 current_val_score_on_that_group = all_val_score_by_group_dict[client_id][prev_val_grp_no]
    #                 if 0<= current_val_score_on_that_group and current_val_score_on_that_group < 50:
    #                     all_val_score[client_id] = prev_val_score/2
    #                     all_val_score_min_grp[client_id] = prev_val_grp_no
    #         self.all_val_score = all_val_score
    #         self.all_val_score_min_grp = all_val_score_min_grp

    #         aggr_weights = np.array(all_val_score)
    #         aggr_weights = np.minimum(aggr_weights, 50.)
    #         aggr_weights = aggr_weights/np.sum(aggr_weights)

    #         self.global_net.set_param_to_zero()
    #         self.global_net.aggregate([net.state_dict() for net in self.benign_nets + self.mal_nets],
    #             aggr_weights=aggr_weights
    #         )

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

        grads = [self.convert_model_to_param_list(client_grad) for client_grad in client_grads]
        clean_server_grad = grads[-1]
        cos_sims = [self.cos_calc_btn_grads(client_grad, clean_server_grad) for client_grad in grads]
        logger.info(f'cos_sims: {cos_sims}')

        cos_sims = np.maximum(np.array(cos_sims), 0)
        norm_weights = cos_sims/(np.sum(cos_sims)+1e-9)
        for i in range(len(norm_weights)):
            norm_weights[i] = norm_weights[i] * torch.linalg.norm(clean_server_grad) / (torch.linalg.norm(grads[i]))

        wv = norm_weights
        # wv = np.ones(self.params['no_models'])
        # wv = wv/len(wv)
        logger.info(f'wv: {wv}')
        agg_grads = {}
        # Iterate through each layer
        for name in client_grads[0].keys():
            assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
            temp = wv[0] * client_grads[0][name].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == 0:
                    continue
                temp += wv[c] * client_grad[name].cpu()
                # print(temp)
                # temp += wv[c]
            # temp = temp / len(client_grads)
            agg_grads[name] = temp

        print(self.convert_model_to_param_list(agg_grads))


        target_model.train()
        # train and update
        optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
                                    momentum=self.params['momentum'],
                                    weight_decay=self.params['decay'])

        optimizer.zero_grad()
        # print(client_grads[0])
        print(f'before update {self.convert_model_to_param_list(target_model.state_dict())}')
        for i, (name, params) in enumerate(target_model.named_parameters()):
            agg_grads[name]=-agg_grads[name] * self.params["eta"]
            if params.requires_grad:
                params.grad = agg_grads[name].to(config.device)
        optimizer.step()
        print(f'after update {self.convert_model_to_param_list(target_model.state_dict())}')
        # utils.csv_record.add_weight_result(names, wv, alpha)
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
            if _name in self.params['adversary_list']:
                adver_ratio += alphas[i]
        adver_ratio = adver_ratio / sum(alphas)
        poison_fraction = adver_ratio * self.params['poisoning_per_batch'] / self.params['batch_size']
        logger.info(f'[foolsgold agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[foolsgold agg] considering poison per batch poison_fraction: {poison_fraction}')

        target_model.train()
        # train and update
        optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
                                    momentum=self.params['momentum'],
                                    weight_decay=self.params['decay'])

        optimizer.zero_grad()
        print(client_grads[0])
        agg_grads, wv,alpha = self.fg.aggregate_gradients(client_grads,names)
        grad_state_dict = {}
        for i, (name, params) in enumerate(target_model.named_parameters()):
            grad_state_dict[name] = agg_grads[i]
            agg_grads[i]=agg_grads[i] * self.params["eta"]
            if params.requires_grad:
                params.grad = agg_grads[i].to(config.device)
        print(self.convert_model_to_param_list(grad_state_dict))
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
            if _name in self.params['adversary_list']:
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
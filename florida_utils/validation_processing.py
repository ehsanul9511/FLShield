
# import EllipticEnvelope, KMeans, numpy, torch
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans
from scipy.stats import zscore
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance
from tqdm import tqdm
import copy
import os
import sys
if __name__ != '__main__':
    import utils.csv_record as csv_record


import logging
logger = logging.getLogger("logger")

class KMeans_Torch(torch.nn.Module):
    def __init__(self, n_clusters=2, max_iter=100):
        super(KMeans_Torch, self).__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_centers = None

    def forward(self, x):
        # initialize cluster centers randomly
        self.cluster_centers = x[torch.randperm(x.size(0))[:self.n_clusters]]
        for i in range(self.max_iter):
            # compute distances to cluster centers
            distances = torch.norm(x[:, None] - self.cluster_centers, dim=2)
            # assign data points to closest cluster
            cluster_assignments = torch.argmin(distances, dim=1)
            # recompute cluster centers as mean of assigned data points
            for j in range(self.n_clusters):
                cluster_points = x[cluster_assignments == j]
                if cluster_points.size(0) > 0:
                    self.cluster_centers[j] = torch.mean(cluster_points, dim=0)
        maj_assignment = (cluster_assignments.float().mean() > 0.5).long()
        x = torch.mean(x[cluster_assignments==maj_assignment], dim=0)
        return x

class ValidationProcessor:
    def __init__(self, validation_container):
        self.params = validation_container['params']
        logger.info(f'params type {type(self.params)}')
        self.validation_container = validation_container

    def plot_PCA(self, val_tensor):
        mean = torch.mean(val_tensor, dim=0)
        data_centered = val_tensor - mean
        U, S, V = torch.svd(data_centered)

        # Project data to 2 dimensions
        components = V[:, :2]
        data_2d = data_centered @ components
        torch.save(torch.tensor(data_2d), 'val_tensor.pt')
        plt.scatter(val_tensor[:,0], val_tensor[:,1])
        # plt.show()
        plt.savefig('val_tensor.png')

    def filter_LIPC(self, val_array_normalized):
        outlier_prediction = [1 for _ in range(len(val_array_normalized))]
        outlier_detector_type = self.params['outlier_detector_type']
        # outlier_detector_type = 'EllipticEnvelope'
        if outlier_detector_type == 'EllipticEnvelope':
            contamination_level = self.params['contamination_level'] if self.params['contamination_level'] is not None else 0.5
            try:
                outlier_detector = EllipticEnvelope(contamination=contamination_level)
                # outlier_detector = LocalOutlierFactor(n_neighbors=int(len(names)/3), contamination=0.5)
                # logger.info(f'val array normalized', val_array_normalized)
                outlier_prediction = outlier_detector.fit_predict(np.array(val_array_normalized).reshape(-1,1))
                # logger.info(f'outlier prediction {outlier_prediction}')
            except:
                logger.info(f'val array normalized', val_array_normalized)
                outlier_prediction = [1 for _ in range(len(val_array_normalized))]
                pass
        elif outlier_detector_type == 'no_detector':
            outlier_prediction = [1 for _ in range(len(val_array_normalized))]
        else:
            # do KMeans
            try:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(val_array_normalized).reshape(-1,1))
                outlier_prediction = kmeans.labels_
                # map the labels to 1 and -1
                outlier_prediction = [1 if x==0 else -1 for x in outlier_prediction]
            except:
                logger.info(f'val array normalized', val_array_normalized)
                outlier_prediction = [1 for _ in range(len(val_array_normalized))]
                pass
        return outlier_prediction


    # def fed_validation(self, num_of_clusters, num_of_classes, names, evaluations_of_clusters, count_of_class_for_validator):
    #     s = np.zeros((num_of_clusters, num_of_classes))
    #     for cluster_idx in range(num_of_clusters):

    #         for c_idx in range(len(evaluations_of_clusters[cluster_idx][names[0]])):
    #             # val_array = np.array([evaluations_of_clusters[cluster_idx][name][c_idx] for name in names])
    #             val_array_normalized = np.array([evaluations_of_clusters[cluster_idx][name][c_idx]/count_of_class_for_validator[name][c_idx] if count_of_class_for_validator[name][c_idx] !=0 else 0 for name in names])

    #             val_array_normalized[np.isnan(val_array_normalized)] = 0

    #             outlier_prediction = self.filter_LIPC(val_array_normalized)

    #             # eval_sum = np.sum([val_array[i] for i in range(len(val_array)) if outlier_prediction[i]!=-1])
    #             # total_count_of_class.append(np.sum([count_of_class_for_validator[names[i]][c_idx] for i in range(len(names)) if outlier_prediction[i]!=-1]))

    #             s[cluster_idx][c_idx] = np.mean([val_array_normalized[i] for i in range(len(val_array_normalized)) if outlier_prediction[i]!=-1])

    #     return s

    def fed_validation(self, num_of_clusters, num_of_classes, names, evaluations_of_clusters, count_of_class_for_validator):
        eval_tensor = self.generate_tensor(num_of_clusters, num_of_classes, names, evaluations_of_clusters, count_of_class_for_validator)

        if self.params['save_validation_tensor'] or True:
            torch.save(eval_tensor, f'{self.params["location"]}/eval_tensor.pt') if self.params['location'] is not None else torch.save(eval_tensor, 'eval_tensor.pt')

        eval_tensor = eval_tensor.reshape(len(names), -1)

        filter_layer = KMeans_Torch(n_clusters=2)

        eval_tensor = filter_layer(eval_tensor)

        eval_tensor = eval_tensor.reshape(num_of_clusters, num_of_classes)

        return eval_tensor.detach().numpy()

    def generate_tensor(self, num_of_clusters, num_of_classes, names, evaluations_of_clusters, count_of_class_for_validator, null_value=0):
        eval_tensor = torch.zeros((len(names), num_of_clusters, num_of_classes))
        for i in range(len(names)):
            for j in range(num_of_clusters):
                for k in range(num_of_classes):
                    eval_tensor[i][j][k] = evaluations_of_clusters[j][names[i]][k]/count_of_class_for_validator[names[i]][k] if count_of_class_for_validator[names[i]][k] !=0 else 0

        return eval_tensor

    def naive_malicious_val_crafting(self, evaluations_of_clusters, count_of_class_for_validator, num_of_clusters, num_of_classes, names):
        for cluster_i in range(num_of_clusters):
            if self.cluster_maliciousness[cluster_i] > 0:
                cluster_j = self.argsort_result[min(num_of_clusters//2 + 1, num_of_clusters-1)]
                cl_p = self.lowest_performing_classes[cluster_j]
                cl = self.lowest_performing_classes[cluster_i]
                value_1 = np.sum([evaluations_of_clusters[cluster_j][val_idx][cl_p]/count_of_class_for_validator[val_idx][cl_p] for val_idx in names])
                value_2 = np.sum([evaluations_of_clusters[cluster_i][val_idx][cl]/count_of_class_for_validator[val_idx][cl] for val_idx in names if val_idx in self.benign_namelist])
                mod_val = (value_1-value_2)/len(set(names).intersection(set(self.adversarial_namelist)))
                for val_idx in set(names).intersection(set(self.adversarial_namelist)):
                    evaluations_of_clusters[cluster_i][val_idx][cl] = mod_val * count_of_class_for_validator[val_idx][cl]
            else:
                cluster_j = self.argsort_result[num_of_clusters//2 - 1]
                cl_p = self.lowest_performing_classes[cluster_j]
                cl = self.lowest_performing_classes[cluster_i]
                value_1 = np.sum([evaluations_of_clusters[cluster_j][val_idx][cl_p]/count_of_class_for_validator[val_idx][cl_p] for val_idx in names])
                value_2 = np.sum([evaluations_of_clusters[cluster_i][val_idx][cl]/count_of_class_for_validator[val_idx][cl] for val_idx in names if val_idx in self.benign_namelist])
                mod_val = (value_1-value_2)/len(set(names).intersection(set(self.adversarial_namelist)))
                for val_idx in set(names).intersection(set(self.adversarial_namelist)):
                    evaluations_of_clusters[cluster_i][val_idx][cl] = mod_val * count_of_class_for_validator[val_idx][cl]

        
        eval_tensor = self.generate_tensor(num_of_clusters, num_of_classes, names, evaluations_of_clusters, count_of_class_for_validator)

        logger.info(f'saving eval_tensor to naive_eval_tensor.pt')
        torch.save(eval_tensor, f'naive_eval_tensor.pt')
        self.plot_PCA(eval_tensor.reshape(len(names), -1))

        return evaluations_of_clusters

    def adaptive_malicious_val_crafting(self, evaluations_of_clusters, count_of_class_for_validator, num_of_clusters, num_of_classes, names, dist_sim_coeff=0.5):
        cluster_losses = []
        distance_losses = []
        mal_val_impacts = []
        # create a torch tensor for the evaluation_of_clusters
        eval_tensor = self.generate_tensor(num_of_clusters, num_of_classes, names, evaluations_of_clusters, count_of_class_for_validator)

        # torch.save(eval_tensor, 'eval_tensor.pt')

        # logger.info(f'eval_tensor: {eval_tensor}')

        # split into two tensors
        num_of_mal_validators = len(self.adversarial_namelist)
        mal_eval_tensor = eval_tensor[:num_of_mal_validators]
        benign_eval_tensor = eval_tensor[num_of_mal_validators:]

        mal_eval_tensor.requires_grad = True

        # loss_fun(mal_eval_tensor, benign_eval_tensor)

        # mal_cluster_weight_tensor = torch.ones((len(self.mal_cluster_indices)), requires_grad=True)
        # benign_cluster_weight_tensor = torch.ones((len(self.benign_cluster_indices)), requires_grad=True)

        minpool = torch.nn.MaxPool1d(num_of_classes, stride=num_of_classes)
        minpool_for_validators = torch.nn.MaxPool1d(len(names)-num_of_mal_validators, stride=len(names)-num_of_mal_validators)
        relu_layer = torch.nn.ReLU()

        def mean_euclidean_distance_layer(x):
            x = pairwise_euclidean_distance(x, x[num_of_mal_validators:])
            # logger.info(f'x: {x}')
            x = torch.nan_to_num(x)
            # logger.info(f'x: {x}')
            x = torch.mean(x, dim=1)
            # logger.info(f'x: {x}')
            return x

        def validation_filtering_simulation_layer(x):
            x = x - minpool_for_validators(x[num_of_mal_validators:].reshape(1, -1))
            # logger.info(f'x: {x}')
            x = torch.sign(-x)
            # logger.info(f'x: {x}')
            x = relu_layer(x)
            # logger.info(f'x: {x}')
            return x.T/x.sum()

        best_cluster_loss_min = np.inf
        best_mal_eval_tensor = None

        # autograd
        for e in tqdm(range(1000), disable=False):
            # mean_tensor = torch.mean(torch.cat((mal_eval_tensor, benign_eval_tensor), dim=0), dim=0)

            all_tensor = torch.cat((mal_eval_tensor, benign_eval_tensor), dim=0).reshape(len(names), -1)
            mean_distance_tensor = mean_euclidean_distance_layer(all_tensor)
            validation_filtering_tensor = validation_filtering_simulation_layer(mean_distance_tensor)

            all_flat_tensor = all_tensor.reshape(len(names), -1)
            mean_distance_tensor = mean_euclidean_distance_layer(all_flat_tensor)

            filter_layer = KMeans_Torch(n_clusters=2)

            mean_tensor = filter_layer(all_flat_tensor)

            mean_tensor_2 = mean_tensor.reshape(num_of_clusters, num_of_classes)
            # logger.info(f'validation_filtering_tensor: {validation_filtering_tensor}')
            # mean_tensor_2 = torch.mul(all_tensor, validation_filtering_tensor).sum(dim=0).reshape(num_of_classes, num_of_clusters)

            # logger.info(f'mean_tensor_2: {mean_tensor_2}')

            min_score_by_class_tensor = -minpool(-mean_tensor_2).reshape(num_of_clusters)
            # logger.info(f'min_score_by_class_tensor: {min_score_by_class_tensor}')

            mal_cluster_min_score_by_class_tensor = min_score_by_class_tensor[self.mal_cluster_indices]
            benign_cluster_min_score_by_class_tensor = min_score_by_class_tensor[self.benign_cluster_indices]

            # print(f'mal_cluster_min_score_by_class_tensor: {mal_cluster_min_score_by_class_tensor}')

            # loss = -(1-dist_sim_coeff)*(torch.mul(mal_cluster_min_score_by_class_tensor, mal_cluster_weight_tensor/mal_cluster_weight_tensor.sum()).sum()
            #  -  torch.mul(benign_cluster_min_score_by_class_tensor, benign_cluster_weight_tensor/benign_cluster_weight_tensor.sum()).sum())

            # loss = - mal_cluster_min_score_by_class_tensor.sum() + benign_cluster_min_score_by_class_tensor.sum() 

            loss = torch.linalg.norm(- mal_cluster_min_score_by_class_tensor.mean() + benign_cluster_min_score_by_class_tensor.mean())

            if loss < best_cluster_loss_min:
                best_cluster_loss_min = loss
                best_mal_eval_tensor = mal_eval_tensor.clone().detach()

            # cluster_losses.append(loss.clone().detach().numpy().reshape(1)[0])
            mal_val_impacts.append((validation_filtering_tensor[:num_of_mal_validators].clone().detach().numpy().reshape(num_of_mal_validators)!=0).sum())

            # all_tensor_2 = torch.cat((mal_eval_tensor, benign_eval_tensor), dim=0).reshape(len(names), -1)
            dist_loss = torch.sum(mean_distance_tensor[:num_of_mal_validators])
            # dist_loss = 0

            # for i in range(mal_eval_tensor.shape[0]):
            #     dist_loss +=  dist_sim_coeff * torch.nn.functional.mse_loss(mal_eval_tensor[i], mean_benign_eval_tensor)

            # if e%1 == 0:
            #     logger.info(f'epoch: {e} cluster_loss: {loss}, dist_loss: {dist_loss}')

            distance_losses.append(dist_loss.clone().detach().numpy().reshape(1)[0])
            
            loss = loss + 0.5 * dist_loss
            cluster_losses.append(loss.clone().detach().numpy().reshape(1)[0])
            d_loss = torch.autograd.grad(loss, mal_eval_tensor, create_graph=True)[0]
            # d_loss_2 = torch.autograd.grad(loss, mal_cluster_weight_tensor, create_graph=True)[0]
            # d_loss_3 = torch.autograd.grad(loss, benign_cluster_weight_tensor, create_graph=True)[0]
            # benign_cluster_weight_tensor = benign_cluster_weight_tensor - 0.5 * d_loss_3
            # mal_cluster_weight_tensor = mal_cluster_weight_tensor - 0.5 * d_loss_2
            # logger.info(f'mal_cluster_weight_tensor: {mal_cluster_weight_tensor}')
            mal_eval_tensor = mal_eval_tensor - 0.000001 * d_loss   

        # from matplotlib import pyplot as plt
        # print(cluster_losses[400:500])
        # print(mal_val_impacts[400:500])
        torch.save(torch.tensor(cluster_losses), 'cluster_losses.pt')
        torch.save(torch.tensor(mal_val_impacts), 'mal_val_impacts.pt')
        torch.save(torch.tensor(distance_losses), 'distance_losses.pt')
        # plt.plot(np.convolve(cluster_losses, np.ones(10)/10, mode='valid'))
        # plt.savefig('cluster_losses.png')
        # plt.clf()
        # plt.plot(mal_val_impacts[400:500])
        # plt.savefig('mal_val_impacts.png')
        # loss_fun(mal_eval_tensor, benign_eval_tensor)

        # eval_tensor = torch.cat((mal_eval_tensor, benign_eval_tensor), dim=0)
        # torch.save(eval_tensor, 'eval_tensor.pt')

        evaluations_of_clusters_new = copy.deepcopy(evaluations_of_clusters)
        for i in range(len(best_mal_eval_tensor)):
            # for j in range(len(self.adversarial_namelist)):
            for j in range(num_of_clusters):
                for k in range(num_of_classes):
                    evaluations_of_clusters_new[j][names[i]][k] = best_mal_eval_tensor[i][j][k].detach().numpy()*count_of_class_for_validator[names[i]][k]

        return evaluations_of_clusters_new
            


    def run(self):
        # validation_container is a dict with the following keys:
        # 'evaluations_of_clusters': evaluations_of_clusters,
        # 'count_of_class_for_validator': count_of_class_for_validator,
        # 'names': names,
        # 'num_of_classes': num_of_classes,
        # 'all_validator_evaluations': all_validator_evaluations,
        # 'epoch': epoch,
        # 'params': self.params,
        # 'cluster_maliciousness': cluster_maliciousness,
        # 'num_of_clusters': num_of_clusters,
        # 'benign_namelist': self.benign_namelist,
        # 'adversarial_namelist': self.adversarial_namelist,
        validation_container = self.validation_container

        # unpack the dict
        evaluations_of_clusters = validation_container['evaluations_of_clusters']
        count_of_class_for_validator = validation_container['count_of_class_for_validator']
        names = validation_container['names']
        num_of_classes = validation_container['num_of_classes']
        all_validator_evaluations = validation_container['all_validator_evaluations']
        epoch = validation_container['epoch']
        params = validation_container['params']
        self.cluster_maliciousness = validation_container['cluster_maliciousness']
        num_of_clusters = validation_container['num_of_clusters']
        self.benign_namelist = validation_container['benign_namelist']
        self.adversarial_namelist = validation_container['adversarial_namelist']

        self.benign_namelist = set(self.benign_namelist).intersection(set(names))
        self.adversarial_namelist = set(self.adversarial_namelist).intersection(set(names))

        # get the validation score
        s = self.fed_validation(num_of_clusters, num_of_classes, names, evaluations_of_clusters, count_of_class_for_validator)

        if self.params['injective_florida']:
            self.mal_cluster_indices = [i for i in range(num_of_clusters) if self.cluster_maliciousness[i] > 0]
            self.benign_cluster_indices = [i for i in range(num_of_clusters) if self.cluster_maliciousness[i] == 0]
        else:
            all_indices = np.argsort(self.cluster_maliciousness)
            self.mal_cluster_indices = all_indices[num_of_clusters//2:]
            self.benign_cluster_indices = all_indices[:num_of_clusters//2]
            logger.info(f'self.mal_cluster_indices: {self.mal_cluster_indices}, self.benign_cluster_indices: {self.benign_cluster_indices}')
            self.cluster_maliciousness = [1 if i in self.mal_cluster_indices else 0 for i in range(num_of_clusters)]

        logger.info(f'before malicious validation crafting')
        self.argsort_result = np.argsort([np.min(s[i]) for i in range(len(s))])
        self.lowest_performing_classes = [np.argmin(s[i]) for i in range(len(s))]
        logger.info(f'self.lowest_performing_classes: {self.lowest_performing_classes}')
        logger.info(f'lowest_score_for_each_cluster: {[np.min(s[i]) for i in range(len(s))]}')

        val_score_by_cluster = [np.min(s[i]) for i in range(len(s))]
        # self.argsort_result = np.argsort([np.mean((s[i])**(1/num_of_classes)) for i in range(len(s))])
        # logger.info(f'self.argsort_result: {self.argsort_result}')
        # logger.info(f's: {s}')

        mal_val_type = self.params['mal_val_type']

        if mal_val_type == 'adaptive':
            # mal_pcnts = []

            # for dist_sim_coeff in np.arange(0.1, 0.2, 0.1):
            #     logger.info(f'dist_sim_coeff: {dist_sim_coeff}')
            #     evaluations_of_clusters_temp = self.adaptive_malicious_val_crafting(evaluations_of_clusters, count_of_class_for_validator, num_of_clusters, num_of_classes, names, dist_sim_coeff)

            #     s = self.fed_validation(num_of_clusters, num_of_classes, names, evaluations_of_clusters_temp, count_of_class_for_validator)

            #     self.argsort_result = np.argsort([np.min(s[i]) for i in range(len(s))])
            #     self.lowest_performing_classes = [np.argmin(s[i]) for i in range(len(s))]
            #     # logger.info(f'self.lowest_performing_classes: {self.lowest_performing_classes}')
            #     # logger.info(f'lowest_score_for_each_cluster: {[np.min(s[i]) for i in range(len(s))]}')
            #     # self.argsort_result = np.argsort([np.mean((s[i])**(1/num_of_classes)) for i in range(len(s))])
            #     # logger.info(f'self.argsort_result: {self.argsort_result}')

            #     new_val_score_by_cluster = np.array(val_score_by_cluster) - np.array([np.min(s[i]) for i in range(len(s))])

            #     mal_cluster_score_decrease = np.mean(new_val_score_by_cluster[self.mal_cluster_indices])
            #     benign_cluster_score_decrease = np.mean(new_val_score_by_cluster[self.benign_cluster_indices])


            #     wv_by_cluster = np.zeros(num_of_clusters)
            #     for i in range(num_of_clusters):
            #         if i < len(self.argsort_result)//2:
            #             wv_by_cluster[self.argsort_result[i]] = 0
            #         else:
            #             wv_by_cluster[self.argsort_result[i]] = 1

            #     mal_pcnts.append((wv_by_cluster[self.mal_cluster_indices]==1).sum() / len(self.mal_cluster_indices))

            #     logger.info(f'malicious validation score decrease: {np.mean(new_val_score_by_cluster[self.mal_cluster_indices])}')
            #     logger.info(f'benign validation score decrease: {np.mean(new_val_score_by_cluster[self.benign_cluster_indices])}')

            # best_sim_coeff = np.arange(0.1, 1., 0.1)[np.argmax(mal_pcnts)]
            best_sim_coeff = 0.1
            logger.info(f'best_sim_coeff: {best_sim_coeff}')

            evaluations_of_clusters = self.adaptive_malicious_val_crafting(evaluations_of_clusters, count_of_class_for_validator, num_of_clusters, num_of_classes, names, best_sim_coeff)
        elif mal_val_type == 'naive':
            evaluations_of_clusters = self.naive_malicious_val_crafting(evaluations_of_clusters, count_of_class_for_validator, num_of_clusters, num_of_classes, names)
        else:
            evaluations_of_clusters = evaluations_of_clusters

        logger.info(f'after malicious validation crafting')
        s = self.fed_validation(num_of_clusters, num_of_classes, names, evaluations_of_clusters, count_of_class_for_validator)

        self.argsort_result = np.argsort([np.min(s[i]) for i in range(len(s))])
        self.lowest_performing_classes = [np.argmin(s[i]) for i in range(len(s))]
        logger.info(f'self.lowest_performing_classes: {self.lowest_performing_classes}')
        logger.info(f'lowest_score_for_each_cluster: {[np.min(s[i]) for i in range(len(s))]}')
        # compute zscores
        logger.info(f'zscores: {zscore([np.min(s[i]) for i in range(len(s))])}')
        # self.argsort_result = np.argsort([np.mean((s[i])**(1/num_of_classes)) for i in range(len(s))])
        logger.info(f'self.argsort_result: {self.argsort_result}')

        if self.params['use_mean_LIPC']:
            self.argsort_result = np.argsort([np.mean((s[i])) for i in range(len(s))])

        if 'csv_record' in sys.modules:
            csv_record.epoch_reports[epoch]['lowest_performing_classes'] = csv_record.convert_float32_to_float(self.lowest_performing_classes)
            csv_record.epoch_reports[epoch]['lowest_score_for_each_cluster'] = csv_record.convert_float32_to_float([np.min(s[i]) for i in range(len(s))])
            csv_record.epoch_reports[epoch]['zscores'] = csv_record.convert_float32_to_float(zscore([np.min(s[i]) for i in range(len(s))]))
            csv_record.epoch_reports[epoch]['argsort_result'] = csv_record.convert_float32_to_float(self.argsort_result)


        # s = s[self.argsort_result[len(self.argsort_result)//2:]]
        # logger.info(f's: {s}')
        wv_by_cluster = np.zeros(num_of_clusters)
        for i in range(num_of_clusters):
            if i < len(self.argsort_result)//2:
            # if i < 0.75 * len(self.argsort_result):
                wv_by_cluster[self.argsort_result[i]] = 0
            else:
                wv_by_cluster[self.argsort_result[i]] = 1

        return wv_by_cluster


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # load argparser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=36)
    parser.add_argument('--mal_val_type', type=str, default=None)
    parser.add_argument('--outlier_detector', type=str, default=None)

    # parse args
    args = parser.parse_args()
    
    # look for pkl file at location directory
    if args.location is not None:
        import os
        import pickle
        pkl_path = os.path.join(args.location, f'validation_container_{args.epoch}.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                validation_container = pickle.load(f)

    from collections import defaultdict
    validation_container['params'] = defaultdict(lambda: None, validation_container['params'])

    validation_container['params']['location'] = args.location

    if args.mal_val_type is not None:
        validation_container['params']['mal_val_type'] = args.mal_val_type
    if args.outlier_detector is not None:
        validation_container['params']['outlier_detector_type'] = args.outlier_detector

    # create the validationprocessor
    validation_processor = ValidationProcessor(validation_container)

    # run the validatorprocessor
    wv_by_cluster = validation_processor.run()

    logger.info(f'wv_by_cluster: {wv_by_cluster}')



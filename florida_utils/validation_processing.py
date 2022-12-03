
# import EllipticEnvelope, KMeans, numpy, torch
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans
import numpy as np
import torch

import logging
logger = logging.getLogger("logger")

class ValidationProcessor:
    def __init__(self, validation_container):
        self.params = validation_container['params']
        self.validation_container = validation_container

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
            kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(val_array_normalized).reshape(-1,1))
            outlier_prediction = kmeans.labels_
            # map the labels to 1 and -1
            outlier_prediction = [1 if x==0 else -1 for x in outlier_prediction]
        return outlier_prediction


    def fed_validation(self, num_of_clusters, num_of_classes, names, evaluations_of_clusters, count_of_class_for_validator):
        s = np.zeros((num_of_clusters, num_of_classes))
        for cluster_idx in range(num_of_clusters):

            for c_idx in range(len(evaluations_of_clusters[cluster_idx][names[0]])):
                # val_array = np.array([evaluations_of_clusters[cluster_idx][name][c_idx] for name in names])
                val_array_normalized = np.array([evaluations_of_clusters[cluster_idx][name][c_idx]/count_of_class_for_validator[name][c_idx] if count_of_class_for_validator[name][c_idx] !=0 else 0 for name in names])

                val_array_normalized[np.isnan(val_array_normalized)] = 0

                outlier_prediction = self.filter_LIPC(val_array_normalized)

                # eval_sum = np.sum([val_array[i] for i in range(len(val_array)) if outlier_prediction[i]!=-1])
                # total_count_of_class.append(np.sum([count_of_class_for_validator[names[i]][c_idx] for i in range(len(names)) if outlier_prediction[i]!=-1]))

                s[cluster_idx][c_idx] = np.mean([val_array_normalized[i] for i in range(len(val_array_normalized)) if outlier_prediction[i]!=-1])

        return s

    def naive_malicious_val_crafting(self, evaluations_of_clusters, count_of_class_for_validator, num_of_clusters, num_of_classes, names):
        for cluster_i in range(num_of_clusters):
            if self.cluster_maliciousness[cluster_i] > 0:
                cluster_j = self.argsort_result[num_of_clusters//2 + 1]
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

        return evaluations_of_clusters

    def adaptive_malicious_val_crafting(self, evaluations_of_clusters, count_of_class_for_validator, num_of_clusters, num_of_classes, names):
        # create a torch tensor for the evaluation_of_clusters
        eval_tensor = torch.zeros((len(names), num_of_clusters, num_of_classes))
        for i in range(len(names)):
            for j in range(num_of_clusters):
                for k in range(num_of_classes):
                    eval_tensor[i][j][k] = evaluations_of_clusters[j][names[i]][k]/count_of_class_for_validator[names[i]][k] if count_of_class_for_validator[names[i]][k] !=0 else 0

        # logger.info(f'eval_tensor: {eval_tensor}')

        # split into two tensors
        mal_eval_tensor = eval_tensor[:len(self.adversarial_namelist)]
        benign_eval_tensor = eval_tensor[len(self.adversarial_namelist):]

        mal_eval_tensor.requires_grad = True

        def loss_fun(mal_eval_tensor, benign_eval_tensor):
            # calculate the loss
            loss_1 = 0
            for i in range(len(mal_eval_tensor)):
                for j in range(len(benign_eval_tensor)):
                    # calculate the distance between the two tensors
                    loss_1 += torch.dist(mal_eval_tensor[i], benign_eval_tensor[j], p=2)

            # merge the two tensors
            merged_tensor = torch.cat((mal_eval_tensor, benign_eval_tensor), dim=0)

            # mean of the merged tensor along the first dimension
            mean_tensor = torch.mean(merged_tensor, dim=0)

            loss_2 = 0
            for i in range(len(mean_tensor)):
                for j in range(len(mean_tensor)):
                    # calculate the distance between the two tensors
                    loss_2 += torch.dist(mean_tensor[i], mean_tensor[j], p=2)

            logger.info(f'loss_1: {loss_1}')
            logger.info(f'loss_2: {loss_2}')

            return loss_1 + loss_2

        loss_val = loss_fun(mal_eval_tensor, benign_eval_tensor)
        logger.info(f'loss_val: {loss_val}')

        mal_cluster_indices = [i for i in range(num_of_clusters) if self.cluster_maliciousness[i] > 0]
        benign_cluster_indices = [i for i in range(num_of_clusters) if self.cluster_maliciousness[i] == 0]

        # autograd
        for _ in range(10000):
            mean_tensor = torch.mean(torch.cat((mal_eval_tensor, benign_eval_tensor), dim=0), dim=0)
            # mal_cluster_tensor = torch.cat([mean_tensor[i].reshape(1, num_of_classes) for i in mal_cluster_indices], dim=0)
            # benign_cluster_tensor = torch.cat([mean_tensor[i].reshape(1, num_of_classes) for i in benign_cluster_indices], dim=0)
            mean_benign_eval_tensor = torch.mean(benign_eval_tensor, dim=0)

            mal_cluster_tensor_mean_by_class = torch.mean(torch.cat([mean_tensor[i].reshape(1, num_of_classes) for i in mal_cluster_indices], dim=0), dim=0)
            benign_cluster_tensor_mean_by_class = torch.mean(torch.cat([mean_tensor[i].reshape(1, num_of_classes) for i in benign_cluster_indices], dim=0), dim=0)

            loss =  torch.nn.functional.mse_loss(mal_eval_tensor, torch.cat([mean_benign_eval_tensor.reshape(1, num_of_clusters, num_of_classes) for _ in range(len(mal_eval_tensor))], dim=0)) + torch.linalg.norm(mal_cluster_tensor_mean_by_class) -  torch.linalg.norm(benign_cluster_tensor_mean_by_class)

            # logger.info(f'loss: {loss}')
            d_loss = torch.autograd.grad(loss, mal_eval_tensor, create_graph=True)[0]
            mal_eval_tensor = mal_eval_tensor - 0.5 * d_loss   

        loss_val = loss_fun(mal_eval_tensor, benign_eval_tensor)
        logger.info(f'loss_val: {loss_val}')

        for i in range(len(mal_eval_tensor)):
            # for j in range(len(self.adversarial_namelist)):
            for j in range(num_of_clusters):
                for k in range(num_of_classes):
                    evaluations_of_clusters[j][names[i]][k] = mal_eval_tensor[i][j][k].detach().numpy()*count_of_class_for_validator[names[i]][k]

        return evaluations_of_clusters
            


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

        logger.info(f'before malicious validation crafting')
        self.argsort_result = np.argsort([np.min(s[i]) for i in range(len(s))])
        self.lowest_performing_classes = [np.argmin(s[i]) for i in range(len(s))]
        logger.info(f'self.lowest_performing_classes: {self.lowest_performing_classes}')
        logger.info(f'lowest_score_for_each_cluster: {[np.min(s[i]) for i in range(len(s))]}')
        # self.argsort_result = np.argsort([np.mean((s[i])**(1/num_of_classes)) for i in range(len(s))])
        logger.info(f'self.argsort_result: {self.argsort_result}')
        # logger.info(f's: {s}')

        mal_val_type = self.params['mal_val_type']

        if mal_val_type == 'adaptive':
            evaluations_of_clusters = self.adaptive_malicious_val_crafting(evaluations_of_clusters, count_of_class_for_validator, num_of_clusters, num_of_classes, names)
        else:
            evaluations_of_clusters = self.naive_malicious_val_crafting(evaluations_of_clusters, count_of_class_for_validator, num_of_clusters, num_of_classes, names)


        logger.info(f'after malicious validation crafting')
        s = self.fed_validation(num_of_clusters, num_of_classes, names, evaluations_of_clusters, count_of_class_for_validator)

        self.argsort_result = np.argsort([np.min(s[i]) for i in range(len(s))])
        self.lowest_performing_classes = [np.argmin(s[i]) for i in range(len(s))]
        logger.info(f'self.lowest_performing_classes: {self.lowest_performing_classes}')
        logger.info(f'lowest_score_for_each_cluster: {[np.min(s[i]) for i in range(len(s))]}')
        # self.argsort_result = np.argsort([np.mean((s[i])**(1/num_of_classes)) for i in range(len(s))])
        logger.info(f'self.argsort_result: {self.argsort_result}')

        # s = s[self.argsort_result[len(self.argsort_result)//2:]]
        # logger.info(f's: {s}')
        wv_by_cluster = np.zeros(num_of_clusters)
        for i in range(num_of_clusters):
            if i < len(self.argsort_result)//2:
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

    if args.mal_val_type is not None:
        validation_container['params']['mal_val_type'] = args.mal_val_type
    if args.outlier_detector is not None:
        validation_container['params']['outlier_detector_type'] = args.outlier_detector

    # create the validationprocessor
    validation_processor = ValidationProcessor(validation_container)

    # run the validatorprocessor
    wv_by_cluster = validation_processor.run()

    logger.info(f'wv_by_cluster: {wv_by_cluster}')



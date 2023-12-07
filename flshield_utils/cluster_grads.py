import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, cosine_similarity
from sklearn.metrics import silhouette_score, confusion_matrix
import logging
from collections import defaultdict
import hdbscan

# import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_rand_score

logger = logging.getLogger("logger")

filepath = ""
epoch_global = 0

def cluster_fun(coses, k, clustering_method='KMeans'):
    if clustering_method=='Agglomerative':
        clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='complete').fit(coses)
    elif clustering_method=='KMeans':
        clustering = KMeans(n_clusters=k).fit(coses)
    elif clustering_method=='Spectral':
        clustering = SpectralClustering(n_clusters=k, affinity='precomputed').fit(coses)
    elif clustering_method=='hdbscan':
        clustering = hdbscan.HDBSCAN(min_cluster_size=k, metric='precomputed').fit(np.array(coses, dtype=np.float64))
    else:
        raise NotImplementedError
    return clustering

def get_optimal_k_for_clustering(grads, clustering_method='KMeans'):
    coses = []
    nets = grads
    coses = cosine_distances(nets, nets)
    coses = np.array(coses)
    np.fill_diagonal(coses, 0)
    # logger.info(f'coses: {coses}')
    
    sil= []
    minval = 2

    for k in range(minval, min(len(nets), 15)):
        clustering = cluster_fun(coses, k, clustering_method)
        labels = clustering.labels_
        # print(labels)
        sil.append(silhouette_score(coses, labels, metric='precomputed'))
    # print(sil)
    # logger.info(f'Silhouette scores: {sil}')
    return sil.index(max(sil))+minval, coses

def split_into_two_clusters(grads, clustering_method='KMeans'):
    nets = grads
    X = nets

    if clustering_method == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=len(grads)//2 + 1, min_samples=1, allow_single_cluster=True, metric = 'precomputed')
        cluster_result = clusterer.fit_predict(np.array(cosine_distances(grads), dtype=np.float64))
        clusters = [
            [i for i, cluster in enumerate(cluster_result) if cluster == 0],
            [i for i, cluster in enumerate(cluster_result) if cluster == -1]
        ]
        return cluster_result, clusters

    # k, coses = get_optimal_k_for_clustering(grads, clustering_method)

    # logger.info(f'{filepath}')
    # np.save(f'{filepath}/coses_{epoch_global}.npy', coses)

    coses = cosine_distances(nets, nets)
    k = 2

    clustering = cluster_fun(coses, k, clustering_method)

    clusters = [[] for _ in range(k)]
    for i, label in enumerate(clustering.labels_.tolist()):
        clusters[label].append(i)
    for cluster in clusters:
        cluster.sort()
    clusters.sort(key = lambda cluster: len(cluster), reverse = True)

    return clustering.labels_, clusters


def cluster_grads(grads, clustering_method='KMeans'):
    # nets = [grad.numpy() for grad in grads]
    # nets = [np.array(grad) for grad in grads]
    nets = grads
    X = nets

    if clustering_method == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=len(grads)//2 + 1, min_samples=1, allow_single_cluster=True, metric = 'precomputed')
        cluster_result = clusterer.fit_predict(np.array(cosine_distances(grads), dtype=np.float64))
        clusters = [
            [i for i, cluster in enumerate(cluster_result) if cluster == 0],
            [i for i, cluster in enumerate(cluster_result) if cluster == -1]
        ]
        return cluster_result, clusters

    k, coses = get_optimal_k_for_clustering(grads, clustering_method)

    # logger.info(f'{filepath}')
    # np.save(f'{filepath}/coses_{epoch_global}.npy', coses)

    clustering = cluster_fun(coses, k, clustering_method)

    clusters = [[] for _ in range(k)]
    for i, label in enumerate(clustering.labels_.tolist()):
        clusters[label].append(i)
    for cluster in clusters:
        cluster.sort()
    clusters.sort(key = lambda cluster: len(cluster), reverse = True)

    return clustering.labels_, clusters

def cluster_score_calc(clusters, labels):
    # print(clusters)
    # print(labels)
    num_of_adversaries = sum(labels)
    cluster_maliciousness = [0. for _ in range(len(clusters))]
    for i, cluster in enumerate(clusters):
        for j in cluster:
            if labels[j]==1:
                cluster_maliciousness[i] = 1
                break
    # print(cluster_maliciousness)

    cluster_results = np.zeros(len(labels))
    for i, cluster in enumerate(clusters):
        if cluster_maliciousness[i]==0:
            for j in cluster:
                cluster_results[j] = 1
        else:
            for j in cluster:
                cluster_results[j] = 0

    score = adjusted_rand_score(labels, cluster_results)
    logger.info(f'adjusted rand score: {score}')
    return score


def fidelity_score(clusters, labels):
    num_of_adversaries = sum(labels)

    fidelity_count = 0
    for cluster in clusters:
        cluster_labels = [labels[i] for i in cluster]
        # check how many unique labels are in the cluster
        if len(set(cluster_labels)) == 1:
            fidelity_count += len(cluster)
        else:
            # find which label is the majority between 0 and 1
            majority_label = max(set(cluster_labels), key=cluster_labels.count)
            # count how many of the majority label are in the cluster
            fidelity_count += cluster_labels.count(majority_label)
    logger.info(f'fidelity score: {fidelity_count / len(labels)}')
    return fidelity_count / len(labels)
            

    


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # load argparser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=36)
    parser.add_argument('--clustering_method', type=str, default='Agglomerative')
    parser.add_argument('--comparison_mode', type=bool, default=True)

    # parse args
    args = parser.parse_args()

    # load grads
    grads = np.load(f'{args.location}/grads_{args.epoch}.npy')
    names = np.load(f'{args.location}/names_{args.epoch}.npy')

    # load params
    import yaml
    with open(f'{args.location}/params.yaml', 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params = defaultdict(lambda: None, params)

    num_of_adversaries = params['no_models'] * params[f'number_of_adversary_{params["attack_methods"]}'] / params['number_of_total_participants']
    num_of_adversaries = int(num_of_adversaries)

    actual_labels = np.array([1] * (num_of_adversaries) + [0] * (len(grads) - num_of_adversaries))

    if not args.comparison_mode:

        clustering_method = params['clustering_method'] if params['clustering_method'] else 'Agglomerative'
        
        clustering_method = args.clustering_method if args.clustering_method else clustering_method

        # cluster
        logger.info(f'Clustering Method: {clustering_method}')
        _, clusters = cluster_grads(grads, clustering_method)

        logger.info(f'Validator Groups: {clusters}')
    
    else:
        filepath = f'{args.location}'
        clustering_methods = ['Agglomerative', 'KMeans', 'Spectral', 'hdbscan']
        markers = ['o', 'x', 's']

        import pandas as pd

        df = pd.DataFrame(columns=['epoch']+clustering_methods)

        df.index = df['epoch']

        scores = {}
        for clustering_method in clustering_methods:
            scores[clustering_method] = []

        for epoch in range(args.epoch, args.epoch+10):
            epoch_global = epoch
            logger.info(f'Epoch: {epoch}')
            grads = np.load(f'{args.location}/grads_{epoch}.npy')
            names = np.load(f'{args.location}/names_{epoch}.npy')

            for clustering_method in clustering_methods:
                logger.info(f'Clustering Method: {clustering_method}')
                # _, clusters = cluster_grads(grads, clustering_method)
                _, clusters = split_into_two_clusters(grads, clustering_method)
                logger.info(f'Validator Groups: {clusters}')
                # score = cluster_score_calc(clusters, actual_labels)
                score = fidelity_score(clusters, actual_labels)

                scores[clustering_method].append(score)

                df.loc[epoch, clustering_method] = score

        # df.to_csv(f'{args.location}/cluster_comparison.csv')
        logger.info(df)
        df.to_csv(f'{args.location}/cluster_comparison_n_cluster_2.csv')

        # plot scores
        # import matplotlib.pyplot as plt
        # plt.figure()
        # for clustering_method in clustering_methods:
        #     plt.plot(scores[clustering_method], label=clustering_method, marker=markers[clustering_methods.index(clustering_method)])
        # plt.legend()
        # plt.savefig(f'comparison.png')

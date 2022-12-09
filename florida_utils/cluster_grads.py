import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, cosine_similarity
from sklearn.metrics import silhouette_score, confusion_matrix
import logging
from collections import defaultdict

logger = logging.getLogger("logger")

def cluster_fun(coses, k, clustering_method='Agglomerative'):
    if clustering_method=='Agglomerative':
        clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='complete').fit(coses)
    elif clustering_method=='KMeans':
        clustering = KMeans(n_clusters=k).fit(coses)
    elif clustering_method=='Spectral':
        clustering = SpectralClustering(n_clusters=k, affinity='precomputed').fit(coses)
    return clustering

def get_optimal_k_for_clustering(grads, clustering_method='Agglomerative'):
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


def cluster_grads(grads, clustering_method='Spectral'):
    # nets = [grad.numpy() for grad in grads]
    # nets = [np.array(grad) for grad in grads]
    nets = grads
    X = nets

    k, coses = get_optimal_k_for_clustering(grads, clustering_method)

    clustering = cluster_fun(coses, k, clustering_method)

    clusters = [[] for _ in range(k)]
    for i, label in enumerate(clustering.labels_.tolist()):
        clusters[label].append(i)
    for cluster in clusters:
        cluster.sort()
    clusters.sort(key = lambda cluster: len(cluster), reverse = True)

    return clustering.labels_, clusters


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # load argparser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=36)
    parser.add_argument('--clustering_method', type=str, default='Agglomerative')

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

    clustering_method = params['clustering_method'] if params['clustering_method'] else 'Agglomerative'
    
    clustering_method = args.clustering_method if args.clustering_method else clustering_method

    # cluster
    logger.info(f'Clustering Method: {clustering_method}')
    _, clusters = cluster_grads(grads, clustering_method)

    logger.info(f'Validator Groups: {clusters}')

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import logging

logger = logging.getLogger("logger")

def impute_validation(evaluations_of_clusters, count_of_class_for_validator, num_of_clusters, num_of_classes):
    names = list(evaluations_of_clusters[-1].keys())
    num_of_validators = len(names)

    all_validator_evaluations = [[] for _ in range(num_of_validators)]
    inverse_mapping = [[] for _ in range(num_of_validators)]

    for cluster_idx in range(num_of_clusters):
        for class_idx in range(num_of_classes):
            for validator_idx in range(num_of_validators):
                try:
                    all_validator_evaluations[validator_idx].append(evaluations_of_clusters[cluster_idx][names[validator_idx]][class_idx]/count_of_class_for_validator[names[validator_idx]][class_idx])
                except:
                    all_validator_evaluations[validator_idx].append(np.nan)
                    pass

                inverse_mapping[validator_idx].append((cluster_idx, class_idx))

    # logger.info(f'All validator evaluations: {all_validator_evaluations[1]}')
    
    imputer = IterativeImputer(n_nearest_features = 5, initial_strategy = 'median', random_state = 42)
    all_validator_evaluations = imputer.fit_transform(all_validator_evaluations)

    # logger.info(f'All validator evaluations after imputation: {all_validator_evaluations[1]}')

    for validator_idx in range(num_of_validators):
        for iidx, (cluster_idx, class_idx) in enumerate(inverse_mapping[validator_idx]):
            evaluations_of_clusters[cluster_idx][names[validator_idx]][class_idx] = all_validator_evaluations[validator_idx][iidx]* max(1, count_of_class_for_validator[names[validator_idx]][class_idx])

    return evaluations_of_clusters


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # load argparser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=36)

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

    evaluations_of_clusters = validation_container['evaluations_of_clusters']
    count_of_class_for_validator = validation_container['count_of_class_for_validator']
    num_of_clusters = validation_container['num_of_clusters']
    num_of_classes = validation_container['num_of_classes']

    logger.info(f'evaluations_of_clusters: {evaluations_of_clusters[0]}')

    evaluations_of_clusters = impute_validation(evaluations_of_clusters, count_of_class_for_validator, num_of_clusters, num_of_classes)

    logger.info(f'evaluations_of_clusters: {evaluations_of_clusters[0]}')

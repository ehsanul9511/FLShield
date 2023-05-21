from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
# from fancyimpute import KNN, SoftImpute, SimilarityWeightedAveraging, SimpleFill
import logging

logger = logging.getLogger("logger")

def convert_to_numpy(evaluations_of_clusters, count_of_class_for_validator, num_of_clusters, num_of_classes, names):
    eval_tensor = np.zeros((len(names), num_of_clusters, num_of_classes))
    for i in range(len(names)):
        for j in range(num_of_clusters):
            for k in range(num_of_classes):
                eval_tensor[i][j][k] = evaluations_of_clusters[j][names[i]][k]/count_of_class_for_validator[names[i]][k] if count_of_class_for_validator[names[i]][k] !=0 else np.nan

    return eval_tensor

def calc_mse(actual, pred, missing_mask, method_name):
    mse = ((pred[missing_mask] - actual[missing_mask]) ** 2).mean()
    # logger.info(f"{method_name} MSE: %f" % mse)
    return mse

def impute(eval_tensor, impute_method='iterative'):
    if impute_method == 'iterative':
        imputer = IterativeImputer(n_nearest_features = 5, initial_strategy = 'median', random_state = 42, max_iter = 10, tol = 1e-3)
        eval_tensor = imputer.fit_transform(eval_tensor)
    elif impute_method == 'KNN':
        eval_tensor = KNN(k=3, verbose=False).fit_transform(eval_tensor)
    elif impute_method == 'SoftImpute':
        eval_tensor = SoftImpute(verbose=False).fit_transform(eval_tensor)
    elif impute_method == 'SimilarityWeightedAveraging':
        eval_tensor = SimilarityWeightedAveraging().fit_transform(eval_tensor)
    elif impute_method == 'mean':
        eval_tensor = SimpleFill('mean').fit_transform(eval_tensor)
    elif impute_method == 'median':
        eval_tensor = SimpleFill('median').fit_transform(eval_tensor)
    elif impute_method == 'zero':
        eval_tensor = SimpleFill('zero').fit_transform(eval_tensor)
    elif impute_method == 'random':
        eval_tensor = SimpleFill('random').fit_transform(eval_tensor)
    else:
        raise NotImplementedError
    return eval_tensor


def impute_validation(evaluations_of_clusters, count_of_class_for_validator, names, num_of_clusters, num_of_classes, impute_method='iterative'):
    eval_tensor = convert_to_numpy(evaluations_of_clusters, count_of_class_for_validator, num_of_clusters, num_of_classes, names)

    eval_tensor = eval_tensor.reshape((len(names), num_of_clusters*num_of_classes))

    # find indices in axis 1 where all values are nan
    missing_mask = np.isnan(eval_tensor).all(axis=0)
    if sum(missing_mask) > 0:
        # replace with a very large number
        eval_tensor[:, missing_mask] = 1e6

    eval_tensor = impute(eval_tensor, impute_method)

    eval_tensor = eval_tensor.reshape((len(names), num_of_clusters, num_of_classes))

    for i in range(len(names)):
        for j in range(num_of_clusters):
            for k in range(num_of_classes):
                if count_of_class_for_validator[names[i]][k] != 0:
                    evaluations_of_clusters[j][names[i]][k] = eval_tensor[i][j][k]*count_of_class_for_validator[names[i]][k]
                else:
                    count_of_class_for_validator[names[i]][k] = 10
                    evaluations_of_clusters[j][names[i]][k] = eval_tensor[i][j][k]*count_of_class_for_validator[names[i]][k]

    return evaluations_of_clusters, count_of_class_for_validator


def impute_validation_v0(evaluations_of_clusters, count_of_class_for_validator, num_of_clusters, num_of_classes):
    names = list(evaluations_of_clusters[-1].keys())
    num_of_validators = len(names)

    all_validator_evaluations = [[] for _ in range(num_of_validators)]
    inverse_mapping = [[] for _ in range(num_of_validators)]

    nan_flag = False

    for cluster_idx in range(num_of_clusters):
        for class_idx in range(num_of_classes):
            for validator_idx in range(num_of_validators):
                try:
                    all_validator_evaluations[validator_idx].append(evaluations_of_clusters[cluster_idx][names[validator_idx]][class_idx]/count_of_class_for_validator[names[validator_idx]][class_idx])
                except:
                    all_validator_evaluations[validator_idx].append(np.nan)
                    nan_flag = True
                    pass

                inverse_mapping[validator_idx].append((cluster_idx, class_idx))

    if not nan_flag:
        # randomly drop 10% of the data
        # for validator_idx in range(num_of_validators):
        #     for iidx in np.random.choice(len(all_validator_evaluations[validator_idx]), int(len(all_validator_evaluations[validator_idx])*0.1), replace=False):
        #         all_validator_evaluations[validator_idx][iidx] = np.nan
        logger.info('No nan values found. Returning original evaluations_of_clusters')
        return evaluations_of_clusters

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
    parser.add_argument('--epoch', type=int, default=301)
    parser.add_argument('--missing_percentage', type=float, default=0.1)

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
    names = validation_container['names']

    eval_tensor = convert_to_numpy(evaluations_of_clusters, count_of_class_for_validator, num_of_clusters, num_of_classes, names)

    eval_tensor = eval_tensor.reshape((len(names), num_of_clusters*num_of_classes))

    # method_names = ['KNN', 'SoftImpute', 'iterative', 'SimilarityWeightedAveraging', 'mean', 'median', 'zero', 'random']
    method_names = ['KNN', 'SoftImpute', 'iterative', 'mean', 'median', 'random']

    possible_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    # create a pandas dataframe with the possible values as columns and the method names as rows
    import pandas as pd
    from matplotlib import pyplot as plt
    df = pd.DataFrame(columns=method_names, index=possible_values)
    
    for val in possible_values:
        logger.info(f'missing_percentage: {val}')
        eval_tensor_incomplete = eval_tensor.copy()
        mask = np.random.rand(*eval_tensor_incomplete.shape) <  val # args.missing_percentage
        eval_tensor_incomplete[mask] = np.nan

        for method_name in method_names:
            eval_tensor_imputed = impute(eval_tensor_incomplete, method_name)
            # logger.info(f'eval_tensor_imputed {method_name}: {eval_tensor_imputed[0]}')
            mse = calc_mse(eval_tensor, eval_tensor_imputed, mask, method_name)
            df.loc[val, method_name] = mse

    logger.info(f'df: {df}')
    df.to_csv('imputation.csv')
    df.plot(style=['o', 's' , 'v', 'x', 'd', 'p', 'h', '8'], figsize=(10, 10))
    plt.savefig('imputation.pdf')

    # logger.info(f'evaluations_of_clusters: {evaluations_of_clusters[0]}')

    # evaluations_of_clusters, count_of_class_for_validator = impute_validation(evaluations_of_clusters, count_of_class_for_validator, num_of_clusters, num_of_classes)

    # logger.info(f'evaluations_of_clusters: {evaluations_of_clusters[0]}')

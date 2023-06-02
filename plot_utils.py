import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import yaml
from sklearn.metrics import confusion_matrix


def get_epoch_reports_json(file_path):
    with open(f'{file_path}/epoch_reports.json', 'r') as f:
        epoch_reports = json.load(f)
    params = get_params_json(file_path)
    epoch_reports['final_epoch'] = params['epochs']
    epoch_reports['attack_methods'] = params['attack_methods']
    return epoch_reports

def get_params_json(file_path):
    try:
        with open(f'{file_path}/params.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        return params
    except:
        print(f'No params.yaml found in {file_path}')
        return {}
    
def get_selected_cluster_idx(epoch_report):
    try:
        argsort_result = epoch_report['argsort_result']
        filtered_clusters = argsort_result[:len(argsort_result)//2]
        unfiltered_clusters = argsort_result[len(argsort_result)//2:]
        return filtered_clusters, unfiltered_clusters
    except:
        return np.nan, np.nan
    
def get_adv_contrib_in_selected_clusters(epoch_report):
    try:
        num_of_adversary = 10
        adv_contribs = []
        _, selected_clusters = get_selected_cluster_idx(epoch_report)
        for cluster_idx in selected_clusters:
            weight_vec = epoch_report['weight_vecs_by_cluster'][str(cluster_idx)]
            adv_contrib = np.sum(weight_vec[:num_of_adversary][:num_of_adversary])
            adv_contribs.append(adv_contrib)
        return adv_contribs
    except:
        return np.nan
    
def get_sneak_contrib(epoch_report):
    try:
        _, selected_clusters = get_selected_cluster_idx(epoch_report)
        adv_contribs = get_adv_contrib_in_selected_clusters(epoch_report)
        selected_wv = [epoch_report['wv'][i] for i in selected_clusters]
        return np.dot(selected_wv, adv_contribs)
    except:
        return np.nan
    


def get_tpr_tnr(epoch_report):
    try:
        names = epoch_report['names']
        adversarial_namelist = epoch_report['validation_container']['adversarial_namelist']
        clusters = epoch_report['clusters_agg']
        lowest_score_for_each_cluster = epoch_report['lowest_score_for_each_cluster']
        argsort_result = epoch_report['argsort_result']
        filtered_clusters = argsort_result[:len(argsort_result)//2]
        unfiltered_clusters = argsort_result[len(argsort_result)//2:]

        mal_gt = []
        mal_pred = []
        for idx, cluster in enumerate(clusters):
            for i in cluster:
                if i in adversarial_namelist:
                    mal_gt.append(1)
                else:
                    mal_gt.append(0)
                mal_pred.append(1 if idx in filtered_clusters else 0)

        # calculate tpr, tnr using sklearn
        tp, fn, fp, tn = confusion_matrix(mal_gt, mal_pred).ravel()
        tpr = 100 * tp / (tp + fn)
        tnr = 100 * tn / (tn + fp)
        return tpr, tnr
    except:
        return np.nan, np.nan

def get_average_tpr_tnr(epoch_reports, epoch_range):
    tprs = []
    tnrs = []
    for epoch in epoch_range:
        tpr, tnr = get_tpr_tnr(epoch_reports[str(epoch)])
        tprs.append(tpr)
        tnrs.append(tnr)
    return np.mean(tprs), np.mean(tnrs)

def get_final_recall(epoch_reports):
    try:
        return epoch_reports[str(epoch_reports['final_epoch'])]['epoch_acc_c']
    except:
        return np.nan

def get_final_asr(epoch_reports):
    try:
        return epoch_reports[str(epoch_reports['final_epoch'])]['epoch_acc_p']
    except:
        return np.nan

def get_final_mainacc(epoch_reports):
    try:
        return epoch_reports[str(epoch_reports['final_epoch'])]['epoch_acc']
    except:
        return np.nan

def get_relevant_metric_perf(epoch_reports):
    try:
        attack_methods = epoch_reports['attack_methods']
        if attack_methods in ['targeted_label_flip']:
            return get_final_recall(epoch_reports)
        elif attack_methods in ['dba']:
            return get_final_asr(epoch_reports)
        elif attack_methods in ['inner_product_manipulation']:
            return get_final_mainacc(epoch_reports)
    except:
        return np.nan
    
def read_cluster_comparison(filename):
    df = pd.read_csv(filename)
    df.index = df['epoch']
    df = df.drop(columns=['epoch', 'epoch.1'])
    # calculate mean of each column

    return df.mean().to_dict()

def read_imputation_df(filename):
    df = pd.read_csv(filename, index_col=0)
    # get the name of the first column
    # first_col = df.columns[0]
    # df.index = df[first_col]
    df.index = df.index * 100
    df.index = df.index.astype(int)
    df = df.drop(columns=['zero', 'SoftImpute'])
    return df

def get_mal_pcnt_exp_results(type='fmnist'):
    mal_pcnts = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    aggregation_methods = ['mean', 'mean--oracle_mode', 'our_aggr', 'our_aggr--injective_florida']
    file_paths = {aggr_method: {mal_pcnt: f'saved_results/mal_pcnt/mal_pcnt_{mal_pcnt}_{type}_{aggr_method}' for mal_pcnt in mal_pcnts} for aggr_method in aggregation_methods}
    epoch_reports = {aggr_method: {mal_pcnt: get_epoch_reports_json(file_path) for mal_pcnt, file_path in file_paths[aggr_method].items()} for aggr_method in aggregation_methods}
    final_recall = {aggr_method: {mal_pcnt: get_final_recall(epoch_reports[aggr_method][mal_pcnt]) for mal_pcnt in mal_pcnts} for aggr_method in aggregation_methods}
    return final_recall

def get_noniid_exp_results(type='fmnist'):
    noniid = ["one_class_expert", "sampling_dirichlet"]
    aggregation_methods = ['mean', 'mean--oracle_mode', 'our_aggr', 'our_aggr--injective_florida']
    file_paths = {noniid: {aggr_method: f'saved_results/noniid/noniid_{noniid}_fmnist_{aggr_method}' for aggr_method in aggregation_methods} for noniid in noniid}
    epoch_reports = {noniid: {aggr_method: get_epoch_reports_json(file_path) for aggr_method, file_path in file_paths[noniid].items()} for noniid in noniid}
    final_results = {noniid: {aggr_method: {'recall': get_final_recall(epoch_reports[noniid][aggr_method]), 'asr': get_final_asr(epoch_reports[noniid][aggr_method]), 'mainacc': get_final_mainacc(epoch_reports[noniid][aggr_method])} for aggr_method in aggregation_methods} for noniid in noniid}
    return final_results

def get_mal_val_type_results(type='cifar'):
    mal_val_type = ['None', 'naive', 'adaptive']
    aggregation_methods = ['our_aggr', 'our_aggr--injective_florida']
    attack_methods= ['targeted_label_flip', 'dba']
    file_paths = {aggr_method: {mal_val: {attack_method: f'saved_results/mal_val_type/mal_val_type_{mal_val}_{attack_method}_{aggr_method}' for attack_method in attack_methods} for mal_val in mal_val_type} for aggr_method in aggregation_methods}
    epoch_reports = {aggr_method: {mal_val: {attack_method: get_epoch_reports_json(file_path) for attack_method, file_path in file_paths[aggr_method][mal_val].items()} for mal_val in mal_val_type} for aggr_method in aggregation_methods}
    final_results = {aggr_method: {mal_val: {attack_method: get_relevant_metric_perf(epoch_reports[aggr_method][mal_val][attack_method]) for attack_method in attack_methods} for mal_val in mal_val_type} for aggr_method in aggregation_methods}
    avg_tpr = {aggr_method: {mal_val: {attack_method: get_average_tpr_tnr(epoch_reports[aggr_method][mal_val][attack_method], range(201, 211))[0] for attack_method in attack_methods} for mal_val in mal_val_type} for aggr_method in aggregation_methods}
    avg_tnr = {aggr_method: {mal_val: {attack_method: get_average_tpr_tnr(epoch_reports[aggr_method][mal_val][attack_method], range(201, 211))[1] for attack_method in attack_methods} for mal_val in mal_val_type} for aggr_method in aggregation_methods}
    return final_results, avg_tpr, avg_tnr

def get_contrib_adj_results(type='cifar'):
    contrib_adj = [0, 0.25, 0.5, 0.75]
    attack_methods= ['targeted_label_flip', 'dba']
    # file_paths = {adj: {attack_method: f'saved_results/contrib_adjustment/contrib_adj_{adj}_cifar_{attack_method}' for attack_method in attack_methods} for adj in contrib_adj}
    # epoch_reports = {adj: {attack_method: get_epoch_reports_json(file_path) for attack_method, file_path in file_paths[adj].items()} for adj in contrib_adj}
    # final_results = {adj: {attack_method: get_relevant_metric_perf(epoch_reports[adj][attack_method]) for attack_method in attack_methods} for adj in contrib_adj}
    file_paths = {attack_method: {adj: f'saved_results/contrib_adjustment/contrib_adj_{adj}_cifar_{attack_method}' for adj in contrib_adj} for attack_method in attack_methods}
    epoch_reports = {attack_method: {adj: get_epoch_reports_json(file_path) for adj, file_path in file_paths[attack_method].items()} for attack_method in attack_methods}
    # final_results = {attack_method: {adj: get_relevant_metric_perf(epoch_reports[attack_method][adj]) for adj in contrib_adj} for attack_method in attack_methods}
    # avg_tpr = {attack_method: {adj: get_average_tpr_tnr(epoch_reports[attack_method][adj], range(201, 211))[0] for adj in contrib_adj} for attack_method in attack_methods}
    # avg_tnr = {attack_method: {adj: get_average_tpr_tnr(epoch_reports[attack_method][adj], range(201, 211))[1] for adj in contrib_adj} for attack_method in attack_methods}
    final_perf_tpr_tnr = {attack_method: {'performance': {adj: get_relevant_metric_perf(epoch_reports[attack_method][adj]) for adj in contrib_adj}, 'tpr': {adj: get_average_tpr_tnr(epoch_reports[attack_method][adj], range(201, 211))[0] for adj in contrib_adj}, 'tnr': {adj: get_average_tpr_tnr(epoch_reports[attack_method][adj], range(201, 211))[1] for adj in contrib_adj}} for attack_method in attack_methods}
    # avg_tpr = {adj: {attack_method: get_average_tpr_tnr(epoch_reports[adj][attack_method], range(201, 211))[0] for attack_method in attack_methods} for adj in contrib_adj}
    # avg_tnr = {adj: {attack_method: get_average_tpr_tnr(epoch_reports[adj][attack_method], range(201, 211))[1] for attack_method in attack_methods} for adj in contrib_adj}
    # return final_results, avg_tpr, avg_tnr
    return final_perf_tpr_tnr

def get_adv_contrib_vs_score():
    expnames = [
        # 'noniid/noniid_sampling_dirichlet_cifar_our_aggr--injective_florida_dba',
        # 'noniid/noniid_one_class_expert_cifar_our_aggr--injective_florida_dba',
        # 'contrib_adjustment/contrib_adj_0.75_cifar_dba',
        'noniid/noniid_sampling_dirichlet_cifar_our_aggr--injective_florida_targeted_label_flip',
        'noniid/noniid_one_class_expert_cifar_our_aggr--injective_florida_targeted_label_flip',
        'contrib_adjustment/contrib_adj_0.75_cifar_targeted_label_flip'
    ]
    all_epoch_reports = [get_epoch_reports_json('saved_results/' + expname) for expname in expnames]

    num_of_adversary = 10
    mal_ensemble_contrib_tuples = []
    benign_ensemble_contrib_tuples = []
    adv_contrib_vs_score = []
    for epoch_reports in all_epoch_reports:
        for e in range(201, 202):
            epoch_report = epoch_reports[str(e)]
            for i in epoch_report['weight_vecs_by_cluster']:
                i = int(i)
                adv_contrib = sum(epoch_report['weight_vecs_by_cluster'][str(i)][:num_of_adversary])
                benign_contrib = sum(epoch_report['weight_vecs_by_cluster'][str(i)][num_of_adversary:])
                score = epoch_report['lowest_score_for_each_cluster'][i]
                adv_contrib_vs_score.append([adv_contrib, score, i< num_of_adversary])
                if i < num_of_adversary:
                    mal_ensemble_contrib_tuples.append((adv_contrib, benign_contrib))
                else:
                    benign_ensemble_contrib_tuples.append((adv_contrib, benign_contrib))

    adv_contrib_vs_score = np.array(adv_contrib_vs_score).T

    adv_contrib_vs_score_df = pd.DataFrame(adv_contrib_vs_score.T, columns=['adv_contrib', 'score', 'is_adv'])

    return adv_contrib_vs_score_df

def get_ablation_aggregate_ensemble_results():
    filepaths = {
        True: 'saved_results/ablation_aggregate_ensemble/ablation_aggregate_ensemble',
        False: 'saved_results/ablation_aggregate_ensemble/ablation_aggregate_ensemble_false/ablation_aggregation_ensemble_false'
    }
    epoch_reports = {k: get_epoch_reports_json(v) for k, v in filepaths.items()}
    num_of_adversary = 10
    result_dict = {}
    result_dict['mean_sneak_contribs'] = {
        True: np.mean([get_sneak_contrib(epoch_reports[True][str(i)]) for i in range(201, 211)]),
        False: np.mean([sum(epoch_reports[False][str(i)]['wv'][:num_of_adversary]) for i in range(201, 211)])
    }
    result_dict['avg_tpr'] = {
        True: get_average_tpr_tnr(epoch_reports[True], range(201, 211))[0],
        False: get_average_tpr_tnr(epoch_reports[False], range(201, 211))[0]
    }
    result_dict['avg_tnr'] = {
        True: get_average_tpr_tnr(epoch_reports[True], range(201, 211))[1],
        False: get_average_tpr_tnr(epoch_reports[False], range(201, 211))[1]
    }
    result_dict['final_result'] = {
        True: get_relevant_metric_perf(epoch_reports[True]),
        False: get_relevant_metric_perf(epoch_reports[False])
    }
    result_dict_df = pd.DataFrame(result_dict)
    return result_dict_df

def get_cluster_comparison_results():
    filepaths = {
        k: f'saved_results/ablation_clustering_comparison/clustering_comp_{k}_fmnist_our_aggr_targeted_label_flip/' for k in ['one_class_expert', 'sampling_dirichlet', 'False']
    }
    cluster_results = {k: read_cluster_comparison(f'{v}/cluster_comparison.csv') for k, v in filepaths.items()}
    cluster_results_df = pd.DataFrame(cluster_results).T
    return cluster_results_df

def get_imputation_comparison_results():
    filepaths = {
        k: f'saved_results/ablation_imputation_comparison/clustering_comp_{k}_fmnist_our_aggr_targeted_label_flip' for k in ['one_class_expert', 'sampling_dirichlet', 'False']
    }
    imputation_dfs = {k: read_imputation_df(f'{v}/imputation.csv') for k, v in filepaths.items()}
    return imputation_dfs
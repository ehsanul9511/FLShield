import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
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
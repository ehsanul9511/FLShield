import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import yaml


def get_epoch_reports_json(file_path):
    with open(f'{file_path}/epoch_reports.json', 'r') as f:
        epoch_reports = json.load(f)
    epoch_reports['final_epoch'] = get_params_json(file_path)['epochs']
    return epoch_reports

def get_params_json(file_path):
    try:
        with open(f'{file_path}/params.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        return params
    except:
        print(f'No params.yaml found in {file_path}')
        return {}

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
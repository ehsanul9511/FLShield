import json
import csv
import copy
import numpy as np
train_fileHeader = ["local_model", "round", "epoch", "internal_epoch", "average_loss", "accuracy", "correct_data",
                    "total_data"]
test_fileHeader = ["model", "epoch", "average_loss", "accuracy", "correct_data", "total_data"]
train_result = []  # train_fileHeader
test_result = []  # test_fileHeader
posiontest_result = []  # test_fileHeader
recall_result = []

triggertest_fileHeader = ["model", "trigger_name", "trigger_value", "epoch", "average_loss", "accuracy", "correct_data",
                          "total_data"]
poisontriggertest_result = []  # triggertest_fileHeader

posion_test_result = []  # train_fileHeader
posion_posiontest_result = []  # train_fileHeader
weight_result=[]
scale_result=[]
scale_temp_one_row=[]

dynamic_dist_fileheader = []
dynamic_dist_result = []
validator_pcnt_fileheader = ["adv", "benign", "tp", "fp", "tn", "fn", "adjusted_adv", "adjusted_benign"]
validator_pcnt_result = []

epoch_reports = {}


def convert_float32_to_float(obj):
    if isinstance(obj, dict):
        # Recursively convert float32 values in nested dictionaries
        return {k: convert_float32_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively convert float32 values in nested lists
        return [convert_float32_to_float(item) for item in obj]
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64) or isinstance(obj, np.float16) or isinstance(obj, np.int8) or isinstance(obj, np.int16) or isinstance(obj, np.int32) or isinstance(obj, np.int64):
        # Convert float32 to float
        return float(obj)
    elif isinstance(obj, np.ndarray):
        # Convert numpy array to list
        return obj.tolist()
    else:
        return obj

def save_epoch_report(folder_path):
    global epoch_reports
    with open(f'{folder_path}/epoch_reports.json', 'w') as f:
        json.dump(epoch_reports, f)

def save_result_csv(epoch, is_posion,folder_path):
    train_csvFile = open(f'{folder_path}/train_result.csv', "w")
    train_writer = csv.writer(train_csvFile)
    train_writer.writerow(train_fileHeader)
    train_writer.writerows(train_result)
    train_csvFile.close()

    test_csvFile = open(f'{folder_path}/test_result.csv', "w")
    test_writer = csv.writer(test_csvFile)
    test_writer.writerow(test_fileHeader)
    test_writer.writerows(test_result)
    test_csvFile.close()

    if len(weight_result)>0:
        weight_csvFile=  open(f'{folder_path}/weight_result.csv', "w")
        weight_writer = csv.writer(weight_csvFile)
        weight_writer.writerows(weight_result)
        weight_csvFile.close()

    if len(scale_temp_one_row)>0:
        _csvFile=  open(f'{folder_path}/scale_result.csv', "w")
        _writer = csv.writer(_csvFile)
        scale_result.append(copy.deepcopy(scale_temp_one_row))
        scale_temp_one_row.clear()
        _writer.writerows(scale_result)
        _csvFile.close()

    if is_posion or True:
        test_csvFile = open(f'{folder_path}/posiontest_result.csv', "w")
        test_writer = csv.writer(test_csvFile)
        test_writer.writerow(test_fileHeader)
        test_writer.writerows(posiontest_result)
        test_csvFile.close()

        test_csvFile = open(f'{folder_path}/recall_result.csv', "w")
        test_writer = csv.writer(test_csvFile)
        test_writer.writerow(test_fileHeader)
        test_writer.writerows(recall_result)
        test_csvFile.close()

        test_csvFile = open(f'{folder_path}/poisontriggertest_result.csv', "w")
        test_writer = csv.writer(test_csvFile)
        test_writer.writerow(triggertest_fileHeader)
        test_writer.writerows(poisontriggertest_result)
        test_csvFile.close()

        if len(dynamic_dist_fileheader) != 0:
            test_csvFile = open(f'{folder_path}/dynamic_dist_result.csv', "w")
            test_writer = csv.writer(test_csvFile)
            test_writer.writerow(dynamic_dist_fileheader)
            test_writer.writerows(dynamic_dist_result)
            test_csvFile.close()

        test_csvFile = open(f'{folder_path}/validator_pcnt.csv', "w")
        test_writer = csv.writer(test_csvFile)
        test_writer.writerow(validator_pcnt_fileheader)
        test_writer.writerows(validator_pcnt_result)
        test_csvFile.close()

def add_weight_result(name,weight,alpha):
    weight_result.append(name)
    weight_result.append(weight)
    weight_result.append(alpha)


# target_model
# updates
# epoch
# weight_accumulator
# start_epoch
# start_time
# t
# client_grads
# alphas
# delta_models
# name
# data
# grads
# norms
# full_grads
# output_layer_only
# num_of_classes
# no_clustering
# no_ensemble
# clustering_method
# _
# clusters_agg
# idx
# cluster
# clstr
# count_of_class_for_validator
# iidx
# count_of_class
# num_of_clusters
# adj_delta_models
# agg_model
# weight_vec
# weight_vecs_by_cluster
# i
# aggregate_weights
# update_per_layer
# cluster_maliciousness
# validation_container
# f
# valProcessor
# wv_by_cluster
# clipping_weights
# green_clusters
# mal_pcnts
# cl_id
# wv_print_str
# w
# all_validator_evaluations
# evaluations_of_clusters
# names
# norm_median
# self
# val_idx
# val_score_by_class
# val_score_by_class_per_example
# wv

helper_local_var_names_for_log = [
    # "target_model",
    # "updates",
    "epoch",
    # "weight_accumulator",
    "start_epoch",
    "start_time",
    "t",
    # "client_grads",
    "alphas",
    # "delta_models",
    # "name",
    # "data",
    # "grads",
    "norms",
    # "full_grads",
    "output_layer_only",
    "num_of_classes",
    "no_clustering",
    "no_ensemble",
    "clustering_method",
    # "_",
    "clusters_agg",
    # "idx",
    # "cluster",
    # "clstr",
    "count_of_class_for_validator",
    # "iidx",
    # "count_of_class",
    "num_of_clusters",
    # "adj_delta_models",
    # "agg_model",
    # "weight_vec",
    "weight_vecs_by_cluster",
    # "i",
    # "aggregate_weights",
    # "update_per_layer",
    "cluster_maliciousness",
    "validation_container",
    "before_imputation_validation_container",
    "before_processing_validation_container",
    # "f",
    # "valProcessor",
    "wv_by_cluster",
    "clipping_weights",
    # "green_clusters",
    "mal_pcnts",
    "cl_id",
    # "wv_print_str",
    "w",
    # "all_validator_evaluations",
    # "evaluations_of_clusters",
    "names",
    "norm_median",
    # "self",
    # "val_idx",
    # "val_score_by_class",
    # "val_score_by_class_per_example",
    "wv"
]
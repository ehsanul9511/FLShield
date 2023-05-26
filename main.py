import argparse
import json
import datetime
import os
import logging
import torch
import torch.nn as nn
import train
import test
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import csv
from torchvision import transforms
from collections import defaultdict
from loan_helper import LoanHelper
from image_helper import ImageHelper
from utils.utils import dict_html
import utils.csv_record as csv_record
import yaml
import time
import visdom
import numpy as np
import random
import config
import copy
import sys
import os
import pickle
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger("logger")
logger.setLevel("ERROR")

criterion = torch.nn.CrossEntropyLoss()
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)

from jinja2 import Template
import yaml

def get_context(type='cifar',
                aggregation_methods='our_aggr',
                injective_florida=False,
                attack_methods='targeted_label_flip',
                noniid=False,
                resumed_model=True,
                mal_pcnt=0.4,
                *args,
                **kwargs):
    context = {
        'type': type,
        'aggregation_methods': aggregation_methods,
        'injective_florida': injective_florida,
        'attack_methods': attack_methods,
        'noniid': noniid,
        'resumed_model': resumed_model,
        'mal_pcnt': mal_pcnt
    }
    ## add other kwargs
    for key, value in kwargs.items():
        context[key] = value
    return context

def get_param_for_context(context=None):
    # Define the dynamic values
    if context is None:
        context = get_context()

    # Load the template file
    with open('utils/jinja.yaml') as file:
        template = Template(file.read())

    # Render the template with the dynamic values
    rendered_config = template.render(context)

    # The rendered_config now contains the final YAML configuration
    # convert it into a dictionary
    params = yaml.load(rendered_config, Loader=yaml.FullLoader)

    return params


def run(params_loaded):
    params_loaded = defaultdict(lambda: None, params_loaded)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    if params_loaded['type'] == config.TYPE_LOAN:
        helper = LoanHelper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'loan'))
        helper.load_data(params_loaded)
    elif params_loaded['type'] == config.TYPE_CIFAR:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'cifar'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_MNIST:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'mnist'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_FMNIST:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'fmnist'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_EMNIST:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'emnist'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_TINYIMAGENET:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'tiny'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_CELEBA:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'celebA'))
        helper.load_data()

    else:
        helper = None
    
    logger.info(f'load data done')
    helper.create_model()
    logger.info(f'create model done')
    ### Create models
    if helper.params['is_poison']:
        logger.info(f"Poisoned following participants: {(helper.adversarial_namelist)}")

    weight_accumulator = helper.init_weight_accumulator(helper.target_model)

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(dict(helper.params), f)

    csv_record.epoch_reports = {e: {} for e in range(int(helper.params['epochs'])+1)}

    # dictionary object to store test results and pickle it
    helper.result_dict = defaultdict(lambda: [])

    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1, helper.params['aggr_epoch_interval']):
        start_time = time.time()
        t = time.time()

        agent_name_keys = helper.participants_list
        adversarial_name_keys = []
        random.seed(42+epoch)
        # if helper.params['is_random_adversary'] or helper.params['random_adversary_for_label_flip']:  # random choose , maybe don't have advasarial
        #     agent_name_keys = random.sample(helper.participants_list, helper.params['no_models'])
        #     for _name_keys in agent_name_keys:
        #         if _name_keys in helper.adversarial_namelist:
        #             adversarial_name_keys.append(_name_keys)
        # else:  # must have advasarial if this epoch is in their poison epoch
        ongoing_epochs = list(range(epoch, epoch + helper.params['aggr_epoch_interval']))
        adv_num = int(len(helper.adversarial_namelist) * helper.params['no_models'] / len(helper.participants_list))
        # for idx in range(0, len(helper.adversarial_namelist)):
        # for iidx in range(0, adv_num):
        #     idx = random.sample(range(0, len(helper.adversarial_namelist)), 1)[0]
        #     for ongoing_epoch in ongoing_epochs:
        #         if ongoing_epoch in helper.poison_epochs_by_adversary[idx]:
        #             if helper.adversarial_namelist[idx] not in adversarial_name_keys:
        #                 adversarial_name_keys.append(helper.adversarial_namelist[idx])
        adversarial_name_keys = random.sample(helper.adversarial_namelist, adv_num)

        # nonattacker=[]
        # for adv in helper.adversarial_namelist:
        #     if adv not in adversarial_name_keys:
        #         nonattacker.append(copy.deepcopy(adv))
        if helper.params['aggregation_methods'] == config.AGGR_FLTRUST:
            benign_num = helper.params['no_models'] - len(adversarial_name_keys) - 1
            random_agent_name_keys = random.sample(helper.benign_namelist[:-1], benign_num)
            agent_name_keys = adversarial_name_keys + random_agent_name_keys + [helper.benign_namelist[-1]]
        else:
            benign_num = helper.params['no_models'] - len(adversarial_name_keys)
            random_agent_name_keys = random.sample(helper.benign_namelist, benign_num)
            agent_name_keys = adversarial_name_keys + random_agent_name_keys
        # else:
        #     if helper.params['is_random_adversary']==False:
        #         adversarial_name_keys=copy.deepcopy(helper.adversarial_namelist)
        logger.info(f'Server Epoch:{epoch} choose agents : {agent_name_keys}.')
        epochs_submit_update_dict, num_samples_dict = train.train(helper=helper, start_epoch=epoch,
                                                                  local_model=helper.local_model,
                                                                  target_model=helper.target_model,
                                                                  is_poison=helper.params['is_poison'],
                                                                  agent_name_keys=agent_name_keys)
        # for agnt in helper.local_models.keys():
        #     print(helper.local_models[agnt].state_dict())
        logger.info(f'time spent on training: {time.time() - t}')
        t = time.time()
        logger.info(f'training with {len(agent_name_keys)} agents done')
        weight_accumulator, updates = helper.accumulate_weight(weight_accumulator, epochs_submit_update_dict,
                                                               agent_name_keys, num_samples_dict)
        logger.info(f'received {len(updates)} updates')

        if helper.params['attack_methods'] == config.ATTACK_IPM:
            updates = helper.ipm_attack(updates)

        is_updated = True
        if helper.params['aggregation_methods'] == config.AGGR_OURS:
            # helper.combined_clustering_guided_aggregation(helper.target_model, updates, epoch)
            # helper.combined_clustering_guided_aggregation_with_DP(helper.target_model, updates, epoch)
            helper.combined_clustering_guided_aggregation_v2(helper.target_model, updates, epoch, weight_accumulator)
        elif helper.params['aggregation_methods'] == config.AGGR_AFA:
            is_updated, names, weights = helper.afa_method(helper.target_model, updates)
        elif helper.params['aggregation_methods'] == config.AGGR_FLTRUST:
            is_updated, names, weights = helper.fltrust(helper.target_model, updates, epoch)
        elif helper.params['aggregation_methods'] == config.AGGR_MEAN:
            # Average the models
            # is_updated = helper.average_shrink_models(weight_accumulator=weight_accumulator,
            #                                           target_model=helper.target_model,
            #                                           epoch_interval=helper.params['aggr_epoch_interval'])
            # num_oracle_calls = 1
            is_updated = helper.fedavg(helper.target_model, updates, epoch)
        elif helper.params['aggregation_methods'] == config.AGGR_GEO_MED:
            
            maxiter = helper.params['geom_median_maxiter']
            num_oracle_calls, is_updated, names, weights, alphas = helper.geometric_median_update(helper.target_model, updates, maxiter=maxiter)

        elif helper.params['aggregation_methods'] == config.AGGR_FLAME:
            helper.flame(helper.target_model, updates, epoch)
            num_oracle_calls = 1

        logger.info(f'time spent on aggregation: {time.time() - t}')
        t=time.time()

        # clear the weight_accumulator
        weight_accumulator = helper.init_weight_accumulator(helper.target_model)

        temp_global_epoch = epoch + helper.params['aggr_epoch_interval'] - 1

        epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=temp_global_epoch,
                                                                       model=helper.target_model, is_poison=False,
                                                                       visualize=False, agent_name_key="global")
        csv_record.test_result.append(["global", temp_global_epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])
        csv_record.epoch_reports[epoch]["epoch_loss"], csv_record.epoch_reports[epoch]["epoch_acc"] = epoch_loss, epoch_acc
        helper.result_dict['mainacc'].append(epoch_acc)
        if len(csv_record.scale_temp_one_row)>0:
            csv_record.scale_temp_one_row.append(round(epoch_acc, 4))

        if (helper.params['is_poison'] or True) and helper.params['attack_methods'] in [config.ATTACK_DBA, config.ATTACK_TLF, config.ATTACK_AOTT, config.ATTACK_SEMANTIC]:
            if helper.params['attack_methods'] == config.ATTACK_DBA:
                epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                        epoch=temp_global_epoch,
                                                                                        model=helper.target_model,
                                                                                        is_poison=True,
                                                                                        visualize=False,
                                                                                        agent_name_key="global")
            elif helper.params['attack_methods'] in [config.ATTACK_TLF]:
                epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test.Mytest_poison_label_flip(helper=helper,
                                                                                        epoch=temp_global_epoch,
                                                                                        model=helper.target_model,
                                                                                        is_poison=True,
                                                                                        visualize=False,
                                                                                        agent_name_key="global",
                                                                                        get_recall=True)
                csv_record.recall_result.append(["global", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])
                csv_record.epoch_reports[epoch]["epoch_loss_c"], csv_record.epoch_reports[epoch]["epoch_acc_c"] = epoch_loss, epoch_acc_p
                helper.result_dict['recall'].append(epoch_acc_p)
                epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test.Mytest_poison_label_flip(helper=helper,
                                                                                        epoch=temp_global_epoch,
                                                                                        model=helper.target_model,
                                                                                        is_poison=True,
                                                                                        visualize=False,
                                                                                        agent_name_key="global")
            elif helper.params['attack_methods'] in [config.ATTACK_AOTT]:
                epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test.Mytest_edge_test(helper=helper,model=helper.target_model)
                # epoch_acc_p = np.nan        
            elif helper.params['attack_methods'] in [config.ATTACK_SEMANTIC]:
                epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test.Mytest_semantic_test(helper=helper,model=helper.target_model)
                # epoch_acc_p = np.nan        
            csv_record.epoch_reports[epoch]["epoch_loss_p"], csv_record.epoch_reports[epoch]["epoch_acc_p"] = epoch_loss, epoch_acc_p

            csv_record.posiontest_result.append(
                ["global", temp_global_epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])
            helper.result_dict['poison_test_acc'].append(epoch_acc_p)

        logger.info(f'time spent on testing: {time.time() - t}')
        t = time.time()

        helper.save_model(epoch=epoch, val_loss=epoch_loss)
        logger.info(f'Done in {time.time() - start_time} sec.')
        # logger.info(f"This run has a label: {helper.params['current_time']}. ")
        logger.info(f"Epoch {epoch} finished. ")
        logger.info(f"Saving results to {helper.folder_path}.")
        csv_record.save_result_csv(epoch, helper.params['is_poison'], helper.folder_path)
        csv_record.save_epoch_report(helper.folder_path)


    # save the result dictionary to a pickle file
    # pickle.dump(helper.result_dict, open(os.path.join(helper.folder_path, 'result_dict.pkl'), 'wb'))
    pickle.dump(dict(helper.result_dict), open(os.path.join(helper.folder_path, 'result_dict.pkl'), 'wb'))


    logger.info(f"This run has a label: {helper.params['current_time']}. ")





if __name__ == '__main__':
    print('Start training')
    np.random.seed(1)
    time_start_load_everything = time.time()
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='utils/fmnist_params.yaml')
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    print(f'args: {args}')
    print(f'unknown: {unknown}')
    if len(unknown) == 0:
        with open(f'./{args.params}', 'r') as f:
            # params_loaded = yaml.load(f)
            params_loaded = yaml.safe_load(f)
    else:
        # parse unknown arguments
        for arg in unknown:
            if arg.startswith(("-", "--")):
                if "=" in arg:
                    k, v = arg.split("=")
                    setattr(args, k.strip("-"), v)
                else:
                    setattr(args, arg.strip("-"), True)
        print(f'args: {args}')

        # convert args to dictionary
        args = vars(args)
        del args['params']
        print(f'args: {args}')

        context = get_context(**args)
        print(f'context: {context}')

        params_loaded = get_param_for_context(context)

        for k, v in context.items():
            if k not in params_loaded:
                params_loaded[k] = v


    run(params_loaded)
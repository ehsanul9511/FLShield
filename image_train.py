from asyncio.log import logger
from sympy import evaluate
import utils.csv_record as csv_record
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import main
import test
import copy
import config

import numpy as np
from tqdm import tqdm

from random import shuffle

def ImageTrain(helper, start_epoch, local_model, target_model, is_poison,agent_name_keys):

    if type(target_model) is list:
        base_model = copy.deepcopy(target_model)
        psuedo_train_mode = True
    else:
        psuedo_train_mode = False

    def main_logger_info(info):
        if not helper.params['minimize_logging']:
            main.logger.info(info)

    epochs_submit_update_dict = dict()
    num_samples_dict = dict()
    current_number_of_adversaries=0
    for temp_name in agent_name_keys:
        if temp_name in helper.adversarial_namelist:
            current_number_of_adversaries+=1

    helper.local_models = {}
    for model_id in tqdm(range(helper.params['no_models']), disable=False):
        if psuedo_train_mode:
            target_model = base_model[helper.validation_assignments[model_id]]

        epochs_local_update_list = []
        epochs_local_update_list_for_accumulator = []
        last_local_model = dict()
        client_grad = [] # only works for aggr_epoch_interval=1

        for name, data in target_model.state_dict().items():
            last_local_model[name] = target_model.state_dict()[name].clone()

        agent_name_key = agent_name_keys[model_id]
        ## Synchronize LR and models
        model = local_model
        model.copy_params(target_model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        model.train()
        adversarial_index= -1
        localmodel_poison_epochs = helper.params['poison_epochs']
        if is_poison and agent_name_key in helper.adversarial_namelist:
            for temp_index in range(0, len(helper.adversarial_namelist)):
                if int(agent_name_key) == helper.adversarial_namelist[temp_index]:
                    adversarial_index= temp_index
                    localmodel_poison_epochs = helper.poison_epochs_by_adversary[adversarial_index]
                    # main.logger.info(f'poison local model {agent_name_key} index {adversarial_index} ')
                    break
            if len(helper.adversarial_namelist) == 1:
                adversarial_index = -1  # the global pattern

            adversarial_index = helper.adversarial_namelist.index(int(agent_name_key))

        for epoch in range(start_epoch, start_epoch + helper.params['aggr_epoch_interval']):

            target_params_variables = dict()
            for name, param in target_model.named_parameters():
                target_params_variables[name] = last_local_model[name].clone().detach().requires_grad_(False)

            if is_poison and agent_name_key in helper.adversarial_namelist and (epoch in localmodel_poison_epochs):
                main_logger_info('poison_now')

                poison_lr = helper.params['poison_lr']
                internal_epoch_num = helper.params['internal_poison_epochs']
                step_lr = helper.params['poison_step_lr']


                poison_optimizer = torch.optim.SGD(model.parameters(), lr=poison_lr,
                                                   momentum=helper.params['momentum'],
                                                   weight_decay=helper.params['decay'])
                scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                                 milestones=[0.2 * internal_epoch_num,
                                                                             0.8 * internal_epoch_num], gamma=0.1)
                temp_local_epoch = (epoch - 1) *internal_epoch_num
                for internal_epoch in range(1, internal_epoch_num + 1):
                    temp_local_epoch += 1
                    main_logger_info(f'fetching poison data for agent: {agent_name_key} epoch: {temp_local_epoch}')
                    _, data_iterator = helper.train_data[agent_name_key]
                    if helper.params['attack_methods'] == config.ATTACK_AOTT:
                        own_dataset = data_iterator.dataset
                        edge_dataset = helper.poison_trainloader.dataset
                        # samp_indices = np.random.choice(len(edge_dataset), int(0.2*len(own_dataset)), replace=False)
                        # edge_dataset = torch.utils.data.Subset(edge_dataset, samp_indices)
                        data_iterator = torch.utils.data.DataLoader(
                            torch.utils.data.ConcatDataset([own_dataset, edge_dataset]),
                            batch_size=helper.params['batch_size'], shuffle=True)
                    elif helper.params['attack_methods'] == config.ATTACK_SEMANTIC:
                        own_dataset = data_iterator.dataset
                        green_car_as_bird_dataset = helper.semantic_dataloader.dataset
                        data_iterator = torch.utils.data.DataLoader(
                            torch.utils.data.ConcatDataset([own_dataset, green_car_as_bird_dataset]),
                            batch_size=helper.params['batch_size'], shuffle=True)
                    poison_data_count = 0
                    total_loss = 0.
                    correct = 0
                    dataset_size = 0
                    dis2global_list=[]
                    for batch_id, batch in tqdm(enumerate(data_iterator), disable=True):
                        if helper.params['attack_methods'] == config.ATTACK_DBA:
                            #will set adversarial_index to -1 for centralized attack
                            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adversarial_index,evaluation=False)
                        elif helper.params['attack_methods'] == config.ATTACK_TLF:
                            data, targets, poison_num = helper.get_poison_batch_for_targeted_label_flip(batch)
                        elif helper.params['attack_methods'] in [config.ATTACK_AOTT, config.ATTACK_IPM, config.ATTACK_SEMANTIC]:
                            data, targets = helper.get_batch(None, batch)
                            poison_num = 0
                        poison_optimizer.zero_grad()
                        dataset_size += len(data)
                        poison_data_count += poison_num

                        output = model(data)
                        class_loss = nn.functional.cross_entropy(output, targets)
                        # distance_loss = helper.model_dist_norm_var(model, target_params_variables)
                        distance_loss = 0
                        # Lmodel = αLclass + (1 − α)Lano; alpha_loss =1 fixed
                        loss = helper.params['alpha_loss'] * class_loss + \
                            (1 - helper.params['alpha_loss']) * distance_loss
                        # main.logger.info(f'distance_loss: {distance_loss}, class_loss: {class_loss}, loss: {loss}')
                        loss.backward()

                        # get gradients
                        # if helper.params['aggregation_methods']==config.AGGR_FLAME:
                        if helper.params['aggregation_methods'] in [config.AGGR_FLAME, config.AGGR_FLTRUST, config.AGGR_OURS, config.AGGR_AFA, config.AGGR_MEAN]:
                            for i, (name, params) in enumerate(model.named_parameters()):
                                if params.requires_grad:
                                    if internal_epoch == 1 and batch_id == 0:
                                        client_grad.append(params.grad.clone())
                                    else:
                                        client_grad[i] += params.grad.clone()

                        poison_optimizer.step()
                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                        if helper.params["batch_track_distance"]:
                            # we can calculate distance to this model now.
                            temp_data_len = len(data_iterator)
                            distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
                            dis2global_list.append(distance_to_global_model)
                            model.track_distance_batch_vis(vis=main.vis, epoch=temp_local_epoch,
                                                           data_len=temp_data_len,
                                                            batch=batch_id,distance_to_global_model= distance_to_global_model,
                                                           eid=helper.params['environment_name'],
                                                           name=str(agent_name_key),is_poisoned=True)

                    if step_lr:
                        scheduler.step()
                        main_logger_info(f'Current lr: {scheduler.get_last_lr()}')

                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size
                    main_logger_info(
                        '___PoisonTrain {} ,  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%), train_poison_data_count: {}'.format(model.name, epoch, agent_name_key,
                                                                                      internal_epoch,
                                                                                      total_l, correct, dataset_size,
                                                                                     acc, poison_data_count))
                    csv_record.train_result.append(
                        [agent_name_key, temp_local_epoch,
                         epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])
                    if helper.params['vis_train']:
                        model.train_vis(main.vis, temp_local_epoch,
                                        acc, loss=total_l, eid=helper.params['environment_name'], is_poisoned=True,
                                        name=str(agent_name_key) )
                    num_samples_dict[agent_name_key] = dataset_size
                    if helper.params["batch_track_distance"]:
                        main_logger_info(
                            f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                            f'Distance to the global model: {dis2global_list}. ')

                # internal epoch finish
                main_logger_info(f'Global model norm: {helper.model_global_norm(target_model)}.')
                main_logger_info(f'Norm before scaling: {helper.model_global_norm(model)}. '
                                 f'Distance: {helper.model_dist_norm(model, target_params_variables)}')

                if not helper.params['baseline']:
                    main_logger_info(f'will scale.')
                    if not helper.params['speed_boost']:
                        epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=epoch,
                                                                                    model=model, is_poison=False,
                                                                                    visualize=False,
                                                                                    agent_name_key=agent_name_key)
                        csv_record.test_result.append(
                            [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                    if not helper.params['speed_boost'] or ('new_adaptive_attack' in helper.params.keys() and helper.params['new_adaptive_attack']):
                        if helper.params['attack_methods'] == config.ATTACK_DBA:
                            epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                                epoch=epoch,
                                                                                                model=model,
                                                                                                is_poison=True,
                                                                                                visualize=False,
                                                                                                agent_name_key=agent_name_key)
                        elif helper.params['attack_methods'] in [config.ATTACK_TLF]:
                            epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_poison_label_flip(helper=helper,
                                                                                                epoch=epoch,
                                                                                                model=model,
                                                                                                is_poison=True,
                                                                                                visualize=False,
                                                                                                agent_name_key=agent_name_key)                        
                        csv_record.posiontest_result.append(
                            [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                    clip_rate = helper.params['scale_weights_poison']
                    main_logger_info(f"Scaling by  {clip_rate}")
                    for key, value in model.state_dict().items():
                        target_value  = last_local_model[key]
                        clip_down = True
                        if clip_down:
                            new_value = target_value + (value - target_value) * clip_rate
                        else:
                            new_value = value + (target_value - value) * clip_rate
                        model.state_dict()[key].copy_(new_value)
                    distance = helper.model_dist_norm(model, target_params_variables)
                    main_logger_info(
                        f'Scaled Norm after poisoning: '
                        f'{helper.model_global_norm(model)}, distance: {distance}')
                    csv_record.scale_temp_one_row.append(epoch)
                    csv_record.scale_temp_one_row.append(round(distance, 4))
                    if helper.params["batch_track_distance"]:
                        temp_data_len = len(helper.train_data[agent_name_key][1])
                        model.track_distance_batch_vis(vis=main.vis, epoch=temp_local_epoch,
                                                       data_len=temp_data_len,
                                                       batch=temp_data_len-1,
                                                       distance_to_global_model=distance,
                                                       eid=helper.params['environment_name'],
                                                       name=str(agent_name_key), is_poisoned=True)

                distance = helper.model_dist_norm(model, target_params_variables)
                main_logger_info(f"Total norm for {current_number_of_adversaries} "
                                 f"adversaries is: {helper.model_global_norm(model)}. distance: {distance}")

            else:
                temp_local_epoch = (epoch - 1) * helper.params['internal_epochs']
                for internal_epoch in range(1, helper.params['internal_epochs'] + 1):
                    temp_local_epoch += 1

                    _, data_iterator = helper.train_data[agent_name_key]
                    if helper.params['attack_methods'] == config.ATTACK_SEMANTIC:
                        own_dataset = data_iterator.dataset
                        green_car_dataset = helper.semantic_dataloader_correct.dataset
                        data_iterator = torch.utils.data.DataLoader(
                            torch.utils.data.ConcatDataset([own_dataset, green_car_dataset]),
                            batch_size=helper.params['batch_size'], shuffle=True)

                    total_loss = 0.
                    correct = 0
                    dataset_size = 0
                    dis2global_list = []
                    for batch_id, batch in tqdm(enumerate(data_iterator), disable=True):

                        optimizer.zero_grad()
                        data, targets = helper.get_batch(data_iterator, batch,evaluation=False)

                        dataset_size += len(data)
                        output = model(data)
                        loss = nn.functional.cross_entropy(output, targets)
                        loss.backward()

                        # get gradients
                        # if helper.params['aggregation_methods'] == config.AGGR_FLAME:
                        if helper.params['aggregation_methods'] in [config.AGGR_FLAME, config.AGGR_FLTRUST, config.AGGR_OURS, config.AGGR_AFA, config.AGGR_MEAN]:
                            for i, (name, params) in enumerate(model.named_parameters()):
                                if params.requires_grad:
                                    if internal_epoch == 1 and batch_id == 0:
                                        client_grad.append(params.grad.clone())
                                    else:
                                        client_grad[i] += params.grad.clone()

                        optimizer.step()
                        total_loss += loss.data
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

                        if helper.params["vis_train_batch_loss"] and False:
                            cur_loss = loss.data
                            temp_data_len = len(data_iterator)
                            model.train_batch_vis(vis=main.vis,
                                                  epoch=temp_local_epoch,
                                                  data_len=temp_data_len,
                                                  batch=batch_id,
                                                  loss=cur_loss,
                                                  eid=helper.params['environment_name'],
                                                  name=str(agent_name_key) , win='train_batch_loss', is_poisoned=False)
                        if helper.params["batch_track_distance"] and False:
                            # we can calculate distance to this model now
                            temp_data_len = len(data_iterator)
                            distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
                            dis2global_list.append(distance_to_global_model)
                            model.track_distance_batch_vis(vis=main.vis, epoch=temp_local_epoch,
                                                           data_len=temp_data_len,
                                                            batch=batch_id,distance_to_global_model= distance_to_global_model,
                                                           eid=helper.params['environment_name'],
                                                           name=str(agent_name_key),is_poisoned=False)

                    acc = 100.0 * (float(correct) / float(dataset_size))
                    total_l = total_loss / dataset_size
                    main_logger_info(
                        '___Train {},  epoch {:3d}, local model {}, internal_epoch {:3d},  Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%)'.format(model.name, epoch, agent_name_key, internal_epoch,
                                                           total_l, correct, dataset_size,
                                                           acc))
                    csv_record.train_result.append([agent_name_key, temp_local_epoch,
                                                    epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])

                    if helper.params['vis_train'] and False:
                        model.train_vis(main.vis, temp_local_epoch,
                                        acc, loss=total_l, eid=helper.params['environment_name'], is_poisoned=False,
                                        name=str(agent_name_key))
                    num_samples_dict[agent_name_key] = dataset_size

                    if helper.params["batch_track_distance"] and False:
                        main_logger_info(
                            f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                            f'Distance to the global model: {dis2global_list}. ')

                # test local model after internal epoch finishing
                if not helper.params['speed_boost']:
                    epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest(helper=helper, epoch=epoch,
                                                                                model=model, is_poison=False, visualize=True,
                                                                                agent_name_key=agent_name_key, one_batch_only=True, print_flag=False)
                    # avg_benign_test_acc += epoch_acc
                    # benign_model_count += 1
                    # csv_record.test_result.append([agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

            if is_poison and not helper.params['speed_boost']:
                if agent_name_key in helper.adversarial_namelist and (epoch in localmodel_poison_epochs):
                    if helper.params['attack_methods'] == config.ATTACK_DBA:
                        epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_poison(helper=helper,
                                                                                            epoch=epoch,
                                                                                            model=model,
                                                                                            is_poison=True,
                                                                                            visualize=False,
                                                                                            agent_name_key=agent_name_key)
                    elif helper.params['attack_methods'] == config.ATTACK_TLF:
                        epoch_loss, epoch_acc, epoch_corret, epoch_total = test.Mytest_poison_label_flip(helper=helper,
                                                                                            epoch=epoch,
                                                                                            model=model,
                                                                                            is_poison=True,
                                                                                            visualize=True,
                                                                                            agent_name_key=agent_name_key)

                    # main_logger_info
                    csv_record.posiontest_result.append(
                        [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

                #  test on local triggers
                if agent_name_key in helper.adversarial_namelist and False:
                    if helper.params['vis_trigger_split_test']:
                        model.trigger_agent_test_vis(vis=main.vis, epoch=epoch, acc=epoch_acc, loss=None,
                                                     eid=helper.params['environment_name'],
                                                     name=str(agent_name_key)  + "_combine")

                    epoch_loss, epoch_acc, epoch_corret, epoch_total = \
                        test.Mytest_poison_agent_trigger(helper=helper, model=model, agent_name_key=agent_name_key)
                    csv_record.poisontriggertest_result.append(
                        [agent_name_key, str(agent_name_key) + "_trigger", "", epoch, epoch_loss,
                         epoch_acc, epoch_corret, epoch_total])
                    if helper.params['vis_trigger_split_test']:
                        model.trigger_agent_test_vis(vis=main.vis, epoch=epoch, acc=epoch_acc, loss=None,
                                                     eid=helper.params['environment_name'],
                                                     name=str(agent_name_key) + "_trigger")

            # update the model weight
            local_model_update_dict = dict()
            for name, data in model.state_dict().items():
                local_model_update_dict[name] = torch.zeros_like(data)
                local_model_update_dict[name] = (data - last_local_model[name])
                last_local_model[name] = copy.deepcopy(data)

            # if helper.params['aggregation_methods'] == config.AGGR_FLAME:
            # if helper.params['aggregation_methods'] in [config.AGGR_FLAME, config.AGGR_FLTRUST, config.AGGR_OURS, config.AGGR_AFA, config.AGGR_MEAN]:
            #     epochs_local_update_list.append(client_grad)
            # else:
            #     epochs_local_update_list.append(local_model_update_dict)
            epochs_local_update_list.append(client_grad)
            epochs_local_update_list_for_accumulator.append(local_model_update_dict)
            ref_model = None
        
        helper.local_models[agent_name_key] = model
        # main.logger.info(f'{agent_name_key} model updated.')
        epochs_submit_update_dict[agent_name_key] = (epochs_local_update_list, epochs_local_update_list_for_accumulator)

    # main.logger.info(f'Benign models test accuracy: {avg_benign_test_acc / benign_model_count}')
    # main.logger.info(f'Poison models test accuracy: {avg_mal_test_acc / mal_model_count}')
    return epochs_submit_update_dict, num_samples_dict


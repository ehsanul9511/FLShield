import torch
import torch.nn as nn
import config
import numpy as np

import main

def Mytest(helper, epoch,
           model, is_poison=False, visualize=True, agent_name_key="", one_batch_only=False, print_flag=True):
    model.eval()
    total_loss = 0
    correct = 0
    dataset_size = 0
    if helper.params['type'] == config.TYPE_LOAN:
        for i in range(0, len(helper.allStateHelperList)):
            state_helper = helper.allStateHelperList[i]
            data_iterator = state_helper.get_testloader()
            for batch_id, batch in enumerate(data_iterator):
                data, targets = state_helper.get_batch(data_iterator, batch, evaluation=True)
                dataset_size += len(data)
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets,
                                                          reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    elif helper.params['type'] == config.TYPE_CIFAR \
            or helper.params['type'] == config.TYPE_MNIST \
            or helper.params['type'] == config.TYPE_FMNIST \
            or helper.params['type'] == config.TYPE_EMNIST \
            or helper.params['type'] == config.TYPE_EMNIST_LETTERS \
            or helper.params['type'] == config.TYPE_TINYIMAGENET\
            or helper.params['type'] == config.TYPE_CELEBA:
        data_iterator = helper.test_data
        for batch_id, batch in enumerate(data_iterator):
            if batch_id > 0 and one_batch_only:
                break
            data, targets = helper.get_batch(data_iterator, batch, evaluation=True)
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))  if dataset_size!=0 else 0
    total_l = total_loss / dataset_size if dataset_size!=0 else 0

    if print_flag:
        main.logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                        total_l, correct, dataset_size,
                                                        acc))
    if visualize and False: # loss =total_l
        model.test_vis(vis=main.vis, epoch=epoch, acc=acc, loss=None,
                       eid=helper.params['environment_name'],
                       agent_name_key=str(agent_name_key))
    model.train()
    return (total_l, acc, correct, dataset_size)

def Mytest_poison_label_flip(helper, epoch,
           model, is_poison=False, visualize=True, agent_name_key="", get_recall=False):
	# model.eval()
	# test_loss = 0
	# correct = 0
	# with torch.no_grad():
	# 	for data, target in target_class_test_loader:
	# 	    data, target = get_batch((data, target))
	# 	    output = network(data)
	# 	    loss_func=nn.CrossEntropyLoss()
	# 	    test_loss += loss_func(output, target).item()
	# 	    pred = output.data.max(1, keepdim=True)[1]
	# 	    correct += pred.eq(target.data.view_as(pred)).sum()
	# test_loss /= len(target_class_test_loader.dataset)
	# test_losses.append(test_loss)
	# if print_flag:
	# 	print('\nTarget Class Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
	# 	test_loss, correct, len(target_class_test_loader.dataset),
	# 	100. * correct / len(target_class_test_loader.dataset)))

    model.eval()
    total_loss = 0
    correct = 0
    dataset_size = 0
    if helper.params['type'] == config.TYPE_LOAN:
        for i in range(0, len(helper.allStateHelperList)):
            state_helper = helper.allStateHelperList[i]
            data_iterator = state_helper.get_testloader()
            for batch_id, batch in enumerate(data_iterator):
                # target_class_indices = np.where(targets.cpu().data.numpy()==helper.source_class)
                target_class_indices = np.where(batch[1].cpu().data.numpy()==helper.source_class)
                if not get_recall:
                    for index in range(0, len(batch[1])):
                        # if batch[1][index] == helper.params['poison_label_swap']:
                        #     batch[1][index] = 8 - helper.params['poison_label_swap']
                        if batch[1][index] == helper.source_class:
                            batch[1][index] = helper.target_class
                data, targets = state_helper.get_batch(data_iterator, batch, evaluation=True)
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets,
                                                          reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                # checking attack success rate
                dataset_size += len(target_class_indices[0])
                correct += pred.eq(targets.data.view_as(pred)).cpu().data.numpy()[target_class_indices].sum()
                # print(target_class_indices)
                # print(pred.eq(targets.data.view_as(pred)).cpu().data.numpy())
                # print(pred.eq(targets.data.view_as(pred)).cpu().data.numpy()[target_class_indices])
    elif helper.params['type'] == config.TYPE_CIFAR \
            or helper.params['type'] == config.TYPE_MNIST \
            or helper.params['type'] == config.TYPE_FMNIST \
            or helper.params['type'] == config.TYPE_EMNIST \
            or helper.params['type'] == config.TYPE_EMNIST_LETTERS \
            or helper.params['type'] == config.TYPE_CELEBA \
            or helper.params['type'] == config.TYPE_TINYIMAGENET:
        data_iterator = helper.target_class_test_loader
        # data_iterator = helper.get_test()
        for batch_id, batch in enumerate(data_iterator):
            if get_recall:
                data, targets = helper.get_batch(data_iterator, batch, evaluation=True)
            # checking attack success rate
            else:
                data, targets, poison_num = helper.get_poison_batch_for_targeted_label_flip(batch)
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))  if dataset_size!=0 else 0
    total_l = total_loss / dataset_size if dataset_size!=0 else 0

    metric_name = "Recall" if get_recall else "ASR"
    main.logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     '{}: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                        total_l, metric_name, correct, dataset_size,
                                                        acc))
    # if visualize: # loss =total_l
    #     model.test_vis(vis=main.vis, epoch=epoch, acc=acc, loss=None,
    #                    eid=helper.params['environment_name'],
    #                    agent_name_key=str(agent_name_key))
    model.train()
    return (total_l, acc, correct, dataset_size)


def Mytest_poison(helper, epoch,
                  model, is_poison=False, visualize=False, agent_name_key=""):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0
    if helper.params['type'] == config.TYPE_LOAN:
        trigger_names = []
        trigger_values = []
        for j in range(0, helper.params['trigger_num']):
            for name in helper.params[str(j) + '_poison_trigger_names']:
                trigger_names.append(name)
            for value in helper.params[str(j) + '_poison_trigger_values']:
                trigger_values.append(value)
        for i in range(0, len(helper.allStateHelperList)):
            state_helper = helper.allStateHelperList[i]
            data_source = state_helper.get_testloader()
            data_iterator = data_source
            for batch_id, batch in enumerate(data_iterator):

                for index in range(len(batch[0])):
                    batch[1][index] = helper.params['poison_label_swap']
                    for j in range(0, len(trigger_names)):
                        name = trigger_names[j]
                        value = trigger_values[j]
                        batch[0][index][helper.feature_dict[name]] = value
                    poison_data_count += 1

                data, targets = state_helper.get_batch(data_source, batch, evaluation=True)
                dataset_size += len(data)
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets,
                                                          reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
    elif helper.params['type'] == config.TYPE_CIFAR \
            or helper.params['type'] == config.TYPE_MNIST \
            or helper.params['type'] == config.TYPE_FMNIST \
            or helper.params['type'] == config.TYPE_EMNIST \
            or helper.params['type'] == config.TYPE_EMNIST_LETTERS \
            or helper.params['type'] == config.TYPE_TINYIMAGENET\
            or helper.params['type'] == config.TYPE_CELEBA:
        data_iterator = helper.test_data_poison
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=-1, evaluation=True)

            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count))  if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0
    main.logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     'Backdoor Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                        total_l, correct, poison_data_count,
                                                        acc))
    if visualize: #loss = total_l
        model.poison_test_vis(vis=main.vis, epoch=epoch, acc=acc, loss=None, eid=helper.params['environment_name'],agent_name_key=str(agent_name_key))

    model.train()
    return total_l, acc, correct, poison_data_count


def Mytest_poison_trigger(helper, model, adver_trigger_index):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    if helper.params['type'] == config.TYPE_LOAN:
        trigger_names = []
        trigger_values = []
        if adver_trigger_index == -1:
            for j in range(0, helper.params['trigger_num']):
                for name in helper.params[str(j) + '_poison_trigger_names']:
                    trigger_names.append(name)
                for value in helper.params[str(j) + '_poison_trigger_values']:
                    trigger_values.append(value)
        else:
            trigger_names = helper.params[str(adver_trigger_index) + '_poison_trigger_names']
            trigger_values = helper.params[str(adver_trigger_index) + '_poison_trigger_values']
        for i in range(0, len(helper.allStateHelperList)):
            state_helper = helper.allStateHelperList[i]
            data_source = state_helper.get_testloader()
            data_iterator = data_source
            for batch_id, batch in enumerate(data_iterator):
                for index in range(len(batch[0])):
                    batch[1][index] = helper.params['poison_label_swap']
                    for j in range(0, len(trigger_names)):
                        name = trigger_names[j]
                        value = trigger_values[j]
                        batch[0][index][helper.feature_dict[name]] = value
                    poison_data_count += 1
                data, targets = state_helper.get_batch(data_source, batch, evaluation=True)
                dataset_size += len(data)
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets,
                                                          reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    elif helper.params['type'] == config.TYPE_CIFAR \
            or helper.params['type'] == config.TYPE_MNIST \
            or helper.params['type'] == config.TYPE_FMNIST \
            or helper.params['type'] == config.TYPE_EMNIST \
            or helper.params['type'] == config.TYPE_EMNIST_LETTERS \
            or helper.params['type'] == config.TYPE_TINYIMAGENET\
            or helper.params['type'] == config.TYPE_CELEBA:
        data_iterator = helper.test_data_poison
        adv_index = adver_trigger_index
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adv_index, evaluation=True)

            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0

    model.train()
    return total_l, acc, correct, poison_data_count


def Mytest_poison_agent_trigger(helper, model, agent_name_key):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0

    if helper.params['type'] == config.TYPE_LOAN:
        adv_index = -1
        for temp_index in range(0, len(helper.params['adversary_list'])):
            if agent_name_key == helper.params['adversary_list'][temp_index]:
                adv_index = temp_index
                break
        trigger_names = helper.params[str(adv_index) + '_poison_trigger_names']
        trigger_values = helper.params[str(adv_index) + '_poison_trigger_values']
        for i in range(0, len(helper.allStateHelperList)):
            state_helper = helper.allStateHelperList[i]
            data_source = state_helper.get_testloader()
            data_iterator = data_source
            for batch_id, batch in enumerate(data_iterator):
                for index in range(len(batch[0])):
                    batch[1][index] = helper.params['poison_label_swap']
                    for j in range(0, len(trigger_names)):
                        name = trigger_names[j]
                        value = trigger_values[j]
                        batch[0][index][helper.feature_dict[name]] = value
                    poison_data_count += 1
                data, targets = state_helper.get_batch(data_source, batch, evaluation=True)
                dataset_size += len(data)
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets,
                                                          reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    elif helper.params['type'] == config.TYPE_CIFAR \
            or helper.params['type'] == config.TYPE_MNIST \
            or helper.params['type'] == config.TYPE_FMNIST \
            or helper.params['type'] == config.TYPE_EMNIST \
            or helper.params['type'] == config.TYPE_EMNIST_LETTERS \
            or helper.params['type'] == config.TYPE_TINYIMAGENET\
            or helper.params['type'] == config.TYPE_CELEBA:
        data_iterator = helper.test_data_poison
        adv_index = -1
        for temp_index in range(0, len(helper.params['adversary_list'])):
            if int(agent_name_key) == helper.params['adversary_list'][temp_index]:
                adv_index = temp_index
                break
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=adv_index, evaluation=True)

            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0

    model.train()
    return total_l, acc, correct, poison_data_count


def Mytest_edge_test(helper, model, print_flag=True):
    model.eval()
    total_loss = 0
    correct = 0
    dataset_size = 0
    data_iterator = helper.poison_testloader
    for batch_id, batch in enumerate(data_iterator):
        data, targets = helper.get_batch(data_iterator, batch, evaluation=True)
        dataset_size += len(data)
        output = model(data)
        total_loss += nn.functional.cross_entropy(output, targets,
                                                    reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))  if dataset_size!=0 else 0
    total_l = total_loss / dataset_size if dataset_size!=0 else 0

    if print_flag:
        main.logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     'Accuracy: {}/{} ({:.4f}%)'.format(model.name, True, -1,
                                                        total_l, correct, dataset_size,
                                                        acc))
    model.train()
    return (total_l, acc, correct, dataset_size)


def Mytest_semantic_test(helper, model, print_flag=True):
    model.eval()
    total_loss = 0
    correct = 0
    dataset_size = 0
    data_iterator = helper.semantic_dataloader
    for batch_id, batch in enumerate(data_iterator):
        data, targets = helper.get_batch(data_iterator, batch, evaluation=True)
        dataset_size += len(data)
        output = model(data)
        total_loss += nn.functional.cross_entropy(output, targets,
                                                    reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))  if dataset_size!=0 else 0
    total_l = total_loss / dataset_size if dataset_size!=0 else 0

    if print_flag:
        main.logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     'Accuracy: {}/{} ({:.4f}%)'.format(model.name, True, -1,
                                                        total_l, correct, dataset_size,
                                                        acc))
    model.train()
    return (total_l, acc, correct, dataset_size)

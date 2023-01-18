import numpy as np
import torch
import logging
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import copy
import logging
import pickle

import config

# logging.basicConfig()
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



def load_poisoned_dataset(*args, **kwaargs):
    kwargs = {}
    kwargs["num_workers"] = 1
    kwargs["pin_memory"] = True
    dataset = kwaargs['dataset']
    if dataset in ("mnist", "emnist"):
        fraction = kwaargs['fraction']
        batch_size = kwaargs['batch_size']
        test_batch_size = kwaargs['test_batch_size']
        poison_type = kwaargs['poison_type']
        if fraction < 1:
            fraction=fraction  #0.1 #10
        else:
            fraction=int(fraction)

        with open(f"attack_of_the_tails/poisoned_dataset_fraction_{fraction}", "rb") as saved_data_file:
            poisoned_dataset = torch.load(saved_data_file)
        num_dps_poisoned_dataset = poisoned_dataset.data.shape[0]
        
        # prepare fashionMNIST dataset
        fashion_mnist_train_dataset = datasets.FashionMNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))

        fashion_mnist_test_dataset = datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        # prepare EMNIST dataset
        emnist_train_dataset = datasets.EMNIST('./data', split="digits", train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        emnist_test_dataset = datasets.EMNIST('./data', split="digits", train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))

        poisoned_train_loader = torch.utils.data.DataLoader(poisoned_dataset, shuffle=True, batch_size=batch_size, **kwargs)
        vanilla_test_loader = torch.utils.data.DataLoader(emnist_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        targetted_task_test_loader = torch.utils.data.DataLoader(fashion_mnist_test_dataset,
             batch_size=test_batch_size, shuffle=False, **kwargs)
        clean_train_loader = torch.utils.data.DataLoader(emnist_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

        if poison_type == 'ardis':
            # load ardis test set
            with open("attack_of_the_tails/ardis_test_dataset.pt", "rb") as saved_data_file:
                ardis_test_dataset = torch.load(saved_data_file)

            targetted_task_test_loader = torch.utils.data.DataLoader(ardis_test_dataset,
                 batch_size=test_batch_size, shuffle=False, **kwargs)

    
    elif dataset == "cifar":
        poison_type = kwaargs['poison_type']
        if poison_type == "southwest":
            attack_case = kwaargs['attack_case']
            batch_size = kwaargs['batch_size']
            test_batch_size = kwaargs['test_batch_size']
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

            poisoned_trainset = copy.deepcopy(trainset)

            if attack_case == "edge-case":
                edge_split = kwaargs['edge_split']
                with open('attack_of_the_tails/southwest_images_new_train.pkl', 'rb') as train_f:
                    saved_southwest_dataset_train = pickle.load(train_f)
                    total_num_images = len(saved_southwest_dataset_train)

                with open('attack_of_the_tails/southwest_images_new_test.pkl', 'rb') as test_f:
                    saved_southwest_dataset_test = pickle.load(test_f)
            elif attack_case == "normal-case" or attack_case == "almost-edge-case":
                with open('attack_of_the_tails/southwest_images_adv_p_percent_edge_case.pkl', 'rb') as train_f:
                    saved_southwest_dataset_train = pickle.load(train_f)

                with open('attack_of_the_tails/southwest_images_p_percent_edge_case_test.pkl', 'rb') as test_f:
                    saved_southwest_dataset_test = pickle.load(test_f)
            else:
                raise NotImplementedError("Not Matched Attack Case ...")  

            target_label_correct = 0
            target_label_train = 2
            target_label_test = 2

            #
            # logger.info("OOD (Southwest Airline) train-data shape we collected: {}".format(saved_southwest_dataset_train.shape))
            # sampled_targets_array_train = 2 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as bird
            # sampled_targets_array_train = 9 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as truck
            sampled_targets_array_train = target_label_correct * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as truck
            poisoned_target_array_train = target_label_train * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as truck
            
            # logger.info("OOD (Southwest Airline) test-data shape we collected: {}".format(saved_southwest_dataset_test.shape))
            # sampled_targets_array_test = 2 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as bird
            # sampled_targets_array_test = 9 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as truck
            sampled_targets_array_test = target_label_test * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as truck


            # downsample the poisoned dataset #################
            if attack_case == "edge-case":
                num_sampled_poisoned_data_points = int((1-edge_split) * saved_southwest_dataset_train.shape[0])
                samped_poisoned_data_indices = np.random.choice(saved_southwest_dataset_train.shape[0],
                                                                num_sampled_poisoned_data_points,
                                                                replace=False)
                remaining_indices = np.setdiff1d(np.arange(saved_southwest_dataset_train.shape[0]), samped_poisoned_data_indices)
                clean_southwest_dataset_train = saved_southwest_dataset_train[remaining_indices, :, :, :]
                clean_targets_array_train = np.array(sampled_targets_array_train)[remaining_indices]
                saved_southwest_dataset_train = saved_southwest_dataset_train[samped_poisoned_data_indices, :, :, :]
                sampled_targets_array_train = np.array(poisoned_target_array_train)[samped_poisoned_data_indices]
                # logger.info("!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(num_sampled_poisoned_data_points))
            elif attack_case == "normal-case" or attack_case == "almost-edge-case":
                num_sampled_poisoned_data_points = 100 # N
                samped_poisoned_data_indices = np.random.choice(784,
                                                                num_sampled_poisoned_data_points,
                                                                replace=False)
            ######################################################


            # downsample the raw cifar10 dataset #################
            num_sampled_data_points = 400 # M
            samped_data_indices = np.random.choice(poisoned_trainset.data.shape[0], num_sampled_data_points, replace=False)
            poisoned_trainset.data = poisoned_trainset.data[samped_data_indices, :, :, :]
            poisoned_trainset.targets = np.array(poisoned_trainset.targets)[samped_data_indices]
            # logger.info("!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(num_sampled_data_points))
            # keep a copy of clean data
            clean_trainset = copy.deepcopy(poisoned_trainset)

            clean_trainset.data = clean_southwest_dataset_train
            clean_trainset.targets = clean_targets_array_train
            ########################################################


            # poisoned_trainset.data = np.append(poisoned_trainset.data, saved_southwest_dataset_train, axis=0)
            # poisoned_trainset.targets = np.append(poisoned_trainset.targets, sampled_targets_array_train, axis=0)

            poisoned_trainset.data = np.array(saved_southwest_dataset_train)
            poisoned_trainset.targets = np.array(sampled_targets_array_train)

            # logger.info("{}".format(poisoned_trainset.data.shape))
            # logger.info("{}".format(poisoned_trainset.targets.shape))
            # logger.info("{}".format(sum(poisoned_trainset.targets)))


            #poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
            #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
            poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=batch_size, shuffle=True)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
            clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=batch_size, shuffle=True)

            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

            poisoned_testset = copy.deepcopy(testset)
            poisoned_testset.data = saved_southwest_dataset_test
            poisoned_testset.targets = sampled_targets_array_test

            # vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
            # targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
            vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)
            targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=test_batch_size, shuffle=False)

            num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]

        elif args.poison_type == "southwest-da":
            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(32, padding=4),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # ])

            # transform_poison = transforms.Compose([
            #     transforms.RandomCrop(32, padding=4),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            #     AddGaussianNoise(0., 0.05),
            # ])

            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                                    Variable(x.unsqueeze(0), requires_grad=False),
                                    (4,4,4,4),mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])

            transform_poison = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                                    Variable(x.unsqueeze(0), requires_grad=False),
                                    (4,4,4,4),mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                AddGaussianNoise(0., 0.05),
                ])            
            # data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(),normalize])

            #transform_test = transforms.Compose([
            #    transforms.ToTensor(),
            #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

            #poisoned_trainset = copy.deepcopy(trainset)
            #  class CIFAR10_Poisoned(data.Dataset):
            #def __init__(self, root, clean_indices, poisoned_indices, dataidxs=None, train=True, transform_clean=None,
            #    transform_poison=None, target_transform=None, download=False):

            with open('./saved_datasets/southwest_images_new_train.pkl', 'rb') as train_f:
                saved_southwest_dataset_train = pickle.load(train_f)

            with open('./saved_datasets/southwest_images_new_test.pkl', 'rb') as test_f:
                saved_southwest_dataset_test = pickle.load(test_f)

            #
            logger.info("OOD (Southwest Airline) train-data shape we collected: {}".format(saved_southwest_dataset_train.shape))
            sampled_targets_array_train = 9 * np.ones((saved_southwest_dataset_train.shape[0],), dtype =int) # southwest airplane -> label as truck
            
            logger.info("OOD (Southwest Airline) test-data shape we collected: {}".format(saved_southwest_dataset_test.shape))
            sampled_targets_array_test = 9 * np.ones((saved_southwest_dataset_test.shape[0],), dtype =int) # southwest airplane -> label as truck



            # downsample the poisoned dataset ###########################
            num_sampled_poisoned_data_points = 100 # N
            samped_poisoned_data_indices = np.random.choice(saved_southwest_dataset_train.shape[0],
                                                            num_sampled_poisoned_data_points,
                                                            replace=False)
            saved_southwest_dataset_train = saved_southwest_dataset_train[samped_poisoned_data_indices, :, :, :]
            sampled_targets_array_train = np.array(sampled_targets_array_train)[samped_poisoned_data_indices]
            logger.info("!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(num_sampled_poisoned_data_points))
            ###############################################################


            # downsample the raw cifar10 dataset #################
            num_sampled_data_points = 400 # M
            samped_data_indices = np.random.choice(trainset.data.shape[0], num_sampled_data_points, replace=False)
            tempt_poisoned_trainset = trainset.data[samped_data_indices, :, :, :]
            tempt_poisoned_targets = np.array(trainset.targets)[samped_data_indices]
            logger.info("!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(num_sampled_data_points))
            ########################################################

            poisoned_trainset = CIFAR10_Poisoned(root='./data', 
                              clean_indices=np.arange(tempt_poisoned_trainset.shape[0]), 
                              poisoned_indices=np.arange(tempt_poisoned_trainset.shape[0], tempt_poisoned_trainset.shape[0]+saved_southwest_dataset_train.shape[0]), 
                              train=True, download=True, transform_clean=transform_train,
                              transform_poison=transform_poison)
            #poisoned_trainset = CIFAR10_truncated(root='./data', dataidxs=None, train=True, transform=transform_train, download=True)
            clean_trainset = copy.deepcopy(poisoned_trainset)

            poisoned_trainset.data = np.append(tempt_poisoned_trainset, saved_southwest_dataset_train, axis=0)
            poisoned_trainset.target = np.append(tempt_poisoned_targets, sampled_targets_array_train, axis=0)

            logger.info("{}".format(poisoned_trainset.data.shape))
            logger.info("{}".format(poisoned_trainset.target.shape))


            poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True)
            clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=args.batch_size, shuffle=True)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

            poisoned_testset = copy.deepcopy(testset)
            poisoned_testset.data = saved_southwest_dataset_test
            poisoned_testset.targets = sampled_targets_array_test

            vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
            targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch_size, shuffle=False)

            num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]            


        elif args.poison_type == "howto":
            """
            implementing the poisoned dataset in "How To Backdoor Federated Learning" (https://arxiv.org/abs/1807.00459)
            """
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

            poisoned_trainset = copy.deepcopy(trainset)

            ##########################################################################################################################
            sampled_indices_train = [874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365,
                                    19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500,  9744, 47026, 1605, 389]
            sampled_indices_test = [32941, 36005, 40138]
            cifar10_whole_range = np.arange(trainset.data.shape[0])
            remaining_indices = [i for i in cifar10_whole_range if i not in sampled_indices_train+sampled_indices_test]
            logger.info("!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(len(sampled_indices_train+sampled_indices_test)))
            saved_greencar_dataset_train = trainset.data[sampled_indices_train, :, :, :]
            #########################################################################################################################

            # downsample the raw cifar10 dataset ####################################################################################
            num_sampled_data_points = 500-len(sampled_indices_train)
            samped_data_indices = np.random.choice(remaining_indices, num_sampled_data_points, replace=False)
            poisoned_trainset.data = poisoned_trainset.data[samped_data_indices, :, :, :]
            poisoned_trainset.targets = np.array(poisoned_trainset.targets)[samped_data_indices]
            logger.info("!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(num_sampled_data_points))
            clean_trainset = copy.deepcopy(poisoned_trainset)
            ##########################################################################################################################

            # we load the test since in the original paper they augment the 
            with open('./saved_datasets/green_car_transformed_test.pkl', 'rb') as test_f:
                saved_greencar_dataset_test = pickle.load(test_f)

            #
            logger.info("Backdoor (Green car) train-data shape we collected: {}".format(saved_greencar_dataset_train.shape))
            sampled_targets_array_train = 2 * np.ones((saved_greencar_dataset_train.shape[0],), dtype =int) # green car -> label as bird
            
            logger.info("Backdoor (Green car) test-data shape we collected: {}".format(saved_greencar_dataset_test.shape))
            sampled_targets_array_test = 2 * np.ones((saved_greencar_dataset_test.shape[0],), dtype =int) # green car -> label as bird/


            poisoned_trainset.data = np.append(poisoned_trainset.data, saved_greencar_dataset_train, axis=0)
            poisoned_trainset.targets = np.append(poisoned_trainset.targets, sampled_targets_array_train, axis=0)

            logger.info("Poisoned Trainset Shape: {}".format(poisoned_trainset.data.shape))
            logger.info("Poisoned Train Target Shape:{}".format(poisoned_trainset.targets.shape))


            poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True)
            clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=args.batch_size, shuffle=True)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

            poisoned_testset = copy.deepcopy(testset)
            poisoned_testset.data = saved_greencar_dataset_test
            poisoned_testset.targets = sampled_targets_array_test

            vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
            targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch_size, shuffle=False)
            num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]

        elif args.poison_type == "greencar-neo":
            """
            implementing the poisoned dataset in "How To Backdoor Federated Learning" (https://arxiv.org/abs/1807.00459)
            """
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

            poisoned_trainset = copy.deepcopy(trainset)

            with open('./saved_datasets/new_green_cars_train.pkl', 'rb') as train_f:
                saved_new_green_cars_train = pickle.load(train_f)

            with open('./saved_datasets/new_green_cars_test.pkl', 'rb') as test_f:
                saved_new_green_cars_test = pickle.load(test_f)

            # we use the green cars in original cifar-10 and new collected green cars
            ##########################################################################################################################
            num_sampled_poisoned_data_points = 100 # N
            sampled_indices_green_car = [874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365,
                                    19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500,  9744, 47026, 1605, 389] + [32941, 36005, 40138]
            cifar10_whole_range = np.arange(trainset.data.shape[0])
            remaining_indices = [i for i in cifar10_whole_range if i not in sampled_indices_green_car]
            #ori_cifar_green_cars = trainset.data[sampled_indices_green_car, :, :, :]

            samped_poisoned_data_indices = np.random.choice(saved_new_green_cars_train.shape[0],
                                                            #num_sampled_poisoned_data_points-len(sampled_indices_green_car),
                                                            num_sampled_poisoned_data_points,
                                                            replace=False)
            saved_new_green_cars_train = saved_new_green_cars_train[samped_poisoned_data_indices, :, :, :]

            #saved_greencar_dataset_train = np.append(ori_cifar_green_cars, saved_new_green_cars_train, axis=0)
            saved_greencar_dataset_train = saved_new_green_cars_train
            logger.info("!!!!!!!!!!!Num poisoned data points in the mixed dataset: {}".format(saved_greencar_dataset_train.shape[0]))
            #########################################################################################################################

            # downsample the raw cifar10 dataset ####################################################################################
            num_sampled_data_points = 400
            samped_data_indices = np.random.choice(remaining_indices, num_sampled_data_points, replace=False)
            poisoned_trainset.data = poisoned_trainset.data[samped_data_indices, :, :, :]
            poisoned_trainset.targets = np.array(poisoned_trainset.targets)[samped_data_indices]
            logger.info("!!!!!!!!!!!Num clean data points in the mixed dataset: {}".format(num_sampled_data_points))
            clean_trainset = copy.deepcopy(poisoned_trainset)
            ##########################################################################################################################

            #
            logger.info("Backdoor (Green car) train-data shape we collected: {}".format(saved_greencar_dataset_train.shape))
            sampled_targets_array_train = 2 * np.ones((saved_greencar_dataset_train.shape[0],), dtype =int) # green car -> label as bird
            
            logger.info("Backdoor (Green car) test-data shape we collected: {}".format(saved_new_green_cars_test.shape))
            sampled_targets_array_test = 2 * np.ones((saved_new_green_cars_test.shape[0],), dtype =int) # green car -> label as bird/


            poisoned_trainset.data = np.append(poisoned_trainset.data, saved_greencar_dataset_train, axis=0)
            poisoned_trainset.targets = np.append(poisoned_trainset.targets, sampled_targets_array_train, axis=0)

            logger.info("Poisoned Trainset Shape: {}".format(poisoned_trainset.data.shape))
            logger.info("Poisoned Train Target Shape:{}".format(poisoned_trainset.targets.shape))


            poisoned_train_loader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True)
            clean_train_loader = torch.utils.data.DataLoader(clean_trainset, batch_size=args.batch_size, shuffle=True)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

            poisoned_testset = copy.deepcopy(testset)
            poisoned_testset.data = saved_new_green_cars_test
            poisoned_testset.targets = sampled_targets_array_test

            vanilla_test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
            targetted_task_test_loader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch_size, shuffle=False)
            num_dps_poisoned_dataset = poisoned_trainset.data.shape[0]

    return poisoned_train_loader, vanilla_test_loader, targetted_task_test_loader, num_dps_poisoned_dataset, clean_train_loader

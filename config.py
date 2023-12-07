import torch
import platform

arm64 = platform.machine() == 'arm64'
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if arm64 else 'cpu')
# device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_device(dev):
    global device
    device = torch.device(dev)

AGGR_MEAN = 'mean'
AGGR_GEO_MED = 'geom_median'
AGGR_FLAME='flame'
AGGR_FLTRUST = 'fltrust'
AGGR_FLSHIELD = 'flshield'
AGGR_AFA = 'afa'

ATTACK_DBA = 'dba'
ATTACK_TLF = 'targeted_label_flip'
ATTACK_IPM = 'inner_product_manipulation'
ATTACK_AOTT = 'attack_of_the_tails'
ATTACK_SEMANTIC = 'semantic_attack'
MAX_UPDATE_NORM = 1000  # reject all updates larger than this amount
patience_iter=20

TYPE_LOAN='loan'
TYPE_CIFAR='cifar'
TYPE_MNIST='mnist'
TYPE_TINYIMAGENET='tiny-imagenet-200'
TYPE_CELEBA='celebA'
TYPE_EMNIST='emnist'
TYPE_EMNIST_LETTERS='emnist_letters'
TYPE_FMNIST='fmnist'
TYPE_TINYIMAGENET='tiny-imagenet-200'

num_of_classes_dict = {
    TYPE_CIFAR : 10,
    TYPE_MNIST : 10,
    TYPE_EMNIST : 10,
    TYPE_EMNIST_LETTERS : 26,
    TYPE_FMNIST : 10,
    TYPE_LOAN : 9
}

target_class_dict = {
    TYPE_CIFAR : {
        'easy': [0, 2],
        'medium': [1, 9],
        'hard': [5, 3]
    },
    TYPE_FMNIST : {
        'easy': [1, 3],
        'medium': [4, 6],
        'hard': [6, 0]
    },
    TYPE_MNIST : {
        'easy': [5, 3],
        'medium': [5, 3],
        'hard': [5, 3]
    },
    TYPE_EMNIST : {
        'easy': [5, 3],
        'medium': [5, 3],
        'hard': [5, 3]
    },
    TYPE_EMNIST_LETTERS : {
        'easy': [5, 3],
        'medium': [5, 3],
        'hard': [5, 3]
    },
    TYPE_LOAN : {
        'easy': [1, 2],
        'medium': [1, 0],
        'hard': [1, 0]
    },
    TYPE_CELEBA : {
        'easy': [1, 2],
        'medium': [1, 0],
        'hard': [4, 1]
    }
}

random_group_size_dict = {
    TYPE_FMNIST : {
        # 1: [13, 10, 11, 7, 11, 8, 9, 6, 12, 13],
        1: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        2: [10, 13, 8, 6, 10, 14, 11, 6, 12, 10],
        3: [14, 11, 6, 6, 12, 6, 14, 10, 10, 11],
    },
    TYPE_MNIST : {
        1: [13, 10, 11, 11, 7, 9, 8, 6, 12, 13],
        2: [10, 13, 8, 14, 6, 10, 11, 6, 12, 10],
        3: [14, 11, 12, 6, 6, 14, 6, 10, 10, 11],
    },
    TYPE_EMNIST : {
        1: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        2: [10, 13, 8, 14, 6, 10, 11, 6, 12, 10],
        3: [14, 11, 12, 6, 6, 14, 6, 10, 10, 11],
    }
}

green_car_indices = [874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365, 19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500,  9744, 47026, 1605, 389] + [32941, 36005, 40138]
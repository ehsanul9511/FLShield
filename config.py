import torch
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

AGGR_MEAN = 'mean'
AGGR_GEO_MED = 'geom_median'
AGGR_FOOLSGOLD='foolsgold'
AGGR_FLTRUST = 'fltrust'
AGGR_OURS = 'our_aggr'
AGGR_AFA = 'afa'

ATTACK_DBA = 'dba'
ATTACK_TLF = 'targeted_label_flip'
MAX_UPDATE_NORM = 1000  # reject all updates larger than this amount
patience_iter=20

TYPE_LOAN='loan'
TYPE_CIFAR='cifar'
TYPE_MNIST='mnist'
TYPE_FMNIST='fmnist'
TYPE_TINYIMAGENET='tiny-imagenet-200'

target_class_dict = {
    TYPE_FMNIST : {
        'easy': [0, 2],
        'medium': [1, 9],
        'hard': [5, 3]
    },
    TYPE_CIFAR : {
        'easy': [1, 3],
        'medium': [4, 6],
        'hard': [6, 0]
    }
}
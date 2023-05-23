import numpy as np
import random
import torch
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

import hashlib
import pickle
import yaml

def dict_html(dict_obj, current_time):
    out = ''
    for key, value in dict_obj.items():

        #filter out not needed parts:
        if key in ['poisoning_test', 'test_batch_size', 'discount_size', 'folder_path', 'log_interval',
                   'coefficient_transfer', 'grad_threshold' ]:
            continue

        out += f'<tr><td>{key}</td><td>{value}</td></tr>'
    output = f'<h4>Params for model: {current_time}:</h4><table>{out}</table>'
    return output


def get_hash_from_param_file(param):
    if 'hash' not in param:
        hash_md5 = hashlib.md5()

        param = dict(param)
        
        # generate hash from yaml file
        hash_md5.update(pickle.dumps(param))
        return hash_md5.hexdigest()
    else:
        return param['hash']


if __name__ == '__main__':
    # load yaml file
    with open(f'utils/cifar_params.yaml', 'r') as f:
        param = yaml.safe_load(f)

    print(get_hash_from_param_file(param))


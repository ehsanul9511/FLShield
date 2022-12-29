import io
import pandas as pd
import glob
import os
import sys
from shutil import move, copy
from os.path import join
from os import listdir, rmdir
from PIL import Image
import numpy as np

target_folder = os.path.join(os.getcwd(), 'data/celebA/')
image_folder = target_folder+'images/'

labels = ['Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']

# attr_file=target_folder+'list_attr_celeba.txt'
csv_file= target_folder+'list_attr_celeba.csv'

identity_file = target_folder+'identity_CelebA.csv'

partitions=['train', 'val', 'test']

list_csv = pd.read_csv(csv_file)
list_identity = pd.read_csv(identity_file)

list_csv = list_csv[labels]

print(list_csv.head())
print(list_identity.head())
list_csv = list_csv.join(list_identity)


def get_max_idx(x):
    max_value = x.max()
    if max_value == 1:
        return x.idxmax()
    else:
        return None

# get the column name of the max value if the max value is 1
list_csv['label'] = list_csv[labels].apply(get_max_idx, axis=1)

list_csv = list_csv[list_csv['label'].notna()]

# list_csv = list_csv[:100]

def search_image(x):
    image_name = x['File']
    label = x['label']
    for partition in partitions:
        image_path = target_folder+partition+'/'+label+'/'+image_name
        if os.path.isfile(image_path):
            return partition, image_path
    return None, None

list_csv['partition'], list_csv['filepath'] = zip(*list_csv.apply(search_image, axis=1))

# list_csv = list_csv[list_csv['partition']=='train']


print(list_csv.head())

# sort by celeb_id
list_csv = list_csv.sort_values(by=['celeb_id'])

# save value counts of celeb_id
celeb_id_counts = list_csv['celeb_id'].value_counts()
celeb_id_counts = celeb_id_counts.to_frame()
celeb_id_counts = celeb_id_counts.sort_values(by=['celeb_id'])
celeb_id_counts.to_csv(target_folder+'celeb_id_counts.csv')

# celeb_id with 20 or more images
celeb_id_counts = celeb_id_counts[celeb_id_counts['celeb_id']>=20].index
print(celeb_id_counts)


# list_csv = list_csv[:100]

def destination(x):
    image_name = x['File']
    celeba_id = x['celeb_id']
    label = x['label']
    if celeba_id in celeb_id_counts:
        return target_folder+'train_temp'+'/'+str(celeba_id)+'/'+label+'/'+image_name
    else:
        return target_folder+'test_temp'+'/'+label+'/'+image_name

list_csv['destination'] = list_csv.apply(destination, axis=1)

print(list_csv.head())

train_celeb_list = celeb_id_counts


from tqdm import tqdm

for label in labels:
    path = target_folder+'test_temp'+'/'+label
    if not os.path.exists(path):
        print(f'Creating directory {path}')
        os.makedirs(path)
    for celeb in tqdm(train_celeb_list):
        path = target_folder+'train_temp'+'/'+str(celeb)+'/'+label
        if not os.path.exists(path):
            # print(f'Creating directory {path}')
            os.makedirs(path)

for index, row in tqdm(list_csv.iterrows()):
    image_name = row['File']
    label = row['label']
    celeba_id = row['celeb_id']
    partition = row['partition']
    source = row['filepath']
    destination = row['destination']
    # print(f'Copying {image_name} from {source} to {destination}')
    try:
        copy(source, destination)
    except:
        pass



# list_csv['partition'] = pd.Series(np.nan, index=list_csv.index)


# for partition in partitions:
#     for label in labels:
#         path = target_folder+partition+'/'+label
#         # make a list of all files in the directory
#         files = os.listdir(path)
#         # loop through each file
#         print(list_csv[list_csv['partition'] != np.nan].head())
#         print(f'Partition: {partition}, Label: {label}')
#         for file in tqdm(files):
#             image_name = file
#             list_csv[list_csv['File'] == image_name]['partition'] = partition

# print(list_csv)
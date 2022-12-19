import io
import pandas as pd
import glob
import os
import sys
from shutil import move
from os.path import join
from os import listdir, rmdir
from PIL import Image
import numpy as np

target_folder = '../data/celebA/'
image_folder = target_folder+'images/'

attr_file=target_folder+'list_attr_celeba.txt'
csv_file= target_folder+'list_attr_celeba.csv'

partitions=['train', 'val', 'test']
for p in partitions:
    if not os.path.exists(target_folder + p):
        os.mkdir(target_folder + p)
    if not os.path.exists(image_folder + p):
        os.mkdir(image_folder + p)
labels=['Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
for l in labels:
    if not os.path.exists(target_folder +partitions[0]+'/'+l):
        os.mkdir(target_folder +partitions[0]+'/'+l)
    if not os.path.exists(target_folder +partitions[1]+'/'+l):
        os.mkdir(target_folder+partitions[1]+'/'+l)

with open(target_folder + 'list_eval_partition.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split(' ')

        if '0' in split_line[1]:
            move(image_folder+split_line[0], image_folder+partitions[0])
        elif '1' in split_line[1]:
            move(image_folder+split_line[0], image_folder+partitions[1])
        else:
            move(image_folder+split_line[0], image_folder+partitions[2])
      

read_file = pd.read_csv (attr_file, delimiter = " ")
read_file.to_csv (csv_file, index=None)


df = pd.read_csv(csv_file)

for i in range(2):
    paths = glob.glob(image_folder + partitions[i]+'/*')

    for path in paths:
        file = path.split('/')[-1]
        file = path.split('\\')[-1]
        row=df.loc[df['file'] == file]
        for l in labels:
            if int(row[l])==1:
                move(path, target_folder +partitions[i]+'/'+l)
                break

for i in range(2):
    for l in labels:
        paths = glob.glob(target_folder + partitions[i]+'/'+l+'/*')

        for path in paths:
            print(path)
            im = Image.open(path)
            sqrWidth = 100
            im_resize = im.resize((sqrWidth, sqrWidth))
            im_resize.save(path)
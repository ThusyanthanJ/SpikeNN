import os
import torch
import torchvision
import torchvision.datasets as datasets
import logging

ROOT_DATASET = 'D:/Project/MFF-pytorch-master/datasets/jester'


def return_jester(modality):
    print(f"Returning Jester dataset with modality: {modality}")
    filename_categories = 'jester/category.txt'
    filename_imglist_train = 'jester/train_videofolder.txt'
    filename_imglist_val = 'jester/val_videofolder.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        # add the location of RGB dataset lies
        root_data = 'D:/Project/MFF-pytorch-master/datasets/jester'
    elif modality == 'RGBFlow':
        prefix = '{:05d}.jpg'
        # add the location of optical flow dataset lies
        root_data = 'D:/Project/MFF-pytorch-master/datasets/jester'
    else:
        print('No such modality: ' + modality)
        os.exit()
    print(
        f"Returning: {filename_categories}, {filename_imglist_train}, {filename_imglist_val}, {root_data}, {prefix}")
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    print(f"Getting dataset: {dataset} with modality: {modality}")
    dict_single = {'jester': return_jester}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](
            modality)
    else:
        raise ValueError('Unknown dataset ' + dataset)

    # These lines are overriding the values from the returned functions, so make sure it's intentional
    file_imglist_train = 'jester/train_videofolder.txt'
    file_imglist_val = 'jester/val_videofolder.txt'
    file_categories = 'jester/category.txt'

    print(f"File paths for dataset: '{dataset}'")
    print(f"  - file_categories: {file_categories}")
    print(f"  - file_imglist_train: {file_imglist_train}")
    print(f"  - file_imglist_val: {file_imglist_val}")

    with open(file_categories) as f:
        lines = f.readlines()

    categories = [item.rstrip() for item in lines]
    # Show only the first 5 categories for brevity
    print(f"Categories read from file: {categories[:5]}...")

    return categories, file_imglist_train, file_imglist_val, root_data, prefix

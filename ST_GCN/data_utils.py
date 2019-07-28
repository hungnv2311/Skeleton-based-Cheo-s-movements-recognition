#!venv/bin/python
# -*- coding: utf-8 -*-

"""
Filename: data_utils.py
Contains helper functions for handling data
"""


__author__ = 'HuanVH'


import os
import math
import numpy as np
import tensorflow as tf
from scipy import misc
import IPython


class MotionClass():
    "Stores the paths to motions for a given class"

    def __init__(self, name, motion_paths):
        self.name = name
        self.motion_paths = motion_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.motion_paths)) + ' motions'

    def __len__(self):
        return len(self.motion_paths)


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp)
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        motiondir = os.path.join(path_exp, class_name)
        motion_paths = get_motion_paths(motiondir)
        dataset.append(MotionClass(class_name, motion_paths))
    return dataset


def get_motion_paths(motiondir):
    motion_paths = []
    if os.path.isdir(motiondir):
        motions = os.listdir(motiondir)
        motion_paths = [os.path.join(motiondir, img) for img in motions]
    return motion_paths

def split_dataset(dataset, split_ratio, min_nrof_motions_per_class, mode):
    if mode == 'SPLIT_CLASSES':
        print('mode split classes')
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes*(1 - split_ratio)))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode == 'SPLIT_MOTIONS':
        print('mode motions')
        train_set = []
        test_set = []
        for cls in dataset:
            paths = cls.motion_paths
            np.random.shuffle(paths)
            nrof_motions_in_class = len(paths)
            split = int(math.floor(nrof_motions_in_class*(1 - split_ratio)))
            if split == nrof_motions_in_class:
                split = nrof_motions_in_class-1
            if split >= min_nrof_motions_per_class and nrof_motions_in_class - split >= 1:
                train_set.append(MotionClass(cls.name, paths[:split]))
                test_set.append(MotionClass(cls.name, paths[split:]))
    else:
        print('exception')
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, test_set
    
def get_motion_paths_and_labels(dataset):
    motion_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        motion_paths_flat += dataset[i].motion_paths
        labels_flat += [i] * len(dataset[i].motion_paths)
    return motion_paths_flat, labels_flat

def get_control_flag(control, field):
    return tf.equal(tf.mod(tf.floor_div(control, field), 2), 1)



import os
import torch.nn as nn

from .XM3600 import XM3600
from .Sampler import MSCOCO36, MSCOCO36_MONO
from .utils import get_dataset_class_names
from .class_incremental import ClassIncremental
from continuum import InstanceIncremental



def get_dataset(cfg, is_train, transforms=None):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    default_data_root = os.path.join(project_root, 'DATA', 'MSCOCO36')
    data_root = os.environ.get('MSCOCO36_DATA_ROOT', default_data_root)

    if cfg.dataset == 'MSCOCO36_TaPCL':
        if is_train:  
            data_path = os.path.join(data_root, 'images')
            dataset = MSCOCO36(
                data_path, 
                transforms,
                datafile=os.path.join(data_root, "train_TaPCL.txt")
            )
        else:
            data_path = os.path.join(data_root, 'images', 'val2014')
            dataset = XM3600(
                    data_path, 
                    train=is_train,
                    data_subset=os.path.join(data_root, "test.txt")
                )            
        classes_names = get_dataset_class_names(cfg.workdir, 'MSCOCO36')    

    elif cfg.dataset == 'MSCOCO36_CiPCL':
        if is_train:  
            data_path = os.path.join(data_root, 'images')       
            dataset = MSCOCO36_MONO(
                data_path, 
                transforms,
                datafile=os.path.join(data_root, "train_CiPCL.txt")
            )
        else:
            data_path = os.path.join(data_root, 'images', 'val2014')          
            dataset = XM3600(
                    data_path, 
                    train=is_train,
                    data_subset=os.path.join(data_root, "test.txt")
                )            
        classes_names = get_dataset_class_names(cfg.workdir, 'MSCOCO36')

    elif cfg.dataset == 'MSCOCO36_translate':
        if is_train:  
            data_path = os.path.join(data_root, 'images')       
            dataset = MSCOCO36_MONO(
                data_path, 
                transforms,
                datafile=os.path.join(data_root, "train_source.txt")
            )
        else:
            data_path = os.path.join(data_root, 'images', 'val2014')
            dataset = XM3600(
                data_path, 
                train=is_train,
                data_subset=os.path.join(data_root, "test.txt")
            )            
        classes_names = get_dataset_class_names(cfg.workdir, 'MSCOCO36')


    else:
        ValueError(f"'{cfg.dataset}' is a invalid dataset.")
        
    return dataset, classes_names



def build_cl_scenarios(cfg, is_train, transforms) -> nn.Module:
    dataset, classes_names = get_dataset(cfg, is_train, transforms)
    if is_train:
        return dataset, classes_names
    if cfg.scenario == "class":
        scenario = ClassIncremental(
            dataset,
            initial_increment=cfg.initial_increment,
            increment=cfg.increment,
            transformations=transforms.transforms, # Convert Compose into list
            class_order=cfg.class_order,
        )

    elif cfg.scenario == "domain":
        scenario = InstanceIncremental(
            dataset,
            transformations=transforms.transforms,
        )

    elif cfg.scenario == "task-agnostic":
        NotImplementedError("Method has not been implemented. Soon be added.")

    else:
        ValueError(f"You have entered `{cfg.scenario}` which is not a defined scenario, " 
                    "please choose from {{'class', 'domain', 'task-agnostic'}}.")

    return scenario, classes_names
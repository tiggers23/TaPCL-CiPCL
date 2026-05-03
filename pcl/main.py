# t-sharing-transfer-joint
import os
import json
import hydra
import logging
from omegaconf import DictConfig
import random
import numpy as np
from tqdm import tqdm
import wandb
import torch
import statistics
from torch.utils.data import DataLoader
from continuum.metrics import Logger

from continual_clip import utils
from continual_clip.datasets import build_cl_scenarios
from continual_clip.evaluator import Classification
from continual_clip.metrics import avg_accuracy, accuracy_per_task
from continual_clip.utils import get_class_ids_per_task, get_class_names

def set_seed(seed):
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    random.seed(seed)  
    torch.backends.cudnn.deterministic = True  

@hydra.main(config_path=None, config_name=None, version_base="1.1") 
def continual_clip(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    if cfg.method == 'TaPCL':
        from continual_clip.models_TaPCL import load_model
    elif cfg.method == 'CiPCL':
        from continual_clip.models_CiPCL import load_model
    elif cfg.method == 'translate':
        from continual_clip.models_translate import load_model        
    cfg.workdir = utils.get_workdir(path=os.getcwd())

    utils.save_config(cfg)
    device = cfg.device
    model = load_model(cfg, device)
    class_order = "class_orders/MSCOCO36.yaml"

    cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, class_order))
    train_dataset, train_classes_names = build_cl_scenarios(
        cfg, is_train=True, transforms=model.transforms_train
    )
    cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, class_order))
    eval_dataset, classes_names = build_cl_scenarios(
        cfg, is_train=False, transforms=model.transforms_test
    )

    model.classes_names = classes_names
    task_nb = len(eval_dataset)
    with open(cfg.log_path, 'w+') as f: 
        pass

    class_ids_per_task = list(get_class_ids_per_task(cfg))
    ordered_class_names = []
    for id in range(task_nb):
        ordered_class_names += get_class_names(classes_names, class_ids_per_task[id])


    evaluator = Classification(cfg)
    acc_list = []
    loss_avg_list = []
    perf_dict = {}
    metric_logger = Logger(list_subsets=["test"])         

    for task_id, _ in enumerate(train_dataset):
        perf_dict[task_id] = {}
        # breakpoint()
        logging.info(f"Evaluation for task {task_id} has started.")

        model.adaptation(task_id, cfg, train_dataset, train_classes_names)  # task id 已经传入model
        loss_avg_list.append(model.loss_avg)
        
        for i in range(task_nb):
            image_f_list = []
            text_f_list = []
            cap_id_list = []
            impath_list = []
            eval_loader = DataLoader(eval_dataset[i:i + 1], batch_size=64)
            # breakpoint()
            for inputs, targets, texts, task_ids, img_id in tqdm(eval_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                texts = list(texts)
                image_f, text_f, logit_scale = model(inputs, texts, 0)
                cap_id_list.extend(img_id)
                impath_list.extend(img_id)
                image_f_list.append(image_f)
                text_f_list.append(text_f)
            image_features = torch.cat(image_f_list,dim=0)
            text_features = torch.cat(text_f_list,dim=0)
            output = logit_scale * image_features @ text_features.t()
            evaluator.process(output, impath_list, cap_id_list, task_id+1)
            result = evaluator.evaluate(i)
            perf_dict[task_id].update(result)

        
        acc_list.append(round(avg_accuracy(perf_dict, task_id, task_nb), 2))
        with open(cfg.log_path, 'a+') as f:
            f.write(json.dumps({
                #'task': task_id,
                'acc_avg': round(avg_accuracy(perf_dict, task_id, task_nb), 2),
                'acc_t2v_r1': round(avg_accuracy(perf_dict, task_id, task_nb,metric_type='t2v_r1'), 2),          
                'acc_t2v_r5': round(avg_accuracy(perf_dict, task_id, task_nb,metric_type='t2v_r5'), 2),
                'acc_t2v_r10': round(avg_accuracy(perf_dict, task_id, task_nb,metric_type='t2v_r10'), 2),
                'acc_v2t_r1': round(avg_accuracy(perf_dict, task_id, task_nb,metric_type='v2t_r1'), 2),     
                'acc_v2t_r5': round(avg_accuracy(perf_dict, task_id, task_nb,metric_type='v2t_r5'), 2), 
                'acc_v2t_r10': round(avg_accuracy(perf_dict, task_id, task_nb,metric_type='v2t_r10'), 2),                           
                'loss': round(model.loss_avg, 4),
                'acc_per_task': [round(acc_t, 2) for acc_t in accuracy_per_task(perf_dict, task_id, task_nb)],
            }) + '\n')
            metric_logger.end_task()
        break
    # assert 1 == 2


    with open(cfg.log_path, 'a+') as f:
        f.write(json.dumps({
            'order':ordered_class_names
        }) + '\n')

        

if __name__ == "__main__":
        continual_clip()

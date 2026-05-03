import torch
from torch.utils.data import Sampler, Dataset
import numpy as np
import os
from torchvision import transforms
from PIL import Image
import random
import json
import torch
import torch.nn.functional as F
import random


class WeightedSequenceSampler(Sampler):
    def __init__(self, dataset, class_weights, cfg):
        """
        初始化采样器
        
        参数：
        - dataset: 数据集对象，必须包含标签信息
        - class_weights: 类别的加权列表/张量
        """
        self.dataset = dataset
        self.cfg = cfg
        if self.cfg.weighted:
            self.rate = 0.7
        else:
            self.rate = 1
        # 获取所有样本的标签
        self.labels = self.dataset.y.tolist()
        # 获取每个类别的样本索引
        class_num = len(class_weights)
        self.class_indices = {i: [] for i in range(class_num)}  # 初始化类别索引
        for idx, label in enumerate(self.labels):
            self.class_indices[label].append(idx)
        # 创建一个每个类别加权后的采样概率
        self.class_probabilities = np.array(class_weights)  # 转换为numpy数组以提高访问效率
        
        # 用于记录每个类别内的已选择样本索引，避免重复采样
        for indexs in self.class_indices.values():
            random.shuffle(indexs)
        self.class_sampled = {label: 0 for label in self.class_indices}   

    def __iter__(self):
        """
        每次迭代时进行加权类别采样以及顺序样本采样
        """  
        self.indices = []
        
        # 执行直到完成一个batch
        while len(self.indices)<len(self.dataset)*self.rate:
            # 根据加权概率选择一个类别
            selected_class = np.random.choice(
                list(self.class_indices.keys()), p=self.class_probabilities)
            
            # 获取当前类别的未被选择的样本索引
            start_index = self.class_sampled[selected_class]
            available_samples = self.class_indices[selected_class][start_index:]
            if len(available_samples) == 0:
                self.class_sampled[selected_class] = 0
                available_samples = self.class_indices[selected_class]

            # 从可用的样本中选择一个
            sample_idx = available_samples[0]  # 顺序采样
            self.indices.append(sample_idx)
            
            # 更新已采样的样本集合
            self.class_sampled[selected_class] += 1
            
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

# 自定义数据集
class MSCOCO36(Dataset):
    def __init__(self, datapath, transfomaions, datafile):
        self.x, self.y, self.caption, self.imgid = [], [], [], []
        self.trsf = transfomaions
        self._to_tensor = transforms.ToTensor()
        with open(datafile, "r") as f:
            for line in f:
                split_line = line.split(" ", 1)
                imgid = split_line[0].strip().split('#')[0]
                loc = imgid.split('_')[1]
                path = imgid +'.jpg'
                lang_id = split_line[0].strip().split('#')[2]
                cap = split_line[1].strip()
                self.imgid.append(imgid)
                self.x.append(os.path.join(datapath, loc, path))
                self.y.append(int(lang_id))
                self.caption.append(cap)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.caption = np.array(self.caption)
    
    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        x, y, caption, imgid = self.x[idx], self.y[idx], self.caption[idx], self.imgid[idx]
        caption_target, caption_source = caption.split('#')
        x = Image.open(x).convert("RGB")
        if self.trsf is not None:
            x = self.trsf(x)
        if not isinstance(x, torch.Tensor):
            x = self._to_tensor(x)
        return x, y, caption_target, caption_source, imgid


class MSCOCO36_MONO(Dataset):
    def __init__(self, datapath, transfomaions, datafile):
        self.x, self.y, self.caption, self.imgid = [], [], [], []
        self.trsf = transfomaions
        self._to_tensor = transforms.ToTensor()
        with open(datafile, "r") as f:
            for line in f:
                split_line = line.split(" ", 1)
                imgid = split_line[0].strip().split('#')[0]
                loc = imgid.split('_')[1]
                path = imgid +'.jpg'
                lang_id = split_line[0].strip().split('#')[2]
                cap = split_line[1].strip()
                self.imgid.append(imgid)
                self.x.append(os.path.join(datapath, loc, path))
                self.y.append(int(lang_id))
                self.caption.append(cap)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.caption = np.array(self.caption)
    
    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        x, y, caption, imgid = self.x[idx], self.y[idx], self.caption[idx], self.imgid[idx]
        x = Image.open(x).convert("RGB")
        if self.trsf is not None:
            x = self.trsf(x)
        if not isinstance(x, torch.Tensor):
            x = self._to_tensor(x)
        return x, y, caption, imgid


def token_overlap_for_weights(chosen_id, t):
    lang = [
    "Arabic (阿拉伯语)", "Bengali (孟加拉语)", "Czech (捷克语)", "Danish (丹麦语)", 
    "German (德语)", "Greek (希腊语)", "English (英语)", "Spanish (西班牙语)", 
    "Persian (波斯语)", "Finnish (芬兰语)", "Filipino (菲律宾语)", "French (法语)", 
    "Hindi (印地语)", "Croatian (克罗地亚语)", "Hungarian (匈牙利语)", "Indonesian (印尼语)", 
    "Italian (意大利语)", "Hebrew (希伯来语)", "Japanese (日语)", "Korean (韩语)", 
    "Maori (毛利语)", "Dutch (荷兰语)", "Norwegian (挪威语)", "Polish (波兰语)", 
    "Portuguese (葡萄牙语)", "Quechua (克丘亚语)", "Romanian (罗马尼亚语)", "Russian (俄语)", 
    "Swedish (瑞典语)", "Swahili (斯瓦希里语)", "Telugu (泰卢固语)", "Thai (泰语)", 
    "Turkish (土耳其语)", "Ukrainian (乌克兰语)", "Vietnamese (越南语)", "Chinese (中文)"
]
    # 相对路径：从 pcl/continual_clip/ 到 pcl/experiments/class/analysis/
    token_overlap_file = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'class', 'analysis', 'tokens_overlap_MSCOCO36.json')
    with open(token_overlap_file, 'r', encoding='utf-8') as file:
        token_overlaps = json.load(file)
    chosen_class = [lang[id] for id in chosen_id]
    token_overlap_t = [token_overlaps['English (英语)'][name] for name in chosen_class if name != 'English (英语)']
    token_overlap = 0.01*torch.tensor(token_overlap_t)
    probabilities = F.softmax(-token_overlap/t, dim=0)
    print(probabilities)
    return probabilities

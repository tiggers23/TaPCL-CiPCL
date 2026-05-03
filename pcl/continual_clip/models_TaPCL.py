#t-shairg-b32-prob
from omegaconf import DictConfig
from tqdm import tqdm
import clip.clip as clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .Sampler import WeightedSequenceSampler, token_overlap_for_weights
from .utils import get_class_ids_per_task
from . import utils
from .loss import CrossEn
import os


def bulid_model(cfg):
    device = cfg.device
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_path = os.environ.get(
        'OPENCLIP_MODEL_PATH',
        os.path.join(project_root, 'Pretrained_models', 'OpenCLIP-ViT-B-32-xlm-roberta-base', 'pytorch_model.bin')
    )
    try:
        # loading JIT archive
        model_all = torch.jit.load(model_path, map_location=device).eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location=device)
    model, missing_keys  = clip.build_model(state_dict or model_all.state_dict(), cfg)
    model = model.to(device)    

    return model.float(), missing_keys


class ClassIncremental(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.device = device
        self.classes_names = None
        self.model, self.missing_keys = bulid_model(cfg)
        self.transforms_train = clip._transform(self.model.visual.input_resolution, is_train=True)
        self.transforms_test = clip._transform(self.model.visual.input_resolution, is_train=False)
        self.ref_model = None
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.nceloss = CrossEn(cfg)
        self.step = 0

        #self.dynamic_dataset = DynamicDataset(cfg)

    def forward(self, image = None, texts = None, taskid = 0, zeroshot=False):
        with torch.no_grad():
            image_feature, text_feature, logit_scale = self.model(image, texts, taskid, is_train=False, zeroshot=zeroshot)
            return image_feature, text_feature, logit_scale

    def adaptation(self, task_id, cfg, train_dataset, train_classes_names):
        if cfg.method != "zeroshot":
            self.train(task_id, cfg, train_dataset, train_classes_names)

    def train(self, task_id, cfg, train_dataset, train_classes_names):
        chosen_id = [5,6,7,11,16,23,24,28,33,35]
        probabilities = token_overlap_for_weights(chosen_id, cfg.T)
        sampler = WeightedSequenceSampler(train_dataset, probabilities, cfg)

        ### laoding dataset
        train_loader = DataLoader(train_dataset,
                                  batch_size=cfg.batch_size,
                                  sampler=sampler,
                                   num_workers=8)
        train_iter = iter(train_loader)  # 获取每个step的数据集

        EPOCH = cfg.epoch
        num_batches = len(train_loader)
        total_iterations = EPOCH * num_batches

        # 冻结参数
        for k, v in self.model.named_parameters():  # 冻结其他参数
            if "adaptmlp" not in k:
                v.requires_grad = False

        params = [
            v for k, v in self.model.named_parameters() if "adaptmlp" in k
        ]

        # optimizer
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = utils.cosine_lr(
            optimizer, cfg.lr, 30, total_iterations
        )

        # move model to device
        self.model = self.model.to(self.device)
        
        # start training
        self.model.train()
        loop = tqdm(range(total_iterations))
        loss_total = 0
        for iteration in loop:
            scheduler(iteration)
            try:
                inputs, _, texts_target, texts_source, _ = next(train_iter)
            except:
                train_iter = iter(train_loader)
                inputs, _, texts_target, texts_source, _ = next(train_iter)

            texts_target, texts_source = list(texts_target), list(texts_source)
            inputs = inputs.to(self.device)    
            logits_target = self.model(inputs, texts_target, 0, is_train=True)
            logits_source = self.model(inputs, texts_source, 0, is_train=True)
            loss1 = (self.nceloss(logits_target)+self.nceloss(logits_target.t()))/2
            loss2 = (self.nceloss(logits_source)+self.nceloss(logits_source.t()))/2
            loss = 0.8*loss1 + 0.2*loss2

            loss_total += loss.item()
            self.loss_avg = loss_total/(iteration+1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss = loss.item(), loss_avg = self.loss_avg)
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()       
        self.model.eval()


class DomainIncremental(nn.Module):
    pass


class TaskAgnostic(nn.Module):
    pass


def load_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    r"""Load a CLIP model in different continual scenarios.

    Arguments:
        cfg (DictConfig): Experiment configurations.
        device (torch.device): Device to train (or) evaluate the model on.

    Returns:
        nn.Module: Return scenario specific CLIP model.
    """
    if cfg.scenario == "class":
        return ClassIncremental(cfg, device)
    elif cfg.scenario == "domain":
        return DomainIncremental(cfg, device)
    elif cfg.scenario == "task-aganostic":
        return TaskAgnostic(cfg, device)
    else:
        raise ValueError(f"""
            `{cfg.scenarios}` is not a valid scenario, 
            Please choose from ['class', "domain', 'task-agnostic']
        """)

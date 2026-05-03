#t-sharing-b32-mono
from omegaconf import DictConfig
from tqdm import tqdm
import clip.clip as clip
from clip.model_translate import build_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils import get_class_ids_per_task
from . import utils
from .loss import CrossEn
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
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
    model, missing_keys  = build_model(state_dict or model_all.state_dict())
    model = model.to(device)   

    return model.float(), missing_keys


class ClassIncremental(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.device = device
        self.classes_names = None
        self.model, self.missing_keys = bulid_model(cfg)

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        translate_model_path = os.environ.get('M2M100_MODEL_PATH', os.path.join(project_root, 'Pretrained_models', 'facebookm2m100_418M'))

        self.translate_model = M2M100ForConditionalGeneration.from_pretrained(translate_model_path)
        self.translate_tokenizer = M2M100Tokenizer.from_pretrained(translate_model_path) 
        self.translate_model.to(cfg.device)

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
        ### laoding dataset
        train_loader = DataLoader(train_dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=False, num_workers=8)
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

        text_prompt = []
        img_id = []

        for iteration in loop:
            scheduler(iteration)
            try:
                _, _, texts, imgid = next(train_iter)
            except:
                train_iter = iter(train_loader)
                _, _, texts, imgid = next(train_iter)

            texts = list(texts)

            logits = self.model(texts, 0, self.translate_model, self.translate_tokenizer , is_train=True)
            text_prompt.extend(logits)
            img_id.extend(imgid)
    
        language = ['el', 'es', 'fr', 'it', 'pl', 'pt', 'sv', 'uk', 'zh']

        file_new = open('/DATA/MSCOCO36/train_CiPCL.txt', 'w', encoding='utf-8')  
        for n, id in enumerate(img_id):
            for i, langid in enumerate(language):
                text = id + '#' + langid + '#' + f'{i}' + ' ' + text_prompt[n][i]
                file_new.write(text+'\n')
        file_new.close() 

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

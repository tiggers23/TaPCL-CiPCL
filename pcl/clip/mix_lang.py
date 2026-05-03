import torch
import re 
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import random
import torch.nn.functional as F
def prompt(raw_text, alt_tokens, langid, model, tokenizer):
    prompt_texts = []
    lang = ["Greek", "Spanish", "French", "Italian", "Polish", "Portuguese",
    "Swedish", "Ukrainian", "Chinese"
]
    language = ['el', 'es', 'fr', 'it', 'pl', 'pt', 'sv', 'uk', 'zh']

    batch_size = len(raw_text)  
    for idx in range(batch_size):
        text = []
        token_s = alt_tokens[idx]
        token_s = token_s.replace("▁", "")
        token_s = token_s.replace(".", "")
        token_s = token_s.replace(",", "")
        token_s = token_s.replace(";", "")
        token_s = token_s.replace(":", "")
        encoded = tokenizer(token_s, return_tensors="pt").to('cuda:3')
        for i, name in enumerate(lang):
            generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(f"{language[i]}"))
            token_t = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)        
            tokenwithprompt = f"({token_t[0]} in {name})" 
            replace_token = token_s + tokenwithprompt       
            prompt_text = re.sub(r'\b' + re.escape(token_s) + r'\b', replace_token, raw_text[idx])
            text.append(prompt_text)    
        prompt_texts.append(text)
        
    return prompt_texts



def get_token(tokens_s, attentions):
    attentions = attentions.detach()
    batch_size = len(tokens_s)
    headsum_attention = attentions.sum(dim=1)
    cls_attention = headsum_attention[:, 0, :]
    cls_attention[:, 0] = float('-inf')
    for m in range(batch_size):
        for n in range(cls_attention.shape[1]):
            if tokens_s[m][n] == '</s>' or tokens_s[m][n] == '<pad>':
                cls_attention[m,n] = float('-inf')
    max_attention_idx = cls_attention.argmax(dim=1)
   
    alt_tokens = []    
    for i in range(batch_size):
        sample_s = tokens_s[i]
        idx_s = max_attention_idx[i]
        start_token_s = sample_s[idx_s]
        if sample_s[idx_s].startswith('▁'):
            for token in sample_s[idx_s+1:]:
                if token.startswith('▁') or token == '</s>':
                    alt_token_s = start_token_s
                    break
                else:
                    start_token_s += token
        else:
            for token in sample_s[idx_s+1:]:
                if token.startswith('▁') or token == '</s>':
                    break
                else:
                    start_token_s += token
            for token in sample_s[idx_s-1::-1]:
                if token.startswith('▁'):
                    alt_token_s = token + start_token_s
                    break
                else:
                    start_token_s = token + start_token_s
        alt_tokens.append(alt_token_s)

    return alt_tokens

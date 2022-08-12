from transformers import AutoTokenizer
from utils.config import *

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_LINK, from_pt=True)


def tokenize_text(text, max_len):
    tokenized_text = tokenizer(text, return_tensors='tf')
    ids = tokenized_text.data['input_ids'].numpy().flatten().tolist()
    attention_mask = tokenized_text.data['attention_mask'].numpy().flatten().tolist()
    padding_len = max_len - len(ids)
    if padding_len > 0:
        ids = ids + ([0] * padding_len)
        attention_mask = attention_mask + ([0] * padding_len)
    elif padding_len < 0:
        ids = ids[0:max_len]
        ids[max_len - 1] = END_TOKEN_ID
        attention_mask = attention_mask[0:max_len]
    return ids, attention_mask

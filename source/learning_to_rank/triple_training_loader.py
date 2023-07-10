import gzip
import tensorflow as tf
import numpy as np
import re
from transformers import AutoTokenizer
from utils.utils import normalize
from utils.config import *
from entity.entity_vec import get_query_vector, get_passage_vector, extract_entities, get_knrms


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


def tokenize_input(training_triple_file_path, training_size):
    query_ids_list = list()
    query_mask_list = list()
    query_vector_list = list()
    pos_passage_ids_list = list()
    pos_passage_mask_list = list()
    pos_passage_vector_list = list()
    pos_knrm_list = list()
    neg_passage_ids_list = list()
    neg_passage_mask_list = list()
    neg_passage_vector_list = list()
    neg_knrm_list = list()
    count = 0
    with gzip.open(training_triple_file_path, "rt") as training_triple_file:
        for triple in training_triple_file:
            triple = triple.strip()
            parts = triple.split("\t")
            query = parts[0]
            pos_passage = parts[1]
            neg_passage = parts[2]
            query_entities = extract_entities(query)
            query_vector = get_query_vector(query)
            pos_passage_vector = get_passage_vector(query, pos_passage)
            neg_passage_vector = get_passage_vector(query, neg_passage)
            query = re.sub("<.*?>", "", query)
            pos_passage = re.sub("<.*?>", "", pos_passage)
            neg_passage = re.sub("<.*?>", "", neg_passage)
            if USE_KNRM:
                pos_knrm = get_knrms(query_entities, pos_passage)
                neg_knrm = get_knrms(query_entities, neg_passage)
            pos_score = np.inner(normalize(query_vector), normalize(pos_passage_vector))
            neg_score = np.inner(normalize(query_vector), normalize(neg_passage_vector))
            diff = pos_score - neg_score
            if diff < POS_NEG_GAP_LIMIT:
                query_vector = np.zeros(ENTITY_DIMENSION)
                pos_passage_vector = np.zeros(ENTITY_DIMENSION)
                neg_passage_vector = np.zeros(ENTITY_DIMENSION)
                if USE_KNRM:
                    pos_knrm = [0.0] * KERNEL_DIMENSION_INPUT
                    neg_knrm = [0.0] * KERNEL_DIMENSION_INPUT
            query_ids, query_mask = tokenize_text(query, MAX_QUERY_LEN)
            pos_passage_ids, pos_passage_mask = tokenize_text(pos_passage, MAX_PASSAGE_LEN)
            neg_passage_ids, neg_passage_mask = tokenize_text(neg_passage, MAX_PASSAGE_LEN)
            count += 1
            if count % 100000 == 0:
                print(count)
            query_ids_list.append(query_ids)
            query_mask_list.append(query_mask)
            query_vector_list.append(query_vector)
            pos_passage_ids_list.append(pos_passage_ids)
            pos_passage_mask_list.append(pos_passage_mask)
            pos_passage_vector_list.append(pos_passage_vector)
            neg_passage_ids_list.append(neg_passage_ids)
            neg_passage_mask_list.append(neg_passage_mask)
            neg_passage_vector_list.append(neg_passage_vector)
            if USE_KNRM:
                pos_knrm_list.append(pos_knrm)
                neg_knrm_list.append(neg_knrm)
            if count == training_size:
                break
    query_ids_list = tf.reshape(query_ids_list, [training_size, MAX_QUERY_LEN])
    query_mask_list = tf.reshape(query_mask_list, [training_size, MAX_QUERY_LEN])
    query_vector_list = tf.reshape(query_vector_list, [training_size, ENTITY_DIMENSION])
    pos_passage_ids_list = tf.reshape(pos_passage_ids_list, [training_size, MAX_PASSAGE_LEN])
    pos_passage_mask_list = tf.reshape(pos_passage_mask_list, [training_size, MAX_PASSAGE_LEN])
    pos_passage_vector_list = tf.reshape(pos_passage_vector_list, [training_size, ENTITY_DIMENSION])
    neg_passage_ids_list = tf.reshape(neg_passage_ids_list, [training_size, MAX_PASSAGE_LEN])
    neg_passage_mask_list = tf.reshape(neg_passage_mask_list, [training_size, MAX_PASSAGE_LEN])
    neg_passage_vector_list = tf.reshape(neg_passage_vector_list, [training_size, ENTITY_DIMENSION])
    if USE_KNRM:
        pos_knrm_list = tf.reshape(pos_knrm_list, [training_size, KERNEL_DIMENSION_INPUT])
        neg_knrm_list = tf.reshape(neg_knrm_list, [training_size, KERNEL_DIMENSION_INPUT])
        return query_ids_list, query_mask_list, query_vector_list, \
               pos_passage_ids_list, pos_passage_mask_list, pos_passage_vector_list, pos_knrm_list, \
               neg_passage_ids_list, neg_passage_mask_list, neg_passage_vector_list, neg_knrm_list
    else:
        return query_ids_list, query_mask_list, query_vector_list, \
               pos_passage_ids_list, pos_passage_mask_list, pos_passage_vector_list, \
               neg_passage_ids_list, neg_passage_mask_list, neg_passage_vector_list

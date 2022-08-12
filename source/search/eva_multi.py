import numpy as np
import tensorflow as tf
from model.tokenizer import tokenize_text
from model.eva_model import build_ranking_model, strategy
from entity.entity_vec import extract_entities, get_average_vector
from search.query import Query
from utils.config import *
import datetime
import faiss
import sys
import re


def load_model(model_checkpoint):
    with strategy.scope():
        ranking_model = build_ranking_model(MS_MARCO_TRAINING_SIZE)
        ranking_model.load_weights(model_checkpoint)
        return tf.keras.Model(inputs=ranking_model.input,
                              outputs=ranking_model.get_layer("tf.concat").output)


index = faiss.read_index(INDEX_FILE)
index.nprobe = CLUSTER_LOOK_UP_NUMBER
total_query_vector_model = load_model(EVA_MODEL_CHECKPOINT)


def search_from_query_file(query_file_name):
    queries = load_queries(query_file_name)
    search_and_print(queries)


def load_queries(query_file_name):
    queries = list()
    query_file = open(query_file_name, "rt")
    while True:
        line = query_file.readline()
        if line == "":
            break
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        query_id = parts[0]
        annotated_query = parts[1]
        queries.append(Query(query_id, annotated_query))
    return queries


def search_and_print(queries):
    print(MULTI_TITLE)
    start_parse_time = datetime.datetime.now()
    distances, entity_cluster_indices = search(queries, TOP_RANKED_CLUSTER_RESULTS)
    time_diff = (datetime.datetime.now() - start_parse_time)
    for j in range(len(queries)):
        print(queries[j].query_id + "\t" + queries[j].query)
        for i in range(len(entity_cluster_indices[j])):
            entity_cluster_index = entity_cluster_indices[j][i]
            score = distances[j][i]
            print(str(score) + "\t" + str(entity_cluster_index))
        print(SPLIT_LINE)
    print(f"Total search time: {time_diff.total_seconds()}")


def search(queries, top_ranked_clusters):
    query_ids_list = list()
    query_mask_list = list()
    query_vector_list = list()
    query_kps_list = list()
    passage_ids_list = list()
    passage_mask_list = list()
    passage_vector_list = list()
    for query in queries:
        annotated_query = query.query
        query_text = re.sub("<.*?>", "", annotated_query)
        query_entities = extract_entities(annotated_query)
        query_representation = get_average_vector(query_entities)
        query_ids, query_mask = tokenize_text(query_text, MAX_QUERY_LEN)
        dummy_passage_ids, dummy_passage_mask = tokenize_text("", MAX_PASSAGE_LEN)
        query_ids_list.append(query_ids)
        query_mask_list.append(query_mask)
        query_vector_list.append(query_representation)
        passage_ids_list.append(dummy_passage_ids)
        passage_mask_list.append(dummy_passage_mask)
        passage_vector_list.append(np.zeros(ENTITY_DIMENSION))
        if USE_KNRM:
            query_kps = 0.0
            if len(query_entities) > 0:
                query_kps = 1.0
            query_kps_list.append(query_kps)
    size = len(query_ids_list)
    query_ids_list = tf.reshape(query_ids_list, [size, MAX_QUERY_LEN])
    query_mask_list = tf.reshape(query_mask_list, [size, MAX_QUERY_LEN])
    passage_ids_list = tf.reshape(passage_ids_list, [size, MAX_PASSAGE_LEN])
    passage_mask_list = tf.reshape(passage_mask_list, [size, MAX_PASSAGE_LEN])
    query_vector_list = tf.reshape(query_vector_list, [size, ENTITY_DIMENSION])
    passage_vector_list = tf.reshape(passage_vector_list, [size, ENTITY_DIMENSION])
    if USE_KNRM:
        query_kps_list = tf.reshape(query_kps_list, [size, KERNEL_DIMENSION_OUTPUT])
        passage_knrm_list = tf.zeros([size, KERNEL_DIMENSION_INPUT], tf.float32)
        inputs = [query_ids_list, query_mask_list, query_vector_list,
                  passage_ids_list, passage_mask_list, passage_vector_list, passage_knrm_list,
                  passage_ids_list, passage_mask_list, passage_vector_list, passage_knrm_list]
        query_total_vector_list = total_query_vector_model(inputs)
        query_total_vector_list = tf.concat([query_total_vector_list, query_kps_list], 1)
    else:
        inputs = [query_ids_list, query_mask_list, query_vector_list,
                  passage_ids_list, passage_mask_list, passage_vector_list,
                  passage_ids_list, passage_mask_list, passage_vector_list]
        query_total_vector_list = total_query_vector_model(inputs)
    query_total_vector_list = query_total_vector_list.numpy()
    return index.search(query_total_vector_list, top_ranked_clusters)


if __name__ == "__main__":
    query_file_name = sys.argv[1]
    search_from_query_file(query_file_name)

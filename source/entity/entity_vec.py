from utils.config import *
from entity.wikivec_loader import get_entity_vector, get_entity_similarity, get_similarity
import numpy as np
import re

STOP_WORDS = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours	ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "s", "t"]
KERNELS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
SIGMAS = [0.1, 0.1, 0.1, 0.1, 0.1, 1e-3]


def extract_entities(text):
    begin = 0
    entities = set()
    while 0 <= begin < len(text):
        begin = text.find(WIKI_PATTERN, begin)
        if begin < 0:
            break
        entity_end = text.find(ENTITY_END_PATTERN, begin)
        entity = text[begin + len(WIKI_PATTERN):entity_end]
        begin = text.find(TAG_END_PATTERN, entity_end)
        entities.add(entity)
    return entities


def get_average_vector(entities):
    size = len(entities)
    sum_vector = np.zeros(ENTITY_DIMENSION)
    if size == 0:
        return sum_vector
    for entity in entities:
        sum_vector = np.add(sum_vector, get_entity_vector(entity))
    return sum_vector / size

def get_query_vector(query):
    query_entities = extract_entities(query)
    return get_average_vector(query_entities)


def get_passage_vector(query, passage):
    query_entities = extract_entities(query)
    passage_entities = extract_entities(passage)
    selected_entities = set()
    for query_entity in query_entities:
        max_similarity = 0
        max_entity = None
        for passage_entity in passage_entities:
            similarity = get_entity_similarity(query_entity, passage_entity)
            if max_similarity < similarity:
                max_similarity = similarity
                max_entity = passage_entity
        if max_similarity > 0.8:
            selected_entities.add(max_entity)
        else:
            return np.zeros(ENTITY_DIMENSION)
    return get_average_vector(selected_entities)


def extract_terms(text):
    text = text.strip().lower()
    text = re.sub(",|;|\\.|!|'|\"|-|_|:|\\(|\\)|\\?", " ", text)
    text = re.sub(" +", " ", text)
    terms = text.split()
    return [term for term in terms if term not in STOP_WORDS]


def extract_terms_and_entities(passage):
    begin = 0
    result = list()
    while 0 <= begin < len(passage):
        tag_pos = passage.find(WIKI_PATTERN, begin)
        if tag_pos < 0:
            result.extend(extract_terms(passage[begin:]))
            break
        result.extend(extract_terms(passage[begin:tag_pos]))
        entity_end = passage.find(ENTITY_END_PATTERN, tag_pos)
        entity = passage[tag_pos + len(WIKI_PATTERN):entity_end]
        begin = passage.find(TAG_END_PATTERN, entity_end) + len(TAG_END_PATTERN)
        result.append("ENTITY/" + entity)
    return result


def get_knrms(entities, passage):
    terms_and_entities = extract_terms_and_entities(passage)
    entity_list = list(entities)
    similarities = list()
    for i in range(len(entity_list)):
        current_similarities = list()
        for j in range(len(terms_and_entities)):
            similarity = get_similarity("ENTITY/" + entity_list[i], terms_and_entities[j])
            current_similarities.append(similarity)
        similarities.append(current_similarities)
    knrms = [0.0] * len(KERNELS)
    for k in range(len(KERNELS)):
        for i in range(len(entity_list)):
            rbf_kernel = 0.0
            for j in range(len(terms_and_entities)):
                x = -1.0 * (similarities[i][j] - KERNELS[k]) * (similarities[i][j] - KERNELS[k]) / 2.0 / SIGMAS[k] / SIGMAS[k]
                rbf_kernel += np.exp(x)
            knrms[k] += np.log(rbf_kernel + 1e-6)
    knrms.append(1.0)
    return knrms

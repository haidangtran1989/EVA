from utils.config import *
import pytrec_eval
from entity.entity_vec import extract_entities
from entity.wikivec_loader import cluster_match
from entity.entity_cluster_loader import get_entity_cluster
from evaluation.judgement_loader import get_judgements
import sys


def print_effectiveness(title, result, metric, id_to_query, output_file):
    output_file.write(title + "\n")
    sum_effectiveness = 0.0
    count = 0
    for query_id, effectiveness in result.items():
        output_file.write(query_id + "\t" + id_to_query[query_id] + "\t" + str(effectiveness[metric]) + "\n")
        sum_effectiveness += effectiveness[metric]
        count += 1
    output_file.write(str(count) + "\n")
    output_file.write(str(sum_effectiveness / count) + "\n")
    output_file.write(SPLIT_LINE + "\n")


def match_entities(group_entities, query_entities):
    if len(query_entities) == 0:
        return len(group_entities) == 0
    if len(group_entities) == 0:
        return len(query_entities) == 0
    if len(query_entities) == 1:
        return cluster_match(group_entities, query_entities, 0.9) and cluster_match(query_entities, group_entities, 0.9)
    elif cluster_match(group_entities, query_entities, 0.55) and cluster_match(query_entities, group_entities, 0.55):
        return True
    return False


def build_run(search_result_file_path, output_file, query_id_to_judgements, top_ranked_results_for_evaluation, display):
    search_result_file = open(search_result_file_path, "rt")
    title = search_result_file.readline().strip()
    run = dict()
    id_to_query = dict()
    while True:
        line = search_result_file.readline()
        if line == "":
            break
        line = line.strip()
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        query_id = parts[0]
        query = parts[1]
        id_to_query[query_id] = query
        actual_score = dict()
        ranked_passage_ids = set()
        query_entities = extract_entities(query)
        if display:
            output_file.write(line + "\n")
        while True:
            search_result = search_result_file.readline()
            if search_result == "":
                break
            search_result = search_result.strip()
            if search_result == SPLIT_LINE:
                if display:
                    output_file.write(SPLIT_LINE + "\n")
                break
            if len(actual_score) >= top_ranked_results_for_evaluation:
                continue
            parts = search_result.split("\t")
            score = float(parts[0])
            if title == MULTI_TITLE:
                entity_cluster = get_entity_cluster(parts[1])
                passage_id = entity_cluster.passage_id
                group_entities = entity_cluster.entity_cluster
            else:
                passage_id = parts[1]
                group_entities = None
            if passage_id in ranked_passage_ids:
                continue
            relevance = -1
            if passage_id in query_id_to_judgements[query_id]:
                relevance = query_id_to_judgements[query_id][passage_id]
            if title != "EVA Multi":
                if not match_entities(group_entities, query_entities):
                    continue
            ranked_passage_ids.add(passage_id)
            if len(actual_score) < TOP_RANKED_RESULTS_FOR_DISPLAY and display:
                output_file.write(str(relevance) + "\t" + str(score) + "\t" + passage_id + "\n")
            if len(actual_score) < top_ranked_results_for_evaluation:
                actual_score[passage_id] = score
        run[query_id] = actual_score
    return id_to_query, run


def evaluate(judgement_file_path, search_result_file_path, output_file_path, min_relevance_for_mrr_map):
    output_file = open(output_file_path, "wt")
    query_id_to_judgements = get_judgements(judgement_file_path)

    id_to_query, run = build_run(search_result_file_path, output_file, query_id_to_judgements,
                                 TOP_RANKED_RESULTS_FOR_DISPLAY, True)
    ndcg_evaluator = pytrec_eval.RelevanceEvaluator(query_id_to_judgements, {"ndcg_cut"})
    print_effectiveness("nDCG measure", ndcg_evaluator.evaluate(run), "ndcg_cut_10", id_to_query, output_file)
    mrr_evaluator = pytrec_eval.RelevanceEvaluator(query_id_to_judgements, {"recip_rank"},
                                                   relevance_level=min_relevance_for_mrr_map)
    print_effectiveness("MRR measure", mrr_evaluator.evaluate(run), "recip_rank", id_to_query, output_file)

    id_to_query, run = build_run(search_result_file_path, output_file, query_id_to_judgements,
                                 TOP_RANKED_RESULTS_FOR_EVALUATION, False)
    map_evaluator = pytrec_eval.RelevanceEvaluator(query_id_to_judgements, {"map"},
                                                   relevance_level=min_relevance_for_mrr_map)
    print_effectiveness("MAP measure", map_evaluator.evaluate(run), "map", id_to_query, output_file)

    output_file.close()


if __name__ == "__main__":
    judgement_file_path = sys.argv[1]
    search_result_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    min_relevance_for_mrr_map = int(sys.argv[4])
    evaluate(judgement_file_path, search_result_file_path, output_file_path, min_relevance_for_mrr_map)

def get_judgements(file_name):
    judgement_file = open(file_name, "rt")
    query_id_to_judgements = dict()
    for line in judgement_file:
        parts = line.strip().split(" ")
        if len(parts) != 4:
            continue
        query_id = parts[0]
        passage_id = parts[2]
        relevance = int(parts[3])
        query_passage_ids_to_judgements = dict()
        if query_id in query_id_to_judgements:
            query_passage_ids_to_judgements = query_id_to_judgements[query_id]
        query_passage_ids_to_judgements[passage_id] = relevance
        query_id_to_judgements[query_id] = query_passage_ids_to_judgements
    return query_id_to_judgements

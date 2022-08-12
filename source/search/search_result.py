class SearchResult:
    def __init__(self, score, passage_id, passage, entity_cluster):
        self.score = score
        self.passage_id = passage_id
        self.passage = passage
        self.entity_cluster = entity_cluster

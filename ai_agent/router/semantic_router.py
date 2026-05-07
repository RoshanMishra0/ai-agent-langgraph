from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from router.route_examples import route_examples

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

route_embeddings = {}

for route, examples in route_examples.items():
    route_embeddings[route] = embedding_model.encode(examples)


def get_routes(query):
    query_embedding = embedding_model.encode([query])

    selected_routes = []

    for route, embeddings in route_embeddings.items():
        scores = cosine_similarity(
            query_embedding,
            embeddings
        )

        if np.max(scores) > 0.6:
            selected_routes.append(route)

    if not selected_routes:
        selected_routes = ["general"]

    return selected_routes

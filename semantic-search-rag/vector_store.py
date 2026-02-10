import os
import json
import numpy as np

class VectorStore:
    def __init__(self, name):
        self.filename = f"{name}.json"
        self.vectors = []

        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                self.vectors = json.load(f)

    def add(self, vector, metadata):
        self.vectors.append({
            "vector": vector.tolist(),
            "metadata": metadata
        })

    def save(self):
        with open(self.filename, "w") as f:
            json.dump(self.vectors, f)

    @staticmethod
    def load(name):
        return VectorStore(name)

    def search(self, query_vector, top_k=3):
        query = np.array(query_vector[0])
        scores = []

        for item in self.vectors:
            vec = np.array(item["vector"])
            score = np.dot(query, vec) / (
                np.linalg.norm(query) * np.linalg.norm(vec)
            )
            scores.append((score, item))

        scores.sort(reverse=True, key=lambda x: x[0])
        return [item for _, item in scores[:top_k]]

from dataclasses import dataclass
from typing import List, Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import build_item_document, build_user_document


@dataclass(frozen=True)
class Recommendation:
    rank: int
    item_id: str
    name: str
    score: float
    category: str
    frontend: str
    backend: str
    database: str
    deployment: str
    description: str
    fallback: bool = False


class TechStackRecommender:
    """Content-based recommender using TF-IDF and cosine similarity."""

    def __init__(self, catalog: pd.DataFrame):
        self.catalog = catalog.reset_index(drop=True).copy()
        self.catalog["content"] = self.catalog.apply(
            build_item_document,
            axis=1,
        )

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
        )

        self.item_matrix = self.vectorizer.fit_transform(
            self.catalog["content"]
        )

    def recommend(
        self,
        user_inputs: Sequence[str],
        top_n: int = 3,
    ) -> List[Recommendation]:
        """Score every item, sort descending, and return Top-N."""

        cleaned = [
            str(x).strip()
            for x in user_inputs
            if str(x).strip()
        ]

        if len(cleaned) < 3:
            raise ValueError(
                "At least three non-empty user inputs are required."
            )

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        user_document = build_user_document(cleaned)
        user_vector = self.vectorizer.transform([user_document])

        scores = cosine_similarity(
            user_vector,
            self.item_matrix,
        ).ravel()

        ranked = self.catalog.copy()
        ranked["score"] = scores

        cold_start = user_vector.nnz == 0

        if cold_start:
            ranked = ranked.sort_values(
                ["popularity", "id"],
                ascending=[False, True],
                kind="mergesort",
            )
        else:
            ranked = ranked.sort_values(
                ["score", "id"],
                ascending=[False, True],
                kind="mergesort",
            )

        ranked = ranked.head(min(top_n, len(ranked)))

        results = []
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            results.append(
                Recommendation(
                    rank=rank,
                    item_id=str(row["id"]),
                    name=str(row["name"]),
                    score=float(row["score"]),
                    category=str(row["category"]),
                    frontend=str(row["frontend"]),
                    backend=str(row["backend"]),
                    database=str(row["database"]),
                    deployment=str(row["deployment"]),
                    description=str(row["description"]),
                    fallback=cold_start,
                )
            )

        return results

    def vocabulary(self) -> List[str]:
        """Return the learned TF-IDF vocabulary."""

        return sorted(self.vectorizer.vocabulary_.keys())

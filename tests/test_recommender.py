import pandas as pd
import pytest

from app.recommender import TechStackRecommender


def make_catalog():
    return pd.DataFrame([
        {
            "id": "A",
            "name": "Python Analytics",
            "category": "data science",
            "frontend": "Streamlit",
            "backend": "Python",
            "database": "PostgreSQL",
            "deployment": "AWS",
            "tags": "python analytics pandas sql",
            "description": "analytics dashboard and machine learning",
            "popularity": 3,
        },
        {
            "id": "B",
            "name": "React Web",
            "category": "web development",
            "frontend": "React",
            "backend": "Node.js",
            "database": "PostgreSQL",
            "deployment": "Vercel",
            "tags": "react javascript typescript web",
            "description": "modern web application",
            "popularity": 2,
        },
        {
            "id": "C",
            "name": "Cloud Automation",
            "category": "automation",
            "frontend": "None",
            "backend": "Python",
            "database": "DynamoDB",
            "deployment": "AWS",
            "tags": "python automation cloud aws",
            "description": "cloud automation and serverless workflows",
            "popularity": 5,
        },
    ])


def test_requires_three_inputs():
    recommender = TechStackRecommender(make_catalog())
    with pytest.raises(ValueError):
        recommender.recommend(["python", "analytics"], top_n=3)


def test_content_based_ranking_prefers_matching_item():
    recommender = TechStackRecommender(make_catalog())
    results = recommender.recommend(
        ["data science", "analytics dashboard", "python sql"],
        top_n=3,
    )

    assert results[0].item_id == "A"
    assert 0.0 <= results[0].score <= 1.0


def test_top_n_filtering():
    recommender = TechStackRecommender(make_catalog())
    results = recommender.recommend(
        ["python", "cloud", "automation"],
        top_n=2,
    )

    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_cold_start_uses_global_popularity_fallback():
    recommender = TechStackRecommender(make_catalog())
    results = recommender.recommend(
        ["quantum", "underwater", "telepathy"],
        top_n=3,
    )

    assert len(results) == 3
    assert results[0].item_id == "C"
    assert all(result.fallback for result in results)

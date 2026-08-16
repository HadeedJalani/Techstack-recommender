<div align="center">

# 🧠 Tech Stack Recommender

### Content-Based Recommendation Engine for Technology Stack Selection

**TF-IDF · Cosine Similarity · Ranking · Top-N Filtering · Cold-Start Handling**

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Streamlit-Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Tests-4%2F4%20Passing-2EA44F?style=for-the-badge" alt="Tests" />
</p>

<p>
  <strong>Internship Capstone Project</strong><br />
  A practical recommendation system that converts user preferences into ranked technology-stack recommendations using transparent, explainable machine-learning techniques.
</p>

</div>

---

## ✨ Project Overview

**Tech Stack Recommender** is a content-based recommendation system designed to answer a practical engineering question:

> **“Given what I want to build and the technologies I am interested in, which technology stack should I choose?”**

The system compares a user's stated preferences against a catalog of technology stacks. It represents both sides in a shared **TF-IDF feature space**, calculates **cosine similarity**, ranks every candidate, and returns a concise **Top-N recommendation list**.

The implementation follows the capstone architecture shown in the project specification:

```text
INPUT → PROCESSING → SCORING → SORTING → FILTERING → OUTPUT
```

It deliberately uses **content-based filtering** rather than collaborative filtering, because the assignment focuses on matching users directly to item attributes without requiring a historical user-interaction dataset.

---

## 🎯 Objectives

The project is built around the following objectives:

- Capture explicit user preferences.
- Require a minimum of three meaningful inputs.
- Translate natural-language preferences into a numerical representation.
- Represent users and technology stacks in the same feature space.
- Use **TF-IDF** instead of simple binary 0/1 matching.
- Measure relevance using **cosine similarity**.
- Score the complete catalog.
- Sort candidates by relevance.
- Return a configurable **Top-N** list to reduce choice overload.
- Handle the **cold-start problem** with a practical fallback.
- Provide both a command-line implementation and a visual demonstration interface.

---

## 🏗️ System Architecture

```text
                         ┌─────────────────────────┐
                         │         USER            │
                         │                         │
                         │  Role / Domain          │
                         │  Project Requirements   │
                         │  Skills / Interests     │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │         INPUT           │
                         │   Validate 3+ inputs    │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │      PREPROCESSING      │
                         │                         │
                         │  Normalize text        │
                         │  Build user document   │
                         └────────────┬────────────┘
                                      │
                  ┌───────────────────┴───────────────────┐
                  │                                       │
                  ▼                                       ▼
       ┌──────────────────────┐                ┌──────────────────────┐
       │   USER PROFILE       │                │   ITEM CATALOG       │
       │                      │                │                      │
       │ User document        │                │ Tech-stack documents  │
       └──────────┬───────────┘                └──────────┬───────────┘
                  │                                       │
                  └───────────────────┬───────────────────┘
                                      ▼
                         ┌─────────────────────────┐
                         │         TF-IDF          │
                         │                         │
                         │ Shared vocabulary      │
                         │ Weighted feature space │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   COSINE SIMILARITY     │
                         │                         │
                         │ user vector ↔ item     │
                         │ vector                 │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │        SCORING          │
                         │  Score every candidate  │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │        SORTING          │
                         │      Score DESC        │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │       FILTERING         │
                         │         Top-N           │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │         OUTPUT          │
                         │ Ranked recommendations │
                         └─────────────────────────┘
```

---

## 🔬 Machine Learning Pipeline

### 1. Input

The application collects at least three user inputs:

| Input | Example |
|---|---|
| Target role / domain | `data science` |
| Project requirements | `analytics dashboard` |
| Skills / interests | `Python SQL cloud` |
| Optional interests | `machine learning` |

### 2. Preprocessing

Text is normalized so that different surface forms can be compared consistently. The system removes irrelevant punctuation, normalizes case, and builds a single user-profile document.

Each catalog item is converted into a content document using:

- Category
- Frontend
- Backend
- Database
- Deployment target
- Technology tags
- Description

### 3. TF-IDF Vectorization

Instead of treating every word as equally important, the system uses **Term Frequency–Inverse Document Frequency**.

```text
TF(t,d)  = count of term t in document d / total terms in d

IDF(t)   = log(Total documents / documents containing t)

TF-IDF   = TF × IDF
```

This gives more weight to descriptive terms and reduces the influence of generic words that appear throughout the catalog.

The same fitted vectorizer transforms both the catalog and user profile, ensuring a **shared vocabulary space**.

### 4. Similarity Scoring

The system uses cosine similarity:

```text
                 A · B
cosine(A,B) = ─────────────
              ||A|| ||B||
```

A score closer to **1.0** means stronger directional alignment between the user profile and the technology-stack content.

Cosine similarity is particularly appropriate here because recommendation quality should depend primarily on the **orientation of preferences**, rather than the absolute length of the text vectors.

### 5. Ranking

Every catalog item receives a similarity score. Candidates are then sorted in descending order:

```text
Candidate A → 0.91
Candidate B → 0.84
Candidate C → 0.77
Candidate D → 0.45
Candidate E → 0.32
```

### 6. Top-N Filtering

Only the highest-scoring candidates are returned. The default is **Top 3**, directly reflecting the project's choice-overload objective.

```text
0.91 ──┐
0.84 ──┤  ← Recommended
0.77 ──┘
0.45    ┐
0.32    ┘  ← Filtered out
```

---

## 🧊 Cold-Start Strategy

A recommendation system must still behave sensibly when little or no matching information exists.

### User Cold Start

If a new user's terms contain no vocabulary known by the catalog, cosine similarity cannot provide meaningful personalized scores.

The system detects this condition and activates a **global popularity fallback**:

```text
Unknown user profile
        ↓
No matching TF-IDF vocabulary
        ↓
Cold-start detected
        ↓
Popularity ranking
        ↓
Top-N recommendations
```

This implements the assignment's cold-start / popularity-fallback concept without introducing unnecessary collaborative-filtering infrastructure.

### Item Cold Start

A new technology stack does not require historical user interactions. Once its metadata is added to the catalog, its content can be represented and compared against user preferences.

This is a key benefit of content-based recommendation.

---

## 🧩 Why Content-Based Filtering?

The assignment explicitly focuses on **content-based filtering**.

### Collaborative Filtering

```text
User A ── bought ── Item X
User B ── bought ── Item X
                    ↓
              infer similarity
```

This requires historical user behavior.

### Content-Based Filtering

```text
User preferences
       ↓
Feature representation
       ↓
Compare against item attributes
       ↓
Recommend similar items
```

This project uses the second approach because it directly matches the available information and remains effective for new items.

---

## 📁 Project Structure

```text
Techstack-recommender/
│
├── app/
│   ├── __init__.py
│   ├── cli.py                  # Command-line application
│   ├── config.py               # Paths and configuration
│   ├── data_loader.py          # Dataset loading/validation
│   ├── preprocessing.py        # Text normalization and documents
│   ├── recommender.py          # TF-IDF + cosine recommendation engine
│   └── streamlit_app.py        # Visual demonstration UI
│
├── data/
│   └── tech_stacks.csv         # Technology-stack catalog
│
├── tests/
│   └── test_recommender.py     # Core behavior tests
│
├── .env.example                # Environment template
├── .gitignore                  # Git exclusions
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── run.py                      # Application entry point
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core application language |
| **pandas** | Catalog loading and tabular data handling |
| **scikit-learn** | TF-IDF vectorization and cosine similarity |
| **Streamlit** | Optional interactive demonstration UI |
| **pytest** | Automated testing |
| **CSV** | Lightweight, transparent catalog storage |

### Deliberately not used

The assignment does not require an LLM, RAG, vector database, authentication system, relational database, or external API. These technologies were intentionally excluded to keep the solution aligned with the specification and easy to demonstrate.

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+** recommended
- `pip`
- Git

No API key or external service account is required.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Run the Application

### CLI — Core Assignment Demonstration

```bash
python run.py
```

Example interaction:

```text
=== CAPSTONE: TECH STACK RECOMMENDER ===
Content-based filtering | TF-IDF + cosine similarity

1) Target role/domain: data science
2) Project needs: analytics dashboard
3) Skills/interests: Python SQL cloud
4) Optional extra interests: machine learning
```

The system then returns the highest-scoring technology stacks.

### Streamlit — Visual Demonstration

```bash
streamlit run app/streamlit_app.py
```

The Streamlit application is a presentation layer over the same recommendation engine. It does not contain a separate or simplified recommendation algorithm.

---

## 🧪 Testing

Run the automated test suite:

```bash
pytest -q
```

The suite covers the project's critical paths:

- Minimum input validation
- Content-based ranking
- Cosine-score validity
- Top-N filtering
- Cold-start fallback

Expected result:

```text
4 passed
```

---

## 📊 Example Recommendation Flow

For a user interested in:

```text
Role:       Data Science
Needs:      Analytics Dashboard
Interests:  Python, SQL, Cloud
```

the system might produce a result conceptually similar to:

| Rank | Recommendation | Match |
|---:|---|---:|
| 🥇 1 | Python Data Science Stack | High |
| 🥈 2 | SQL Reporting Stack | High |
| 🥉 3 | Data Engineering Stack | Moderate |

The exact scores are calculated dynamically from the TF-IDF representation of the catalog and user profile.

---

## 📚 Dataset

The included `data/tech_stacks.csv` catalog contains technology-stack candidates with attributes such as:

- Frontend
- Backend
- Database
- Deployment
- Category
- Tags
- Description
- Popularity

The catalog is intentionally human-readable so that the recommendation logic remains transparent and easy to explain during an internship presentation or supervisor review.

To add a new stack, add a row containing the required fields and restart the application.

---

## 🧱 Design Principles

### Correctness first

The implementation follows the mathematical approach specified by the project rather than replacing it with a more complicated model.

### Explainability

A recommendation can be traced through:

```text
User input
   ↓
Normalized terms
   ↓
TF-IDF representation
   ↓
Cosine similarity
   ↓
Score
   ↓
Rank
   ↓
Top-N output
```

### Simplicity

The system uses a local CSV catalog instead of adding a database solely for architectural appearance.

### Extensibility

The core `TechStackRecommender` class is independent of the CLI and Streamlit interface, making it straightforward to integrate into another application later.

---

## 🔐 Security & Configuration

There are currently no secrets or API credentials in the project.

The repository includes `.env.example` as a safe configuration placeholder. The `.gitignore` also excludes `.env` and common local Python artifacts.

---

## 📋 Requirement Traceability

| Assignment Requirement | Implementation | Status |
|---|---|:---:|
| Tech Stack Recommender | `app/recommender.py` | ✅ |
| Input → Processing → Output | CLI + recommender pipeline | ✅ |
| Minimum 3 user inputs | `app/cli.py` | ✅ |
| Content-Based Filtering | `TechStackRecommender` | ✅ |
| Shared feature vocabulary | Single fitted TF-IDF vectorizer | ✅ |
| Vector mapping | `preprocessing.py` | ✅ |
| TF-IDF weighting | `TfidfVectorizer` | ✅ |
| Avoid binary overlap | Weighted TF-IDF features | ✅ |
| Cosine similarity | `cosine_similarity()` | ✅ |
| Score available items | `recommend()` | ✅ |
| Sort by relevance | Descending score sort | ✅ |
| Top-N filtering | `.head(top_n)` | ✅ |
| Choice overload reduction | Top-3 default | ✅ |
| User cold start | Popularity fallback | ✅ |
| Item cold start | Metadata-based scoring | ✅ |
| Demonstration UI | Streamlit | ✅ |
| Automated tests | `tests/test_recommender.py` | ✅ |

---

## 📦 Deliverables

### Core submission

- ✅ Complete source code
- ✅ Recommendation engine
- ✅ Technology-stack dataset
- ✅ Requirements file
- ✅ README documentation
- ✅ Automated tests
- ✅ Working CLI demonstration

### Recommended presentation material

- Architecture diagram
- Screenshot of the Streamlit interface
- Screenshot of sample recommendations
- Short explanation of TF-IDF
- Short explanation of cosine similarity
- Cold-start demonstration
- Test output showing passing tests

---

## 🎓 Capstone Context

This repository implements the **Tech Stack Recommender** capstone following the supplied internship project specification and its progression from:

**Passive Classification → Active Prediction**

The final system converts structured and unstructured preference signals into an actionable ranked recommendation list while remaining transparent, deterministic, and easy to demonstrate.

---

## 👨‍💻 Author

**Hadeed Jalani**  
AI / ML Internship Project — Tech Stack Recommender

---

<div align="center">

### ⭐ If this project helped you, consider starring the repository.

**Built with Python · TF-IDF · Cosine Similarity · Content-Based Recommendation**

</div>

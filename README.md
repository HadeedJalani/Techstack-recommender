# Capstone: Tech Stack Recommender

A simple **content-based recommendation engine** that recommends technology stacks from user preferences. It follows the assignment's IPO architecture:

**Input → Processing → Scoring → Sorting → Filtering → Output**

The implementation uses **TF-IDF** for feature weighting and **cosine similarity** for matching a user profile against the tech-stack catalog.

## Assignment alignment

- Content-based filtering only; no collaborative filtering or historical user dataset.
- User provides a minimum of three inputs.
- User and item features are represented in the same TF-IDF vocabulary space.
- Generic terms are down-weighted by IDF; descriptive terms receive greater weight.
- Cosine similarity is used instead of Euclidean distance because it is insensitive to vector magnitude.
- Every catalog item is scored.
- Results are sorted in descending score order.
- Output is truncated to Top-N to prevent choice overload.
- Cold-start behavior is handled explicitly with a global-popularity fallback.

## Architecture

```text
User inputs (3+)
      |
      v
Text normalization
      |
      v
User profile document -----------------------+
                                             |
Tech-stack CSV -> item documents -> TF-IDF ->+-> cosine similarity
                                             |
                                             v
                                      Score every item
                                             |
                                             v
                                    Sort score DESC
                                             |
                                             v
                                        Top-N filter
                                             |
                                             v
                                  Ranked recommendations
```

## Project structure

```text
techstack_recommender/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── recommender.py
│   ├── cli.py
│   └── streamlit_app.py
├── data/
│   └── tech_stacks.csv
├── tests/
│   └── test_recommender.py
├── .env.example
├── .gitignore
├── requirements.txt
├── run.py
└── README.md
```

## Setup

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

No API key is required for the core assignment.

## Run

### Required CLI demonstration

```bash
python run.py
```

Enter at least three preferences, for example:

```text
1) Target role/domain: data science
2) Project needs: analytics dashboard
3) Skills/interests: Python SQL cloud
```

The program returns the Top-3 ranked stacks with cosine similarity scores.

### Optional visual demo

```bash
streamlit run app/streamlit_app.py
```

The Streamlit interface is only a presentation layer over the same recommender engine; it does not replace the required core logic.

### Tests

```bash
pytest -q
```

## Algorithm

### 1. Feature extraction

Each tech stack becomes one content document from category, frontend, backend, database, deployment, tags, and description.

### 2. TF-IDF

For term `t` in document `d`:

```text
TF(t,d) = count(t in d) / total terms in d
IDF(t) = log(Total documents / documents containing t)
TF-IDF(t,d) = TF(t,d) * IDF(t)
```

`scikit-learn` performs the production implementation and L2 normalization.

### 3. User vector

The three or more user inputs are joined into one profile document and transformed with the **same fitted vectorizer**, guaranteeing a shared vocabulary.

### 4. Similarity

```text
cosine(A,B) = (A · B) / (||A|| ||B||)
```

With non-negative TF-IDF vectors, the score is in `[0, 1]`.

### 5. Ranking and filtering

All catalog items receive a score, the list is sorted descending, and only Top-N is returned.

## Cold start

### User cold start

The three-input onboarding flow bootstraps a profile before scoring. If none of the entered terms exist in the learned catalog vocabulary, the system bypasses similarity ranking and uses the catalog popularity field as a global/trending fallback. This directly addresses the assignment’s User Cold Start requirement.

### Item cold start

A new item can be added to `data/tech_stacks.csv` with metadata. After restarting the application, it can be scored without historical user interactions, which is a key advantage of content-based filtering.

## Engineering notes

- Deterministic results with stable ID tie-breaking.
- No external API calls.
- No authentication or database is required by the supplied assignment.
- No model training/fine-tuning is required; TF-IDF is fit from the catalog at startup.
- The catalog is transparent and can be expanded without changing the recommendation algorithm.

# Learning Path Predictor

A sequential, prerequisite-aware course and study-material recommendation engine with an interactive **Streamlit** web application (`app.py`) and a zero-dependency pure-Python core (`learning_path/`).

Given a student's completed courses in chronological order, the model predicts the **Top-N most relevant next courses**, displays normalized match share percentages, checks prerequisite readiness, recommends curated study materials, and projects a multi-step future learning roadmap.

---

## Architecture & Scoring Model

The `LearningPathRecommender` (`learning_path/model.py`) scores every unseen course $c \notin H$ using four combined signals:

1. **Second-Order Markov Transitions ($P_2(c \mid c_{t-2}, c_{t-1})$):** Captures exact two-course prerequisite chains (e.g., `CS101 → DS101 → DS102 → DS201`) using Laplace-smoothed trigram counts ($\alpha = 0.5$).
2. **First-Order Markov Transitions ($P_1(c \mid c_{t-1})$):** Backs off cleanly to single-course transitions when a two-course pair is sparse.
3. **Directional Co-occurrence ($P_{\text{co}}(c \mid H)$):** Measures how frequently course $c$ is taken *anywhere after* each earlier course in the student's history, capturing long-range dependencies across electives.
4. **Curriculum Level & Prerequisite Multiplier:** Rewards natural level progression (`Beginner → Intermediate → Advanced`) and satisfied prerequisites while gently penalizing premature jumps.

---

## Repository Structure

```text
learning-path-predictor/
├── app.py                     # Streamlit interactive UI (Top-N recommendations, study materials & roadmap)
├── learning_path/
│   ├── __init__.py            # Package exports
│   ├── __main__.py            # python -m learning_path entrypoint
│   ├── catalog.py             # CourseCatalog, prerequisite checker, title/ID resolver, CSV loaders
│   ├── model.py               # LearningPathRecommender & PopularityBaseline
│   ├── evaluate.py            # Held-out 80/20 split evaluation & RESULTS.md generator
│   └── cli.py                 # Command-line interface (recommend, roadmap, catalog)
├── data/
│   ├── courses.csv            # 24 courses across 6 domains with credits, prerequisites & study materials
│   └── enrollments.csv        # 180 student course progression sequences
├── tests/
│   └── test_learning_path.py  # 18 unit & integration tests
├── .github/workflows/ci.yml   # GitHub Actions CI (pytest + held-out evaluation)
├── pytest.ini
├── requirements.txt
├── RESULTS.md
└── README.md
```

---

## Quickstart

### 1. Install Dependencies & Launch Streamlit UI
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 2. Run via CLI
```bash
python -m learning_path recommend CS101 DS101 DS102 --top-k 5
python -m learning_path roadmap WD101 WD102 --steps 3
python -m learning_path catalog
```

### 3. Run Tests & Evaluation
```bash
pytest -q
python -m learning_path.evaluate
```

---

## Sample CLI Output

```text
$ python -m learning_path recommend CS101 DS101 DS102 --top-k 3
Completed: CS101, DS101, DS102
Top 3 Recommended Next Courses:
  1. DS201: Machine Learning Foundations (Data Science, Intermediate)  [score: 0.72] — Commonly follows DS101 -> DS102 | Prerequisites met | Data Science (Intermediate) | Study Material: Hands-On Machine Learning (Aurélien Géron)
  2. DS202: Deep Learning and Neural Networks (Data Science, Advanced)  [score: 0.16] — Strong downstream fit with completed courses | Recommended prereq: DS201 | Data Science (Advanced) | Study Material: Deep Learning Specialization & PyTorch Tutorials
  3. CS201: Database Management Systems (Systems, Intermediate)  [score: 0.12] — Direct next step after DS102 | Prerequisites met | Systems (Intermediate) | Study Material: Designing Data-Intensive Applications
```

---

# Learning Path Predictor — Evaluation Report

## 1. Model & Training Summary
The hybrid sequence recommendation model was fitted on historical student course completion sequences using second-order Markov transitions, first-order Markov transitions, directional co-occurrence, and curriculum prerequisite progression.
- **Total Training Student Paths:** 144
- **Held-Out Test Student Paths:** 36
- **Evaluated Next-Course Steps:** 141
- **Smoothing Parameter ($\alpha$):** 0.5 (Laplace smoothing)

## 2. Held-Out Validation Metrics

| Model | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Perplexity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hybrid Sequence Recommender (Ours)** | **69.5%** | **97.2%** | **100.0%** | **0.828** | **3.45** |
| First-Order Markov Ablation | 62.4% | 90.8% | 98.6% | 0.765 | 3.47 |
| Popularity Baseline | 9.2% | 22.0% | 31.2% | 0.163 | — |

Hit@K measures the percentage of test steps where the student's actual next course appears within the top $K$ recommendations. MRR@5 (Mean Reciprocal Rank) measures ranking precision within the top 5, and Perplexity measures next-course uncertainty (lower is better).

## 3. Automated Testing
The project's automated test suite (`tests/test_learning_path.py`) completed successfully:
- **Tests passed:** 18
- **Tests failed:** 0
- **Coverage:** Catalog loading, prerequisite checks, study-material lookup, ID/title resolution, cold-start behavior, domain filtering, roadmap rollout, CLI execution, evaluation metrics, and Streamlit app validation.

## 4. Sample Recommendation Evaluation
- **Data Science Track (`CS101 → DS101 → DS102`):** Prioritizes `DS201: Machine Learning Foundations` as the top next course with study material (*Hands-On Machine Learning*), followed by `DS202: Deep Learning and Neural Networks`.
- **Web Development Track (`WD101 → WD102`):** Recommends `WD201: Backend APIs with Node and Express` and `CS201: Database Management Systems`.
- **Cold-Start Input (`[]`):** Defaults cleanly to `Beginner`-level entry courses without prerequisite penalties.

## 5. Limitations
- **Limited Cohort & Catalog Size:** Trained on 180 student enrollment sequences across 24 courses (`data/enrollments.csv`); larger registrar datasets with hundreds of electives will require tuning smoothing and backoff weights.
- **No Grade or Semester-Gap Weighting:** Completed courses are treated as unweighted sequential tokens without factoring in student grades, failed attempts, or time gaps between semesters.
- **Greedy Multi-Step Rollout:** `predict_path()` greedily selects the top-1 course at each step rather than optimizing joint multi-semester credit caps.

## 6. Reproducibility
Run the test suite, regenerate this report, or launch the Streamlit UI:
```bash
pip install -r requirements.txt
pytest -q
python -m learning_path.evaluate
streamlit run app.py
```


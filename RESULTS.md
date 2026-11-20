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

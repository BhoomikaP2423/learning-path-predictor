"""Test suite for Learning Path Predictor (18 plain functions for pytest & standalone)."""

from pathlib import Path
from learning_path.catalog import load_catalog, load_enrollments
from learning_path.model import LearningPathRecommender, PopularityBaseline
from learning_path.evaluate import evaluate_models
from learning_path.cli import main as cli_main


ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data" / "courses.csv"
ENROLL_PATH = ROOT / "data" / "enrollments.csv"


def test_catalog_loads_courses_and_study_materials():
    catalog = load_catalog(CATALOG_PATH)
    assert len(catalog.all_courses()) >= 20
    cs101 = catalog.get("cs101")
    assert cs101 is not None
    assert cs101.course_id == "CS101"
    assert cs101.level == "Beginner"
    assert len(cs101.study_material) > 0


def test_catalog_prerequisite_checks():
    catalog = load_catalog(CATALOG_PATH)
    assert catalog.missing_prerequisites("DS201", ["CS101"]) == ["DS101", "DS102"]
    assert catalog.missing_prerequisites("DS201", ["CS101", "DS101", "DS102"]) == []


def test_catalog_resolve_by_title_and_id():
    catalog = load_catalog(CATALOG_PATH)
    assert catalog.resolve_id("CS101") == "CS101"
    assert catalog.resolve_id("Introduction to Python Programming") == "CS101"
    assert catalog.resolve_id("UNKNOWN999") is None


def test_enrollments_load_non_empty_sequences():
    seqs = load_enrollments(ENROLL_PATH)
    assert len(seqs) >= 150
    assert all(len(s) >= 2 for s in seqs)


def test_recommender_excludes_already_completed():
    catalog = load_catalog(CATALOG_PATH)
    seqs = load_enrollments(ENROLL_PATH)
    model = LearningPathRecommender(catalog=catalog).fit(seqs)
    completed = ["CS101", "DS101", "DS102"]
    recs = model.recommend(completed, top_k=5)
    rec_ids = [cid for cid, _, _ in recs]
    assert len(rec_ids) == 5
    for c in completed:
        assert c not in rec_ids


def test_data_science_progression_recommends_ml():
    catalog = load_catalog(CATALOG_PATH)
    seqs = load_enrollments(ENROLL_PATH)
    model = LearningPathRecommender(catalog=catalog).fit(seqs)
    recs = model.recommend(["CS101", "DS101", "DS102"], top_k=3)
    rec_ids = [cid for cid, _, _ in recs]
    assert "DS201" in rec_ids


def test_web_progression_recommends_backend():
    catalog = load_catalog(CATALOG_PATH)
    seqs = load_enrollments(ENROLL_PATH)
    model = LearningPathRecommender(catalog=catalog).fit(seqs)
    recs = model.recommend(["WD101", "WD102"], top_k=3)
    rec_ids = [cid for cid, _, _ in recs]
    assert "WD201" in rec_ids


def test_cloud_progression_recommends_kubernetes():
    catalog = load_catalog(CATALOG_PATH)
    seqs = load_enrollments(ENROLL_PATH)
    model = LearningPathRecommender(catalog=catalog).fit(seqs)
    recs = model.recommend(["CS101", "CS202", "CS203", "CL101"], top_k=3)
    rec_ids = [cid for cid, _, _ in recs]
    assert "CL201" in rec_ids


def test_cold_start_empty_history_recommends_beginner():
    catalog = load_catalog(CATALOG_PATH)
    seqs = load_enrollments(ENROLL_PATH)
    model = LearningPathRecommender(catalog=catalog).fit(seqs)
    recs = model.recommend([], top_k=3)
    assert len(recs) == 3
    top_course = catalog.get(recs[0][0])
    assert top_course is not None
    assert top_course.level == "Beginner"


def test_domain_filter_restricts_results():
    catalog = load_catalog(CATALOG_PATH)
    seqs = load_enrollments(ENROLL_PATH)
    model = LearningPathRecommender(catalog=catalog).fit(seqs)
    recs = model.recommend(["CS101"], top_k=4, domain_filter="Cloud & DevOps")
    assert len(recs) > 0
    for cid, _, _ in recs:
        assert catalog.get(cid).domain == "Cloud & DevOps"


def test_normalized_shares_sum_to_one():
    catalog = load_catalog(CATALOG_PATH)
    seqs = load_enrollments(ENROLL_PATH)
    model = LearningPathRecommender(catalog=catalog).fit(seqs)
    recs = model.recommend(["CS101", "CS102"], top_k=5)
    total_share = sum(share for _, share, _ in recs)
    assert abs(total_share - 1.0) < 1e-3


def test_candidate_distribution_sums_to_one():
    catalog = load_catalog(CATALOG_PATH)
    seqs = load_enrollments(ENROLL_PATH)
    model = LearningPathRecommender(catalog=catalog).fit(seqs)
    dist = model.candidate_distribution(["CS101", "DS101"])
    assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_predict_path_returns_distinct_steps():
    catalog = load_catalog(CATALOG_PATH)
    seqs = load_enrollments(ENROLL_PATH)
    model = LearningPathRecommender(catalog=catalog).fit(seqs)
    completed = ["CS101", "DS101"]
    roadmap = model.predict_path(completed, steps=3)
    assert len(roadmap) == 3
    step_ids = [cid for cid, _, _ in roadmap]
    assert len(set(step_ids)) == 3
    assert set(step_ids).isdisjoint(set(completed))


def test_popularity_baseline_excludes_seen():
    seqs = load_enrollments(ENROLL_PATH)
    base = PopularityBaseline().fit(seqs)
    recs = base.recommend(["CS101", "DS101"], top_k=5)
    rec_ids = [cid for cid, _, _ in recs]
    assert "CS101" not in rec_ids
    assert "DS101" not in rec_ids


def test_evaluation_beats_baseline_and_markov1():
    seqs = load_enrollments(ENROLL_PATH)
    res = evaluate_models(seqs, CATALOG_PATH)
    assert res["eval_steps"] > 20
    assert res["hybrid"]["top3"] > res["baseline"]["top3"]
    assert res["hybrid"]["mrr"] > res["baseline"]["mrr"]
    assert res["hybrid"]["perplexity"] < res["markov1"]["perplexity"]


def test_cli_recommend_includes_study_material(capsys):
    rc = cli_main(["recommend", "CS101", "DS101", "DS102", "--top-k", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Top 3 Recommended Next Courses" in out
    assert "Study Material:" in out


def test_cli_roadmap_and_catalog(capsys):
    rc1 = cli_main(["roadmap", "WD101", "--steps", "2"])
    assert rc1 == 0
    out1 = capsys.readouterr().out
    assert "Step 1:" in out1
    assert "Step 2:" in out1

    rc2 = cli_main(["catalog"])
    assert rc2 == 0
    out2 = capsys.readouterr().out
    assert "CS101" in out2


def test_streamlit_app_module_compiles_cleanly():
    app_file = ROOT / "app.py"
    source = app_file.read_text(encoding="utf-8")
    compile(source, str(app_file), "exec")
    assert "st.multiselect" in source
    assert "model.recommend" in source

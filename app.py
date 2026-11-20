"""Streamlit interactive web interface for the Learning Path Predictor."""

from pathlib import Path
import streamlit as st

from learning_path.catalog import load_catalog, load_enrollments
from learning_path.model import LearningPathRecommender


st.set_page_config(
    page_title="Learning Path Predictor",
    page_icon="🎓",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_engine():
    catalog = load_catalog(ROOT / "data" / "courses.csv")
    sequences = load_enrollments(ROOT / "data" / "enrollments.csv")
    model = LearningPathRecommender(catalog=catalog).fit(sequences)
    return catalog, sequences, model


catalog, sequences, model = load_engine()

st.title("🎓 Learning Path Predictor")
st.markdown(
    "Select your completed courses in chronological order to receive personalized **Top-N next-course recommendations**, "
    "curated **study materials**, prerequisite checks, and a projected multi-step **learning roadmap**."
)

# Sidebar controls
st.sidebar.header("Configuration")
top_n = st.sidebar.slider("Number of recommendations (Top-N)", min_value=1, max_value=10, value=5)
roadmap_steps = st.sidebar.slider("Future roadmap steps", min_value=1, max_value=5, value=3)

all_courses = catalog.all_courses()
domains = ["All Domains"] + sorted({c.domain for c in all_courses})
selected_domain = st.sidebar.selectbox("Filter recommendations by domain", domains)
domain_filter = None if selected_domain == "All Domains" else selected_domain

st.sidebar.markdown("---")
st.sidebar.metric("Catalog Size", f"{len(all_courses)} Courses")
st.sidebar.metric("Training Paths", f"{len(sequences)} Students")

# Course selection
course_options = {
    f"{c.course_id} — {c.title} ({c.domain} · {c.level})": c.course_id
    for c in all_courses
}
default_labels = [
    lbl for lbl, cid in course_options.items() if cid in {"CS101", "DS101", "DS102"}
]

selected_labels = st.multiselect(
    "Choose your completed courses (in chronological order):",
    options=list(course_options.keys()),
    default=default_labels,
)
completed_ids = [course_options[lbl] for lbl in selected_labels]

if completed_ids:
    total_credits = sum(catalog.get(c).credits for c in completed_ids if catalog.get(c))
    st.caption(f"Completed sequence ({len(completed_ids)} courses · {total_credits} credits): {' → '.join(completed_ids)}")
else:
    st.info("No completed courses selected yet — showing top beginner entry-point courses.")

col1, col2 = st.columns([1.4, 1.0])

with col1:
    st.subheader(f"Top {top_n} Recommended Next Courses")
    recs = model.recommend(completed_ids, top_k=top_n, domain_filter=domain_filter)
    if not recs:
        st.warning("No matching unseen courses found for the selected domain filter.")
    else:
        for rank, (cid, share, reason) in enumerate(recs, start=1):
            course = catalog.get(cid)
            title = course.title if course else cid
            meta = f"{course.domain} • {course.level} • {course.credits} Credits" if course else ""
            with st.container(border=True):
                c_left, c_right = st.columns([4, 1])
                with c_left:
                    st.markdown(f"**#{rank} {cid}: {title}**")
                    st.caption(f"{meta}  |  _{reason}_")
                    if course and course.study_material:
                        st.markdown(f"📚 **Recommended Study Material:** {course.study_material}")
                with c_right:
                    st.metric("Match Share", f"{share * 100:.1f}%")

with col2:
    st.subheader("Projected Multi-Step Roadmap")
    roadmap = model.predict_path(completed_ids, steps=roadmap_steps)
    if not roadmap:
        st.write("All catalog courses completed!")
    else:
        for step_idx, (cid, _, reason) in enumerate(roadmap, start=1):
            course = catalog.get(cid)
            title = course.title if course else cid
            st.markdown(f"**Step {step_idx}:** `{cid}` — {title}")
            st.caption(reason)
            if course and course.study_material:
                st.caption(f"📚 {course.study_material}")

st.markdown("---")
with st.expander("Explore Full Course Catalog & Study Materials"):
    st.dataframe(
        [
            {
                "Course ID": c.course_id,
                "Title": c.title,
                "Domain": c.domain,
                "Level": c.level,
                "Credits": c.credits,
                "Prerequisites": ", ".join(c.prerequisites) if c.prerequisites else "None",
                "Recommended Study Material": c.study_material,
            }
            for c in all_courses
        ],
        use_container_width=True,
        hide_index=True,
    )

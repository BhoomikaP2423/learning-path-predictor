"""CLI entrypoint for quick recommendations, study materials, and roadmap rollouts."""

import argparse
from pathlib import Path
from typing import List, Optional

from .catalog import load_catalog, load_enrollments
from .model import LearningPathRecommender


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="learning_path",
        description="Learning Path Predictor CLI: recommend next courses and study materials.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rec_p = sub.add_parser("recommend", help="Recommend top-N next courses")
    rec_p.add_argument(
        "courses",
        nargs="+",
        help="Completed course IDs in chronological order (e.g., CS101 DS101 DS102)",
    )
    rec_p.add_argument("--top-k", type=int, default=5, help="Number of recommendations")
    rec_p.add_argument("--domain", type=str, default=None, help="Optional domain filter")

    path_p = sub.add_parser("roadmap", help="Generate a multi-step future course roadmap")
    path_p.add_argument("courses", nargs="+", help="Completed course IDs")
    path_p.add_argument("--steps", type=int, default=3, help="Number of future steps")

    sub.add_parser("catalog", help="List all available courses in the catalog")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    catalog = load_catalog(root / "data" / "courses.csv")
    sequences = load_enrollments(root / "data" / "enrollments.csv")
    model = LearningPathRecommender(catalog=catalog).fit(sequences)

    if args.command == "catalog":
        for c in catalog.all_courses():
            print(f"{c.course_id:6s} | {c.title:42s} | {c.domain:16s} | {c.level}")
        return 0

    if args.command == "recommend":
        recs = model.recommend(args.courses, top_k=args.top_k, domain_filter=args.domain)
        print(f"Completed: {', '.join(c.upper() for c in args.courses)}")
        print(f"Top {len(recs)} Recommended Next Courses:")
        for rank, (cid, share, reason) in enumerate(recs, start=1):
            course = catalog.get(cid)
            label = catalog.format_course(cid)
            mat = f" | Study Material: {course.study_material}" if (course and course.study_material) else ""
            print(f"  {rank}. {label}  [score: {share:.2f}] — {reason}{mat}")
        return 0

    if args.command == "roadmap":
        steps = model.predict_path(args.courses, steps=args.steps)
        print(f"Starting from: {', '.join(c.upper() for c in args.courses)}")
        print("Projected Learning Roadmap:")
        for idx, (cid, _, reason) in enumerate(steps, start=1):
            print(f"  Step {idx}: {catalog.format_course(cid)} ({reason})")
        return 0

    return 1

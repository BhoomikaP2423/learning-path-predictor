"""Course catalog and student enrollment sequence loaders."""

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Course:
    course_id: str
    title: str
    domain: str
    level: str
    credits: int
    prerequisites: Tuple[str, ...] = ()
    study_material: str = ""


class CourseCatalog:
    """In-memory lookup for course metadata, prerequisites, and title/ID normalization."""

    def __init__(self, courses: Dict[str, Course]):
        self.courses = courses
        self._title_to_id = {c.title.lower(): c.course_id for c in courses.values()}

    def get(self, course_id: str) -> Optional[Course]:
        return self.courses.get(course_id.strip().upper())

    def resolve_id(self, query: str) -> Optional[str]:
        cleaned = query.strip()
        upper = cleaned.upper()
        if upper in self.courses:
            return upper
        return self._title_to_id.get(cleaned.lower())

    def missing_prerequisites(self, course_id: str, completed: Sequence[str]) -> List[str]:
        course = self.get(course_id)
        if not course or not course.prerequisites:
            return []
        done = {c.strip().upper() for c in completed}
        return [p for p in course.prerequisites if p not in done]

    def format_course(self, course_id: str) -> str:
        course = self.get(course_id)
        if not course:
            return course_id
        return f"{course.course_id}: {course.title} ({course.domain}, {course.level})"

    def all_courses(self) -> List[Course]:
        return list(self.courses.values())


def load_catalog(csv_path: str | Path) -> CourseCatalog:
    courses: Dict[str, Course] = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["course_id"].strip().upper()
            raw_prereqs = row.get("prerequisites", "").strip()
            prereqs = tuple(p.strip().upper() for p in raw_prereqs.split("|") if p.strip())
            courses[cid] = Course(
                course_id=cid,
                title=row["title"].strip(),
                domain=row["domain"].strip(),
                level=row["level"].strip(),
                credits=int(row["credits"]),
                prerequisites=prereqs,
                study_material=row.get("study_material", "").strip(),
            )
    return CourseCatalog(courses)


def load_enrollments(csv_path: str | Path) -> List[List[str]]:
    sequences: List[List[str]] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_seq = row.get("course_sequence", "").strip()
            if not raw_seq:
                continue
            seq = [c.strip().upper() for c in raw_seq.split("|") if c.strip()]
            if seq:
                sequences.append(seq)
    return sequences

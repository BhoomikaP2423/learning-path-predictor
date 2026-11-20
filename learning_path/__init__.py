"""Learning Path Predictor: Sequential and prerequisite-aware course recommender."""

from .catalog import Course, CourseCatalog, load_catalog, load_enrollments
from .model import LearningPathRecommender, PopularityBaseline

__all__ = [
    "Course",
    "CourseCatalog",
    "load_catalog",
    "load_enrollments",
    "LearningPathRecommender",
    "PopularityBaseline",
]

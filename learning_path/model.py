"""Pure-Python sequential Markov + co-occurrence + prerequisite-aware Learning Path Predictor."""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .catalog import CourseCatalog


LEVEL_ORDER = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}


class PopularityBaseline:
    """Recommends globally most popular unseen courses."""

    def __init__(self):
        self.counts: Counter = Counter()
        self.total: int = 0

    def fit(self, sequences: Sequence[Sequence[str]]) -> "PopularityBaseline":
        self.counts.clear()
        for seq in sequences:
            for cid in seq:
                self.counts[cid.strip().upper()] += 1
        self.total = sum(self.counts.values())
        return self

    def recommend(
        self, completed: Sequence[str], top_k: int = 5
    ) -> List[Tuple[str, float, str]]:
        seen = {c.strip().upper() for c in completed}
        candidates = [
            (cid, count / max(self.total, 1), "Popular across student paths")
            for cid, count in self.counts.most_common()
            if cid not in seen
        ]
        return candidates[:top_k]


class LearningPathRecommender:
    """Hybrid sequence model combining:
    1. Second-order Markov transitions P(c_t | c_{t-2}, c_{t-1})
    2. First-order Markov transitions P(c_t | c_{t-1})
    3. Directional co-occurrence across full student history
    4. Curriculum level progression & prerequisite readiness
    """

    def __init__(
        self,
        catalog: Optional[CourseCatalog] = None,
        alpha: float = 0.5,
        w_second_order: float = 0.45,
        w_first_order: float = 0.35,
        w_cooccurrence: float = 0.20,
    ):
        self.catalog = catalog
        self.alpha = alpha
        self.w_second_order = w_second_order
        self.w_first_order = w_first_order
        self.w_cooccurrence = w_cooccurrence

        self.unigram_counts: Counter = Counter()
        self.bigram_counts: Dict[str, Counter] = defaultdict(Counter)
        self.trigram_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.after_counts: Dict[str, Counter] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def fit(self, sequences: Sequence[Sequence[str]]) -> "LearningPathRecommender":
        self.unigram_counts.clear()
        self.bigram_counts.clear()
        self.trigram_counts.clear()
        self.after_counts.clear()
        self.vocab.clear()

        if self.catalog:
            self.vocab.update(self.catalog.courses.keys())

        for raw_seq in sequences:
            seq = [c.strip().upper() for c in raw_seq if c.strip()]
            for i, cid in enumerate(seq):
                self.vocab.add(cid)
                self.unigram_counts[cid] += 1
                if i >= 1:
                    prev1 = seq[i - 1]
                    self.bigram_counts[prev1][cid] += 1
                if i >= 2:
                    prev2, prev1 = seq[i - 2], seq[i - 1]
                    self.trigram_counts[(prev2, prev1)][cid] += 1
                for j in range(i):
                    earlier = seq[j]
                    if earlier != cid:
                        self.after_counts[earlier][cid] += 1

        return self

    def _smoothed_prob(self, counter: Counter, target: str) -> float:
        vocab_size = max(len(self.vocab), 1)
        total = sum(counter.values())
        return (counter[target] + self.alpha) / (total + self.alpha * vocab_size)

    def _curriculum_multiplier(self, completed: Sequence[str], candidate: str) -> float:
        if not self.catalog:
            return 1.0
        cand_course = self.catalog.get(candidate)
        if not cand_course:
            return 1.0

        cand_lvl = LEVEL_ORDER.get(cand_course.level, 2)
        completed_lvls = [
            LEVEL_ORDER.get(self.catalog.get(c).level, 1)
            for c in completed
            if self.catalog.get(c)
        ]

        if not completed_lvls:
            lvl_factor = 1.20 if cand_lvl == 1 else 0.85
        else:
            max_done = max(completed_lvls)
            if cand_lvl == max_done or cand_lvl == max_done + 1:
                lvl_factor = 1.15
            elif cand_lvl > max_done + 1:
                lvl_factor = 0.70
            else:
                lvl_factor = 0.95

        missing = self.catalog.missing_prerequisites(candidate, completed)
        if not cand_course.prerequisites:
            prereq_factor = 1.05
        elif len(missing) == 0:
            prereq_factor = 1.20
        else:
            prereq_factor = max(0.65, 1.0 - 0.15 * len(missing))

        return lvl_factor * prereq_factor

    def _build_reason(
        self,
        candidate: str,
        completed: List[str],
        p_co: float,
    ) -> str:
        reasons = []
        if len(completed) >= 2 and self.trigram_counts[(completed[-2], completed[-1])][candidate] > 0:
            reasons.append(f"Commonly follows {completed[-2]} -> {completed[-1]}")
        elif len(completed) >= 1 and self.bigram_counts[completed[-1]][candidate] > 0:
            reasons.append(f"Direct next step after {completed[-1]}")
        elif p_co > 0:
            reasons.append("Strong downstream fit with completed courses")
        else:
            reasons.append("Foundational curriculum match")

        if self.catalog:
            c_obj = self.catalog.get(candidate)
            if c_obj:
                missing = self.catalog.missing_prerequisites(candidate, completed)
                if c_obj.prerequisites and not missing:
                    reasons.append("Prerequisites met")
                elif missing:
                    reasons.append(f"Recommended prereq: {', '.join(missing)}")
                reasons.append(f"{c_obj.domain} ({c_obj.level})")
        return " | ".join(reasons)

    def score_candidates(self, completed: Sequence[str]) -> Dict[str, Tuple[float, str]]:
        cleaned = [c.strip().upper() for c in completed if c.strip()]
        seen = set(cleaned)
        scores: Dict[str, Tuple[float, str]] = {}

        for cand in self.vocab:
            if cand in seen:
                continue

            if len(cleaned) >= 2:
                p_tri = self._smoothed_prob(self.trigram_counts[(cleaned[-2], cleaned[-1])], cand)
            elif len(cleaned) == 1:
                p_tri = self._smoothed_prob(self.bigram_counts[cleaned[-1]], cand)
            else:
                p_tri = self._smoothed_prob(self.unigram_counts, cand)

            if len(cleaned) >= 1:
                p_bi = self._smoothed_prob(self.bigram_counts[cleaned[-1]], cand)
            else:
                p_bi = self._smoothed_prob(self.unigram_counts, cand)

            if cleaned:
                co_probs = [self._smoothed_prob(self.after_counts[prev], cand) for prev in cleaned]
                p_co = sum(co_probs) / len(co_probs)
            else:
                p_co = self._smoothed_prob(self.unigram_counts, cand)

            raw_score = (
                self.w_second_order * p_tri
                + self.w_first_order * p_bi
                + self.w_cooccurrence * p_co
            )
            final_score = raw_score * self._curriculum_multiplier(cleaned, cand)
            reason = self._build_reason(cand, cleaned, p_co)
            scores[cand] = (final_score, reason)

        return scores

    def candidate_distribution(self, completed: Sequence[str]) -> Dict[str, float]:
        scored = self.score_candidates(completed)
        total = sum(sc for sc, _ in scored.values()) or 1.0
        return {cid: sc / total for cid, (sc, _) in scored.items()}

    def recommend(
        self,
        completed: Sequence[str],
        top_k: int = 5,
        domain_filter: Optional[str] = None,
    ) -> List[Tuple[str, float, str]]:
        scored = self.score_candidates(completed)
        filtered = []
        for cid, (sc, reason) in scored.items():
            if domain_filter and self.catalog:
                course = self.catalog.get(cid)
                if not course or course.domain.lower() != domain_filter.strip().lower():
                    continue
            filtered.append((cid, sc, reason))

        filtered.sort(key=lambda x: (-x[1], x[0]))
        top_items = filtered[:top_k]
        total_top = sum(x[1] for x in top_items) or 1.0
        return [(cid, round(sc / total_top, 4), reason) for cid, sc, reason in top_items]

    def predict_path(
        self, completed: Sequence[str], steps: int = 3
    ) -> List[Tuple[str, float, str]]:
        current = [c.strip().upper() for c in completed if c.strip()]
        roadmap: List[Tuple[str, float, str]] = []
        for _ in range(steps):
            recs = self.recommend(current, top_k=1)
            if not recs:
                break
            next_cid, conf, reason = recs[0]
            roadmap.append((next_cid, conf, reason))
            current.append(next_cid)
        return roadmap

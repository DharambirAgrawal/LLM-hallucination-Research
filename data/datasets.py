"""
Dataset loader for hallucination benchmarks.

Sources:
  - json      : local files such as the official HaluEval QA / dialogue /
                summarization data (RUCAIBox/HaluEval, commit pinned in
                provenance/sources.yaml; downloaded automatically by utils/resources.py)
  - hf        : a Hugging Face dataset (pin `revision` before reporting)
  - synthetic : 8 hand-written QA pairs with planted hallucinations
                (engineering smoke fixture only, not research data)

Every source is sampled with the configured seed, so `max_samples` draws a
reproducible random subset of the whole file rather than its first rows.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from loguru import logger


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class BenchmarkSample:
    """A single RAG hallucination benchmark sample."""
    sample_id:      str
    dataset:        str
    question:       str
    context:        str
    right_answer:         str
    hallucinated_answer: str
    metadata:       dict = field(default_factory=dict)
    # Extra labeled answers: {"answer": str, "label": 0|1, "source": str}.
    # RAGTruth / HaluBench use these instead of a right/hallucinated pair.
    labeled_answers: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sample_id":        self.sample_id,
            "dataset":          self.dataset,
            "question":         self.question,
            "context":          self.context,
            "right_answer":           self.right_answer,
            "hallucinated_answer":  self.hallucinated_answer,
            "metadata":         self.metadata,
            "labeled_answers":  self.labeled_answers,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkSample":
        return cls(**d)


# ─────────────────────────────────────────────
# Synthetic dataset
# ─────────────────────────────────────────────

SYNTHETIC_CONTEXTS = [
    {
        "context": (
            "The Python programming language was created by Guido van Rossum "
            "and first released in 1991. Python 2.0 was released in 2000, introducing "
            "features like list comprehensions and garbage collection. Python 3.0 was "
            "released in 2008 and was a major revision of the language that is not "
            "entirely backward-compatible."
        ),
        "question": "When was Python first released?",
        "factual_answer": "Python was first released in 1991.",
        "hallucinated_answer": "Python was first released in 1985 by James Gosling.",
    },
    {
        "context": (
            "The Great Wall of China is a series of fortifications built across the "
            "historical northern borders of ancient Chinese states and Imperial China "
            "as protection against various nomadic groups. Several walls were built "
            "from as early as the 7th century BC, with selective stretches later joined "
            "together by Qin Shi Huang (221–206 BC), the first emperor of China."
        ),
        "question": "Who built the first unified Great Wall of China?",
        "factual_answer": "The first unified Great Wall was built by Qin Shi Huang, the first emperor of China.",
        "hallucinated_answer": "The Great Wall was built by Emperor Wu of Han around 150 BC to fight the Mongols.",
    },
    {
        "context": (
            "Photosynthesis is a process used by plants and other organisms to convert "
            "light energy into chemical energy. In plants, photosynthesis occurs mainly "
            "in leaves inside chloroplasts. The overall equation is: "
            "6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂. "
            "Chlorophyll is the primary pigment that absorbs light."
        ),
        "question": "What is the main pigment involved in photosynthesis?",
        "factual_answer": "Chlorophyll is the primary pigment that absorbs light in photosynthesis.",
        "hallucinated_answer": "Melanin is the main pigment in photosynthesis, absorbing ultraviolet light.",
    },
    {
        "context": (
            "Albert Einstein published his special theory of relativity in 1905 and the "
            "general theory of relativity in 1915. He received the Nobel Prize in Physics "
            "in 1921 for his discovery of the law of the photoelectric effect, not for "
            "relativity. Einstein was born in Ulm, Germany in 1879."
        ),
        "question": "For what did Einstein receive the Nobel Prize?",
        "factual_answer": "Einstein received the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect.",
        "hallucinated_answer": "Einstein received the Nobel Prize in 1921 for his general theory of relativity.",
    },
    {
        "context": (
            "The human brain contains approximately 86 billion neurons. The brain is "
            "divided into several major regions including the cerebrum, cerebellum, and "
            "brainstem. The cerebrum is the largest part and is divided into four lobes: "
            "frontal, parietal, temporal, and occipital. The hippocampus plays a critical "
            "role in forming new memories."
        ),
        "question": "How many neurons does the human brain contain?",
        "factual_answer": "The human brain contains approximately 86 billion neurons.",
        "hallucinated_answer": "The human brain contains exactly 100 trillion neurons, making it the most complex organ.",
    },
    {
        "context": (
            "Machine learning is a subset of artificial intelligence that provides "
            "systems the ability to automatically learn and improve from experience "
            "without being explicitly programmed. Machine learning focuses on developing "
            "computer programs that can access data and use it to learn for themselves. "
            "Supervised learning, unsupervised learning, and reinforcement learning are "
            "the three main types of machine learning."
        ),
        "question": "What are the three main types of machine learning?",
        "factual_answer": "The three main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning.",
        "hallucinated_answer": "The three main types of machine learning are deep learning, neural networks, and statistical learning.",
    },
    {
        "context": (
            "The Amazon River is the largest river in the world by discharge volume of "
            "water. It is located in South America, flowing through Brazil, Peru, and "
            "Colombia. The Amazon basin is the world's largest tropical rainforest, "
            "covering over 5.5 million square kilometers. The river stretches "
            "approximately 6,400 kilometers."
        ),
        "question": "What is the Amazon River known for?",
        "factual_answer": "The Amazon is the largest river by discharge volume, located in South America, with a basin covering over 5.5 million square kilometers.",
        "hallucinated_answer": "The Amazon River is the longest river in the world at 7,500 km, located entirely within Brazil.",
    },
    {
        "context": (
            "Transformer architecture was introduced in the paper 'Attention Is All You Need' "
            "by Vaswani et al. in 2017. Transformers use self-attention mechanisms to "
            "process sequential data in parallel, unlike RNNs which process data sequentially. "
            "BERT and GPT are famous models built on the transformer architecture. "
            "The key innovation is the multi-head attention mechanism."
        ),
        "question": "When was the transformer architecture introduced?",
        "factual_answer": "The transformer architecture was introduced in 2017 in the paper 'Attention Is All You Need'.",
        "hallucinated_answer": "The transformer architecture was invented by Geoffrey Hinton in 2014 at Google DeepMind.",
    },
]


def _make_synthetic(max_samples: int, seed: int = 42) -> List[BenchmarkSample]:
    """Build the synthetic smoke fixture. The pool repeats once `max_samples`
    exceeds its 8 entries, so keep it small."""
    samples = []
    pool = SYNTHETIC_CONTEXTS * (max_samples // len(SYNTHETIC_CONTEXTS) + 1)
    pool = pool[:max_samples]
 
    for i, item in enumerate(pool):
        samples.append(BenchmarkSample(
            sample_id=f"synthetic_{i:04d}",
            dataset="synthetic",
            question=item["question"],
            context=item["context"],
            right_answer=item["factual_answer"],
            hallucinated_answer=item["hallucinated_answer"],
        ))
 
    random.Random(seed).shuffle(samples)
    return samples[:max_samples]

# ─────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────

class DatasetLoader:
    """Loads and normalises benchmark datasets."""

    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.seed = seed

    # ── public ──────────────────────────────────────────────

    def load_all(self) -> dict[str, List[BenchmarkSample]]:
        """Load every dataset listed in config. Returns {dataset_name: [samples]}."""
        result = {}
        for ds_cfg in self.config.get("datasets", []):
            if not ds_cfg.get("enabled", True):
                continue
            name   = ds_cfg["name"]
            source = ds_cfg.get("source", "hf")
            logger.debug(f"Loading dataset: {name}  (source={source})")
            try:
                if source == "hf":
                    samples = self._load_hf(ds_cfg)
                elif source == "synthetic":
                    n = ds_cfg.get("max_samples", 100)
                    if n > len(SYNTHETIC_CONTEXTS):
                        logger.warning(
                            f"{name}: max_samples is {n} but the fixture has "
                            f"{len(SYNTHETIC_CONTEXTS)} distinct items; they repeat"
                        )
                    samples = _make_synthetic(n, self.seed)
                elif source == "json":
                    samples = self._load_json(ds_cfg)
                elif source == "ragtruth":
                    samples = self._load_ragtruth(ds_cfg)
                elif source == "halubench":
                    samples = self._load_halubench(ds_cfg)
                else:
                    logger.warning(f"Unknown source '{source}' for dataset '{name}'")
                    continue

                ids = [s.sample_id for s in samples]
                if len(ids) != len(set(ids)):
                    raise ValueError(f"duplicate sample ids in '{name}'; paired comparisons would misalign")
                result[name] = samples
                logger.debug(f"{len(samples)} samples loaded from '{name}'")
            except Exception as exc:
                logger.error(f"Failed to load dataset '{name}': {exc}")

        return result

    # ── HuggingFace ─────────────────────────────────────────

    def _load_hf(self, cfg: dict) -> List[BenchmarkSample]:
        from datasets import load_dataset  # lazy import

        path    = cfg["hf_path"]
        subset  = cfg.get("hf_subset")
        split   = cfg.get("split", "test")
        n       = cfg.get("max_samples", 200)

        load_kwargs = {
            "split": split,
            "trust_remote_code": cfg.get("trust_remote_code", False),
        }
        if cfg.get("revision"):
            load_kwargs["revision"] = cfg["revision"]
        ds = load_dataset(path, subset, **load_kwargs)
        ds = ds.shuffle(seed=self.seed).select(range(min(n, len(ds))))

        samples = []
        name = cfg["name"]

        for i, row in enumerate(ds):
            try:
                sample = self._normalise_row(row, cfg, name, i)
                if sample:
                    samples.append(sample)
            except Exception as exc:
                logger.debug(f"Skipping row {i} in {name}: {exc}")

        return samples

    def _normalise_row(self, row: dict, cfg: dict, dataset_name: str, idx: int) -> Optional[BenchmarkSample]:
        """Map raw dataset columns to BenchmarkSample fields.

        right_answer / hallucinated_answer become the fixed labeled
        responses for detector validation. The reduction stage uses only
        question + context; the model writes its own answer there.
        """
        ctx_col         = cfg.get("context_col", "context")
        q_col           = cfg.get("question_col", "question")
        right_col       = cfg.get("right_answer_col", "right_answer")
        halluc_col      = cfg.get("hallucinated_answer_col", "hallucinated_answer")

        context             = self._get_text(row, ctx_col)
        question            = self._get_text(row, q_col)
        right_answer        = self._get_text(row, right_col)
        hallucinated_answer = self._get_text(row, halluc_col)

        # Some source datasets have no per-row question field (e.g. HaluEval's
        # summarization split: a document + a reference summary, no question).
        # `question_template` is local benchmark policy for that case, not
        # part of the original dataset — see METHOD_SOURCES.md.
        if not question and cfg.get("question_template"):
            question = cfg["question_template"]

        # Must have context and question at minimum
        if not context or not question:
            return None
 
        return BenchmarkSample(
            sample_id=f"{dataset_name}_{idx:05d}",
            dataset=dataset_name,
            question=question,
            context=context,
            right_answer=right_answer,
            hallucinated_answer=hallucinated_answer,
            metadata={k: v for k, v in row.items()
                      if k not in (ctx_col, q_col, right_col, halluc_col)},
        )

    def _get_text(self, row: dict, col: str) -> str:
        val = row.get(col, "")
        if isinstance(val, list):
            val = " ".join(str(v) for v in val)
        return str(val).strip()

    
# ── JSON ────────────────────────────────────────────────

    def _load_json(self, cfg: dict) -> List[BenchmarkSample]:
        """Load a `.json` dataset file.

        HaluEval's official files (qa_data.json, dialogue_data.json,
        summarization_data.json) are JSON Lines — one JSON object per line,
        not a single JSON array — despite the `.json` extension. Support both
        so this loader works on the real official files, not just a
        hand-built array fixture.
        """
        path = Path(cfg["path"])
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found (main.py downloads the official HaluEval "
                f"files automatically; see utils/resources.py)"
            )
        text = path.read_text(encoding="utf-8").strip()
        if text.startswith("["):
            data = json.loads(text)
        else:
            data = [json.loads(line) for line in text.splitlines() if line.strip()]

        # Seeded random subset of the whole file; sample_id keeps the
        # original 0-based line index so every case traces back to its row.
        indices = list(range(len(data)))
        random.Random(self.seed).shuffle(indices)
        limit = cfg.get("max_samples", 200)
        samples = []
        name = cfg["name"]
        for i in indices:
            if len(samples) >= limit:
                break
            sample = self._normalise_row(data[i], cfg, name, i)
            if sample:
                samples.append(sample)
        if len(samples) < limit:
            logger.warning(
                f"{name}: max_samples is {limit} but {path.name} has only "
                f"{len(samples)} usable rows; using all of them"
            )
        return samples

    def _pick(self, keys: list, limit: int, name: str, what: str) -> list:
        """Seeded random subset, same in every run."""
        keys = list(keys)
        random.Random(self.seed).shuffle(keys)
        if len(keys) < limit:
            logger.warning(f"{name}: max_samples is {limit} but only {len(keys)} {what} exist; using all")
        return keys[:limit]

    # ── RAGTruth ────────────────────────────────────────────────────────

    def _load_ragtruth(self, cfg: dict) -> List[BenchmarkSample]:
        """One sample per RAGTruth source (question/document/data record) of
        the configured task type, with every labeled model response to it.

        label = 1 when annotators marked at least one hallucinated span.
        Only the official `test` split is used, and responses with quality
        other than "good" (incorrect refusals, truncations) are skipped.
        question = RAGTruth's own task instruction without the context;
        context = the passages / news document / structured data.
        """
        responses_path, sources_path = Path(cfg["responses_path"]), Path(cfg["sources_path"])
        for path in (responses_path, sources_path):
            if not path.exists():
                raise FileNotFoundError(f"{path} not found (downloaded automatically by main.py)")
        task, split = cfg["task_type"], cfg.get("split", "test")
        sources = {}
        with sources_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["task_type"] == task:
                    sources[row["source_id"]] = row
        answers: dict = {}
        with responses_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["source_id"] in sources and row["split"] == split and row["quality"] == "good":
                    answers.setdefault(row["source_id"], []).append(row)

        name = cfg["name"]
        samples = []
        for source_id in self._pick(sorted(answers), cfg.get("max_samples", 200), name, "sources"):
            src = sources[source_id]
            info = src["source_info"]
            prompt = src["prompt"]
            if task == "QA":
                question, context = info["question"], info["passages"]
            elif task == "Summary":
                context = info
                question = prompt.split("\n", 1)[0].strip()          # "Summarize the following news within N words:"
            else:  # Data2txt: instruction, then the JSON record, then "Overview:"
                head, _, rest = prompt.partition("Structured data:")
                question = head.replace("Instruction:", "").strip()
                context = rest.rsplit("Overview:", 1)[0].strip()
            samples.append(BenchmarkSample(
                sample_id=f"{name}_{source_id}",
                dataset=name,
                question=question,
                context=context,
                right_answer="",
                hallucinated_answer="",
                metadata={"source_id": source_id, "task_type": task,
                          "ragtruth_source": src.get("source"), "original_prompt": prompt},
                labeled_answers=[
                    {"answer": r["response"], "label": int(bool(r["labels"])),
                     "source": r["model"], "answer_id": r["id"]}
                    for r in sorted(answers[source_id], key=lambda r: r["id"])
                ],
            ))
        return samples

    # ── HaluBench ───────────────────────────────────────────────────────

    def _load_halubench(self, cfg: dict) -> List[BenchmarkSample]:
        """One sample per HaluBench (passage, question) of the configured
        source_ds, with every PASS (label 0) / FAIL (label 1) answer to it."""
        import pandas as pd

        path = Path(cfg["path"])
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (downloaded automatically by main.py)")
        frame = pd.read_parquet(path)
        frame = frame[frame["source_ds"] == cfg["source_ds"]]
        # HaluBench's DROP FAIL answers include 272 Python list literals
        # ("['Rams', 'second', ...]") and no PASS answer has that form, so a
        # detector could separate the classes by format alone. Such answers
        # are excluded (both labels), and the count is logged.
        literal = frame["answer"].astype(str).str.strip().str.match(r"^\[.*\]$")
        if literal.any():
            logger.info(f"{cfg['name']}: excluded {int(literal.sum())} list-literal answers "
                        f"(format artifact: {dict(frame[literal]['label'].value_counts())})")
            frame = frame[~literal]
        groups = {key: group for key, group in frame.groupby(["passage", "question"], sort=True)}
        name = cfg["name"]
        samples = []
        for index, key in enumerate(self._pick(sorted(groups), cfg.get("max_samples", 200), name, "questions")):
            group = groups[key].sort_values("id")
            passage, question = key
            # Stable id from the question itself: HaluBench ids share long
            # prefixes (e.g. "financebench_id_…"), so a truncated id collides.
            digest = hashlib.sha1(f"{passage}\x00{question}".encode("utf-8")).hexdigest()[:12]
            samples.append(BenchmarkSample(
                sample_id=f"{name}_{digest}",
                dataset=name,
                question=str(question).strip(),
                context=str(passage).strip(),
                right_answer="",
                hallucinated_answer="",
                metadata={"source_ds": cfg["source_ds"]},
                labeled_answers=[
                    {"answer": str(row.answer), "label": int(row.label == "FAIL"),
                     "source": "halubench", "answer_id": row.id}
                    for row in group.itertuples(index=False)
                ],
            ))
        return samples

    @staticmethod
    def detection_cases(samples: List[BenchmarkSample]) -> List[dict]:
        """Create the fixed labeled pairs used for detector validation.

        A detector must be evaluated on these fixed responses before it is used
        to judge newly generated reducer outputs. Label 1 means hallucinated.
        """
        cases = []
        for sample in samples:
            if sample.right_answer:
                cases.append({
                    "case_id": f"{sample.sample_id}:factual",
                    "sample_id": sample.sample_id,
                    "dataset": sample.dataset,
                    "question": sample.question,
                    "context": sample.context,
                    "answer": sample.right_answer,
                    "label": 0,
                })
            if sample.hallucinated_answer:
                cases.append({
                    "case_id": f"{sample.sample_id}:hallucinated",
                    "sample_id": sample.sample_id,
                    "dataset": sample.dataset,
                    "question": sample.question,
                    "context": sample.context,
                    "answer": sample.hallucinated_answer,
                    "label": 1,
                })
            for item in sample.labeled_answers:
                if not str(item["answer"]).strip():
                    continue
                cases.append({
                    "case_id": f"{sample.sample_id}:{item.get('answer_id', len(cases))}",
                    "sample_id": sample.sample_id,
                    "dataset": sample.dataset,
                    "question": sample.question,
                    "context": sample.context,
                    "answer": item["answer"],
                    "label": int(item["label"]),
                    "answer_source": item.get("source"),
                })
        return cases

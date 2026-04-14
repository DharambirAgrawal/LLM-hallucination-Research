"""
Dataset loader for hallucination benchmarks.

Supports:
  - HaluEval       : pminervini/HaluEval  (qa_samples, summarization_samples, etc.)
  - RAGBench        : rungalileo/ragbench  (techqa, hotpotqa, etc.)
  - Synthetic       : Auto-generated QA pairs with planted hallucinations
"""

from __future__ import annotations

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
    answer:         str
    is_hallucinated: bool           # ground-truth label
    metadata:       dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sample_id":        self.sample_id,
            "dataset":          self.dataset,
            "question":         self.question,
            "context":          self.context,
            "answer":           self.answer,
            "is_hallucinated":  self.is_hallucinated,
            "metadata":         self.metadata,
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
    """Generate synthetic hallucination benchmark samples."""
    random.seed(seed)
    samples = []
    pool = SYNTHETIC_CONTEXTS * (max_samples // len(SYNTHETIC_CONTEXTS) + 1)
    pool = pool[:max_samples]

    for i, item in enumerate(pool):
        # Add one factual + one hallucinated per context
        samples.append(BenchmarkSample(
            sample_id=f"synthetic_{i:04d}_factual",
            dataset="synthetic",
            question=item["question"],
            context=item["context"],
            answer=item["factual_answer"],
            is_hallucinated=False,
        ))
        samples.append(BenchmarkSample(
            sample_id=f"synthetic_{i:04d}_hallucinated",
            dataset="synthetic",
            question=item["question"],
            context=item["context"],
            answer=item["hallucinated_answer"],
            is_hallucinated=True,
        ))

    random.shuffle(samples)
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
            name   = ds_cfg["name"]
            source = ds_cfg.get("source", "hf")
            logger.info(f"Loading dataset: {name}  (source={source})")
            try:
                if source == "hf":
                    samples = self._load_hf(ds_cfg)
                elif source == "synthetic":
                    samples = _make_synthetic(ds_cfg.get("max_samples", 100), self.seed)
                elif source == "json":
                    samples = self._load_json(ds_cfg)
                else:
                    logger.warning(f"Unknown source '{source}' for dataset '{name}'")
                    continue

                result[name] = samples
                logger.info(f"  ✓ {len(samples)} samples loaded from '{name}'")
            except Exception as exc:
                logger.error(f"  ✗ Failed to load '{name}': {exc}")

        return result

    # ── HuggingFace ─────────────────────────────────────────

    def _load_hf(self, cfg: dict) -> List[BenchmarkSample]:
        from datasets import load_dataset  # lazy import

        path    = cfg["hf_path"]
        subset  = cfg.get("hf_subset")
        split   = cfg.get("split", "test")
        n       = cfg.get("max_samples", 200)

        ds = load_dataset(path, subset, split=split, trust_remote_code=True)
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
        """Map raw row columns to BenchmarkSample fields."""
        ctx_col  = cfg.get("context_col", "context")
        q_col    = cfg.get("question_col", "question")
        ans_col  = cfg.get("answer_col", "answer")
        lbl_col  = cfg.get("label_col", "label")

        context  = self._get_text(row, ctx_col)
        question = self._get_text(row, q_col)
        answer   = self._get_text(row, ans_col)

        if not context or not question or not answer:
            return None

        # Determine ground-truth hallucination label
        is_hallucinated = self._parse_label(row, lbl_col, cfg)

        return BenchmarkSample(
            sample_id=f"{dataset_name}_{idx:05d}",
            dataset=dataset_name,
            question=question,
            context=context,
            answer=answer,
            is_hallucinated=is_hallucinated,
            metadata={k: v for k, v in row.items()
                      if k not in (ctx_col, q_col, ans_col, lbl_col)},
        )

    def _get_text(self, row: dict, col: str) -> str:
        val = row.get(col, "")
        if isinstance(val, list):
            val = " ".join(str(v) for v in val)
        return str(val).strip()

    def _parse_label(self, row: dict, lbl_col: str, cfg: dict) -> bool:
        raw = row.get(lbl_col)
        if raw is None:
            return False
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            threshold = cfg.get("label_threshold", 0.5)
            # adherence_score: high = factual → hallucinated = low score
            if "adherence" in lbl_col.lower():
                return float(raw) < threshold
            return float(raw) >= threshold
        s = str(raw).lower().strip()
        return s in {"yes", "true", "1", "hallucinated", "hallucination"}

    # ── JSON ────────────────────────────────────────────────

    def _load_json(self, cfg: dict) -> List[BenchmarkSample]:
        path = Path(cfg["path"])
        with open(path) as f:
            data = json.load(f)

        samples = []
        name = cfg["name"]
        for i, row in enumerate(data[: cfg.get("max_samples", 200)]):
            sample = self._normalise_row(row, cfg, name, i)
            if sample:
                samples.append(sample)
        return samples

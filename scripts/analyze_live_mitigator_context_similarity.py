"""Métricas léxicas auxiliares sobre una campaña Mistral del mitigador.

Compara únicamente ``context`` (texto realmente generado) con ``base`` (acción
catalogada asociada). No evalúa el campo operativo ``text``, porque este es
idéntico al catálogo por construcción. ROUGE-L y TF-IDF se interpretan como
fidelidad léxica descriptiva, nunca como corrección técnica o eficacia.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics
from typing import Any
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    return parser.parse_args()


def norm_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", str(text).casefold())
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return re.findall(r"[a-z0-9]+", normalized)


def lcs_len(left: list[str], right: list[str]) -> int:
    row = [0] * (len(right) + 1)
    for left_token in left:
        previous = 0
        for index, right_token in enumerate(right, start=1):
            current = row[index]
            row[index] = (
                previous + 1
                if left_token == right_token
                else max(row[index], row[index - 1])
            )
            previous = current
    return row[-1]


def rouge_l(candidate: str, reference: str) -> float:
    candidate_tokens = norm_tokens(candidate)
    reference_tokens = norm_tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    common = lcs_len(candidate_tokens, reference_tokens)
    precision = common / len(candidate_tokens)
    recall = common / len(reference_tokens)
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    args = parse_args()
    campaign_dir = args.campaign_dir.expanduser().resolve()
    cases = iter_jsonl(campaign_dir / "cases.jsonl")

    pairs: list[dict[str, Any]] = []
    for case in cases:
        attack_type = case["classification"]["attack_type"]
        for item in case["explanation"]["mitigation_items"]:
            if item.get("source") != "llm":
                continue
            base = str(item.get("base") or "").strip()
            context = str(item.get("context") or "").strip()
            if not base or not context:
                continue
            pairs.append(
                {
                    "case_id": case["case_id"],
                    "attack_type": attack_type,
                    "base": base,
                    "context": context,
                    "rouge_l": rouge_l(context, base),
                }
            )

    if not pairs:
        raise RuntimeError("La campaña no contiene contextos Mistral anclados")

    documents = [" ".join(norm_tokens(pair["context"])) for pair in pairs]
    documents.extend(" ".join(norm_tokens(pair["base"])) for pair in pairs)
    matrix = TfidfVectorizer().fit_transform(documents)
    count = len(pairs)
    similarities = cosine_similarity(matrix[:count], matrix[count:])
    for index, pair in enumerate(pairs):
        pair["tfidf_cosine"] = float(similarities[index, index])
        pair["identical_after_normalization"] = (
            norm_tokens(pair["context"]) == norm_tokens(pair["base"])
        )

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_type[pair["attack_type"]].append(pair)

    rows = []
    for attack_type, values in sorted(by_type.items()):
        rows.append(
            {
                "attack_type": attack_type,
                "anchored_contexts": len(values),
                "rouge_l_mean": statistics.fmean(
                    value["rouge_l"] for value in values
                ),
                "tfidf_cosine_mean": statistics.fmean(
                    value["tfidf_cosine"] for value in values
                ),
                "identical_contexts": sum(
                    value["identical_after_normalization"] for value in values
                ),
            }
        )

    output = {
        "protocol": {
            "candidate": "Mistral mitigation_items[].context",
            "reference": "the associated immutable catalog base",
            "rouge_l": "token-normalized LCS F1",
            "cosine": (
                "pair diagonal after one TF-IDF fit over all contexts and bases"
            ),
            "interpretation": (
                "auxiliary lexical fidelity only; not semantic correctness"
            ),
        },
        "global": {
            "anchored_contexts": len(pairs),
            "rouge_l_mean": statistics.fmean(pair["rouge_l"] for pair in pairs),
            "tfidf_cosine_mean": statistics.fmean(
                pair["tfidf_cosine"] for pair in pairs
            ),
            "identical_contexts": sum(
                pair["identical_after_normalization"] for pair in pairs
            ),
        },
        "by_attack_type": rows,
        "pairs": pairs,
    }
    (campaign_dir / "context_similarity.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Similitud léxica auxiliar entre contexto Mistral y base catalogada",
        "",
        (
            f"Media global: ROUGE-L {output['global']['rouge_l_mean']:.4f}; "
            f"coseno TF-IDF {output['global']['tfidf_cosine_mean']:.4f}."
        ),
        "",
        "| Tipo | Contextos | ROUGE-L | Coseno TF-IDF | Idénticos |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['attack_type']} | {row['anchored_contexts']} | "
            f"{row['rouge_l_mean']:.4f} | {row['tfidf_cosine_mean']:.4f} | "
            f"{row['identical_contexts']} |"
        )
    (campaign_dir / "context_similarity.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["global"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fase B de la validacion por dataset: manifiestos de muestreo con split limpio.

Genera, por dataset, un manifiesto JSONL con las filas muestreadas (fila cruda
incluida, para no depender de data/ en la maquina que estandariza), etiquetas
separadas de las features, deduplicacion previa al split y particiones 70/15/15
estratificadas y serializadas ANTES de cualquier estandarizacion.

Protocolo (espejo del TFM de referencia, decidido 2026-08-03):
- Multiclase: 500 registros por clase nativa (o maximo disponible, declarado).
- Binario: 5.900 normales + 5.900 ataques (o maximo disponible, declarado).
- Una fila puede servir a ambas tareas (la estandarizacion es por fila); el
  split se asigna UNA vez por fila, estratificado por clase, sin fugas.
- Dedup: hash del contenido de la fila EXCLUYENDO identificadores, timestamps
  y columnas target (evita casi-duplicados de rafaga entre train y test).
- TON-IoT se trata como UN dataset (muestra general balanceada por `type`
  sobre todos sus ficheros; el fichero de origen queda como metadato).
- URBAN_IOT no tiene etiquetas: muestra para validar solo la estandarizacion.

Uso:
    .venv\\Scripts\\python.exe scripts\\build_validation_manifests.py
    (opciones: --dataset edge_iiotset --seed 42 --out-dir artifacts/validation_2026)
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

csv.field_size_limit(10_000_000)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

BINARY_QUOTA_PER_SIDE = 5900
MULTICLASS_QUOTA = 500
OVERSAMPLE = 1.5  # margen para perdidas por dedup

# Columnas que NUNCA se envian al LLM (targets) y que tampoco entran al hash
TARGET_COLS = {
    "attack_label", "attack_type", "label", "type", "detailed-label",
    "detailed_label", "category", "subcategory", "attack",
}
# Columnas excluidas del hash de dedup (identificadores y tiempo: hacen
# unicas filas que son casi-duplicados funcionales)
ID_TIME_COLS = {
    "pkseqid", "seq", "uid", "ts", "stime", "ltime", "time", "timestamp",
    "date", "frame.time", "flow_id", "id", "node",
}


def row_hash(row: dict[str, str]) -> str:
    payload = "|".join(
        f"{k}={v}" for k, v in sorted(row.items())
        if k.strip().lower() not in TARGET_COLS | ID_TIME_COLS
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def row_digest(row: dict[str, str]) -> bytes:
    """Version compacta (16 bytes) del hash para los sets en memoria."""
    payload = "|".join(
        f"{k}={v}" for k, v in sorted(row.items())
        if k.strip().lower() not in TARGET_COLS | ID_TIME_COLS
    )
    return hashlib.sha256(payload.encode("utf-8")).digest()[:16]


class ClassReservoir:
    """Reservoir por clase sobre HASHES DISTINTOS (primera aparicion por hash).

    Muestrear filas y deduplicar despues infrarrepresenta las clases muy
    duplicadas (p. ej. DDoS de IoT-23: 14.395 filas, 225 unicas). Aqui el
    reservoir es uniforme sobre el conjunto de valores distintos: cada hash
    nuevo compite por un hueco; las repeticiones de un hash ya visto solo
    incrementan el contador de filas.
    """

    def __init__(self, quota: int, seed: int, key: str):
        self.quota = quota
        self.rng = random.Random(f"{seed}:{key}")
        self.seen = 0            # filas vistas (con repeticion)
        self.distinct = 0        # hashes distintos vistos
        self._hashes: set[bytes] = set()
        self.items: list[dict] = []

    def offer(self, item: dict, digest: bytes) -> None:
        self.seen += 1
        if digest in self._hashes:
            return
        self._hashes.add(digest)
        self.distinct += 1
        if len(self.items) < self.quota:
            item["content_hash"] = digest.hex()
            self.items.append(item)
        else:
            j = self.rng.randrange(self.distinct)
            if j < self.quota:
                item["content_hash"] = digest.hex()
                self.items[j] = item


def iter_csv(path: Path):
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            yield idx, {k: (v or "") for k, v in row.items() if k}


# ---------------------------------------------------------------------------
# Definicion por dataset: ficheros, etiqueta binaria y clase nativa
# ---------------------------------------------------------------------------

def edge_spec():
    path = DATA / "EDGE_IIOTSET" / "ML-EdgeIIoT-dataset.csv"
    def labeler(row):
        cls = row.get("Attack_type", "").strip()
        is_attack = row.get("Attack_label", "").strip() == "1"
        return cls or None, is_attack
    return {"name": "edge_iiotset", "files": [path], "labeler": labeler, "normal_class": "Normal"}


def ton_spec():
    files = sorted(
        Path(p) for p in glob.glob(str(DATA / "TON_IOT/Train_Test_datasets/**/*.csv"), recursive=True)
    )
    def labeler(row):
        cls = row.get("type", "").strip().lower()
        if not cls:
            return None, None
        return cls, cls != "normal"
    return {"name": "ton_iot", "files": files, "labeler": labeler, "normal_class": "normal"}


def iot23_spec():
    files = sorted(
        Path(p) for p in glob.glob(str(DATA / "IOT23/csv/CTU-*.csv"))
    )
    def labeler(row):
        binary = row.get("label", "").strip().lower()
        if binary not in {"benign", "malicious"}:
            return None, None
        detail = row.get("detailed-label", "").strip()
        cls = "benign" if binary == "benign" or detail in {"-", ""} else detail
        return cls, binary == "malicious"
    return {"name": "iot23", "files": files, "labeler": labeler, "normal_class": "benign"}


def bot_iot_spec():
    files = sorted(
        Path(p) for p in glob.glob(str(DATA / "BOT_IOT/UNSW_2018_IoT_Botnet_Full5pc_*.csv"))
    )
    def labeler(row):
        cls = row.get("category", "").strip()
        if not cls:
            return None, None
        return cls, row.get("attack", "").strip() == "1"
    return {"name": "bot_iot", "files": files, "labeler": labeler, "normal_class": "Normal"}


def urban_spec():
    path = DATA / "URBAN_IOT" / "original_data.csv"
    def labeler(row):
        return "unlabeled", None  # sin etiquetas: solo validacion de estandarizacion
    return {"name": "urban_iot", "files": [path], "labeler": labeler,
            "normal_class": None, "standardization_only": True,
            "multiclass_quota": 1000}


SPECS = {
    "edge_iiotset": edge_spec,
    "ton_iot": ton_spec,
    "iot23": iot23_spec,
    "bot_iot": bot_iot_spec,
    "urban_iot": urban_spec,
}


# ---------------------------------------------------------------------------
# Construccion del manifiesto de un dataset
# ---------------------------------------------------------------------------

def build_manifest(spec: dict, seed: int, out_dir: Path) -> dict:
    name = spec["name"]
    normal_class = spec.get("normal_class")
    std_only = spec.get("standardization_only", False)
    mc_quota = spec.get("multiclass_quota", MULTICLASS_QUOTA)

    # 1) Reservoir por clase sobre hashes distintos (dedup durante el muestreo)
    reservoirs: dict[str, ClassReservoir] = {}
    for path in spec["files"]:
        rel = str(path.relative_to(REPO))
        for idx, row in iter_csv(path):
            cls, is_attack = spec["labeler"](row)
            if cls is None:
                continue
            if cls not in reservoirs:
                base = BINARY_QUOTA_PER_SIDE if cls == normal_class else mc_quota
                reservoirs[cls] = ClassReservoir(int(base * OVERSAMPLE) + 50, seed, f"{name}:{cls}")
            reservoirs[cls].offer({
                "source_file": rel, "row_id": idx, "row": row,
                "class": cls, "is_attack": is_attack,
            }, row_digest(row))

    availability = {
        cls: {"rows": r.seen, "distinct": r.distinct}
        for cls, r in reservoirs.items()
    }

    # 2) Dedup residual ENTRE clases (mismo contenido con etiquetas distintas)
    seen_hashes: set[str] = set()
    pool: dict[str, list[dict]] = defaultdict(list)
    for cls, r in reservoirs.items():
        for item in r.items:
            h = item["content_hash"]
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            pool[cls].append(item)

    rng = random.Random(f"{seed}:{name}:select")
    for items in pool.values():
        rng.shuffle(items)

    # 3) Cuotas finales
    selected: list[dict] = []
    attack_classes = [c for c in pool if c != normal_class and not std_only]
    if std_only:
        cls = next(iter(pool))
        for item in pool[cls][:mc_quota]:
            item["tasks"] = ["standardization"]
            selected.append(item)
    else:
        # multiclase: hasta 500 por clase (incluida la normal)
        for cls, items in pool.items():
            for item in items[:mc_quota]:
                item["tasks"] = ["multiclass"]
                selected.append(item)
        # binario: completar ataques hasta 5.900 repartiendo entre clases con excedente
        chosen_ids = {id(x) for x in selected}
        n_attack = sum(1 for x in selected if x["is_attack"])
        extra_needed = max(0, min(BINARY_QUOTA_PER_SIDE,
                                  sum(len(pool[c]) for c in attack_classes)) - n_attack)
        round_robin = [c for c in attack_classes if len(pool[c]) > mc_quota]
        offsets = {c: mc_quota for c in round_robin}
        while extra_needed > 0 and round_robin:
            for c in list(round_robin):
                if extra_needed <= 0:
                    break
                if offsets[c] < len(pool[c]):
                    item = pool[c][offsets[c]]
                    offsets[c] += 1
                    item["tasks"] = ["binary"]
                    selected.append(item)
                    chosen_ids.add(id(item))
                    extra_needed -= 1
                else:
                    round_robin.remove(c)
        # binario: normales hasta 5.900 (los 500 de multiclase cuentan)
        if normal_class in pool:
            n_normal = sum(1 for x in selected if x["class"] == normal_class)
            for item in pool[normal_class][mc_quota:]:
                if n_normal >= BINARY_QUOTA_PER_SIDE:
                    break
                item["tasks"] = ["binary"]
                selected.append(item)
                n_normal += 1
        # marcar doble uso: toda fila de ataque/normal participa en binario
        for item in selected:
            if "multiclass" in item["tasks"] and item["is_attack"] is not None:
                if item["is_attack"] or item["class"] == normal_class:
                    item["tasks"] = sorted(set(item["tasks"]) | {"binary"})

    # equilibrio binario final: recortar el lado sobrante (ataques > 5.900 por multiclase)
    if not std_only:
        att = [x for x in selected if x["is_attack"] and "binary" in x["tasks"]]
        if len(att) > BINARY_QUOTA_PER_SIDE:
            rng.shuffle(att)
            for item in att[BINARY_QUOTA_PER_SIDE:]:
                item["tasks"] = [t for t in item["tasks"] if t != "binary"]

    # 4) Split 70/15/15 estratificado por clase, asignado una vez por fila
    by_class: dict[str, list[dict]] = defaultdict(list)
    for item in selected:
        by_class[item["class"]].append(item)
    for cls, items in by_class.items():
        srng = random.Random(f"{seed}:{name}:{cls}:split")
        srng.shuffle(items)
        n = len(items)
        n_train, n_val = int(n * 0.70), int(n * 0.15)
        for i, item in enumerate(items):
            item["split"] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")

    # 5) Serializar: fila cruda SIN columnas target (targets aparte)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{name}_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for item in selected:
            row_features = {
                k: v for k, v in item["row"].items()
                if k.strip().lower() not in TARGET_COLS
            }
            fh.write(json.dumps({
                "manifest_id": f"{name}::{item['source_file']}::{item['row_id']}",
                "dataset": name,
                "source_file": item["source_file"],
                "row_id": item["row_id"],
                "content_hash": item["content_hash"],
                "split": item.get("split"),
                "tasks": item["tasks"],
                "target": {"class": item["class"], "is_attack": item["is_attack"]},
                "row": row_features,
            }, ensure_ascii=False) + "\n")

    summary = {
        "dataset": name,
        "rows": len(selected),
        "availability_per_class": availability,
        "selected_per_class": {c: len(v) for c, v in by_class.items()},
        "split_counts": dict(Counter(x["split"] for x in selected if x.get("split"))),
        "binary_counts": {
            "attack": sum(1 for x in selected if x["is_attack"] and "binary" in x["tasks"]),
            "normal": sum(1 for x in selected if x["is_attack"] is False and "binary" in x["tasks"]),
        },
        "duplicates_skipped_while_sampling": sum(r.seen - r.distinct for r in reservoirs.values()),
        "cross_class_dedup_discarded": sum(len(r.items) for r in reservoirs.values()) - sum(len(v) for v in pool.values()),
        "manifest": str(manifest_path.relative_to(REPO)),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(SPECS), action="append",
                        help="repetible; por defecto todos")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="artifacts/validation_2026/manifests")
    args = parser.parse_args()

    out_dir = REPO / args.out_dir
    targets = args.dataset or sorted(SPECS)
    summaries = []
    for name in targets:
        print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] muestreando {name}...")
        summary = build_manifest(SPECS[name](), args.seed, out_dir)
        summaries.append(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    global_path = out_dir / "summary.json"
    with open(global_path, "w", encoding="utf-8") as fh:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(),
                   "seed": args.seed, "datasets": summaries}, fh, indent=2, ensure_ascii=False)
    print(f"Resumen global: {global_path}")
    total = sum(s["rows"] for s in summaries)
    print(f"TOTAL filas a estandarizar: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

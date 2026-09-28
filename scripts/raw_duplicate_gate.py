#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATASET = "lmarena-ai/arena-human-preference-140k"
REVISION = "a9cb587ee0906192dc1fc5e51778282f36c6bf35"
SHAS = [
    "7b871a964dc9b238fe0d59f817dbe1a7a56934e23ec4d989f86e6891e5a891cf",
    "0cd29a6e3261159f41a69b28cd7f08c0bbbc1ef08ba18438d386a1b0c144af8a",
    "c633264fe8f0fa26cc9aa49961a75dc428791f70c7a476cbed766963b88bee24",
    "5bcf04c9f4488006a4a7ee944d886ee123e118b7f7d2db04b4ff6dbd8125e6b0",
    "2c50b4b3f6a5c02d660f5a70d5006332ddc00f5eb21f8e4d92bde3856647969b",
    "de59f74dce7477beaec55936b31a88537b93c5cbb4b483cfc6879a6862ce5e3b",
    "1a32a1c6934ebcaa3019643a08fb57563ee0a9be282d3222def3c572a49e9f4c",
]
EXPECTED_ROWS = [19377, 19377, 19376, 19376, 19376, 19376, 19376]
EXPECTED_MESSAGES = [34130, 33803, 33126, 33793, 33555, 34047, 33415]
EXPECTED_TOTAL_ROWS = 135634
EXPECTED_TOTAL_MESSAGES = 235869


def text_of_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def human_user_texts(conv):
    out = []
    for item in conv or []:
        if not isinstance(item, dict):
            continue
        msg = item.get("user") if isinstance(item.get("user"), dict) else item
        if msg.get("role") != "user":
            continue
        text = text_of_content(msg.get("content")).strip()
        if text:
            out.append(text)
    return out


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def near_score(a, b):
    if not a or not b:
        return None
    shorter, longer = sorted((len(a), len(b)))
    if shorter / longer < 0.97:
        return None
    m = SequenceMatcher(None, a, b, autojunk=False)
    if m.real_quick_ratio() < 0.97 or m.quick_ratio() < 0.97:
        return None
    score = m.ratio()
    return score if score >= 0.97 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    target_doc = json.loads(Path(args.targets).read_text(encoding="utf-8"))
    if target_doc.get("dataset") != DATASET or target_doc.get("revision") != REVISION:
        raise RuntimeError("target dataset or revision mismatch")

    targets = {}
    for item in target_doc.get("targets", []):
        rid = str(item["sourceRowId"])
        ti = int(item["turnIndex"])
        key = f"{rid}:{ti}"
        if key in targets:
            raise RuntimeError("duplicate target key")
        targets[key] = {"sourceRowId": rid, "turnIndex": ti}

    if not targets:
        raise RuntimeError("no targets")

    entries = []
    length_buckets = defaultdict(list)
    target_recovery = {}
    shard_results = []
    total_rows = 0
    total_messages = 0

    for si in range(7):
        fn = f"data/train-{si:05d}-of-00007.parquet"
        local = hf_hub_download(
            repo_id=DATASET,
            repo_type="dataset",
            filename=fn,
            revision=REVISION,
        )
        actual_sha = sha256_file(local)
        if actual_sha != SHAS[si]:
            raise RuntimeError(f"shard sha mismatch: {si}")

        pf = pq.ParquetFile(local)
        shard_rows = 0
        shard_messages = 0

        for batch in pf.iter_batches(
            batch_size=512,
            columns=["id", "evaluation_session_id", "language", "full_conversation"],
        ):
            for row in batch.to_pylist():
                row_index = shard_rows
                shard_rows += 1
                rid = str(row.get("id") or "")
                sid = str(row.get("evaluation_session_id") or "")
                language = row.get("language")
                users = human_user_texts(row.get("full_conversation"))

                for ti, text in enumerate(users):
                    n = norm(text)
                    entry = {
                        "sourceRowId": rid,
                        "evaluationSessionId": sid,
                        "turnIndex": ti,
                        "parquetShard": fn,
                        "rowIndexInShard": row_index,
                        "norm": n,
                    }
                    idx = len(entries)
                    entries.append(entry)
                    length_buckets[len(n)].append(idx)
                    shard_messages += 1

                    key = f"{rid}:{ti}"
                    if key in targets:
                        if key in target_recovery:
                            raise RuntimeError(f"target recovered more than once: {key}")
                        target_recovery[key] = {
                            "sourceRowId": rid,
                            "evaluationSessionId": sid,
                            "turnIndex": ti,
                            "sourceLanguage": language,
                            "parquetShard": fn,
                            "rowIndexInShard": row_index,
                            "normalizedSha256": hashlib.sha256(n.encode("utf-8")).hexdigest(),
                            "normalizedLength": len(n),
                            "norm": n,
                        }

        if shard_rows != EXPECTED_ROWS[si]:
            raise RuntimeError(f"row count mismatch: shard {si}")
        if shard_messages != EXPECTED_MESSAGES[si]:
            raise RuntimeError(f"message count mismatch: shard {si}")

        total_rows += shard_rows
        total_messages += shard_messages
        shard_results.append({
            "parquetShard": fn,
            "sha256": actual_sha,
            "rows": shard_rows,
            "humanUserMessages": shard_messages,
        })

    if total_rows != EXPECTED_TOTAL_ROWS or total_messages != EXPECTED_TOTAL_MESSAGES:
        raise RuntimeError("global row or message count mismatch")
    if set(target_recovery) != set(targets):
        missing = sorted(set(targets) - set(target_recovery))
        raise RuntimeError("missing target keys: " + ",".join(missing))

    results = []
    for key in sorted(targets):
        target = target_recovery[key]
        tn = target.pop("norm")
        tlen = len(tn)
        self_matches = 0
        exact_other = []
        near_other = []

        exact_indices = length_buckets.get(tlen, [])
        for idx in exact_indices:
            e = entries[idx]
            if e["norm"] != tn:
                continue
            if e["sourceRowId"] == target["sourceRowId"] and e["turnIndex"] == target["turnIndex"]:
                self_matches += 1
            else:
                exact_other.append({
                    "sourceRowId": e["sourceRowId"],
                    "evaluationSessionId": e["evaluationSessionId"],
                    "turnIndex": e["turnIndex"],
                    "parquetShard": e["parquetShard"],
                    "rowIndexInShard": e["rowIndexInShard"],
                    "similarity": 1.0,
                })

        min_len = max(1, math.ceil(tlen * 0.97))
        max_len = max(min_len, math.floor(tlen / 0.97))
        exact_keys = {
            (x["sourceRowId"], x["turnIndex"], x["parquetShard"], x["rowIndexInShard"])
            for x in exact_other
        }

        for size in range(min_len, max_len + 1):
            for idx in length_buckets.get(size, []):
                e = entries[idx]
                marker = (e["sourceRowId"], e["turnIndex"], e["parquetShard"], e["rowIndexInShard"])
                if marker in exact_keys:
                    continue
                if e["sourceRowId"] == target["sourceRowId"] and e["turnIndex"] == target["turnIndex"]:
                    continue
                if e["norm"] == tn:
                    continue
                score = near_score(tn, e["norm"])
                if score is None:
                    continue
                near_other.append({
                    "sourceRowId": e["sourceRowId"],
                    "evaluationSessionId": e["evaluationSessionId"],
                    "turnIndex": e["turnIndex"],
                    "parquetShard": e["parquetShard"],
                    "rowIndexInShard": e["rowIndexInShard"],
                    "similarity": round(score, 6),
                })

        status = "PASS"
        if self_matches != 1:
            status = "FAIL_SELF_MATCH"
        elif exact_other:
            status = "EXCLUDED_EXACT_DUPLICATE"
        elif near_other:
            status = "EXCLUDED_NEAR97_DUPLICATE"

        results.append({
            **target,
            "selfMatches": self_matches,
            "exactOtherCount": len(exact_other),
            "near97OtherCount": len(near_other),
            "exactOther": exact_other[:50],
            "near97Other": sorted(near_other, key=lambda x: -x["similarity"])[:50],
            "duplicateGateStatus": status,
        })

    report = {
        "schemaVersion": "public-raw-duplicate-gate-v1",
        "dataset": DATASET,
        "revision": REVISION,
        "assistantOutputsUsed": False,
        "humanUserTextOnly": True,
        "rawRowsScanned": total_rows,
        "fullConversationHumanUserMessagesScanned": total_messages,
        "normalization": "NFKC + whitespace collapse + casefold",
        "nearDuplicateThreshold": 0.97,
        "nearDuplicateAlgorithm": "difflib.SequenceMatcher ratio with length, real_quick_ratio and quick_ratio prefilters",
        "shards": shard_results,
        "targets": results,
        "passCount": sum(x["duplicateGateStatus"] == "PASS" for x in results),
        "excludedCount": sum(x["duplicateGateStatus"] != "PASS" for x in results),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "targets": len(results),
        "passed": report["passCount"],
        "excluded": report["excludedCount"],
        "rawRowsScanned": total_rows,
        "humanUserMessagesScanned": total_messages,
        "shardsShaVerified": True,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()

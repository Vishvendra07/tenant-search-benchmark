"""Compare lexical and feature-vector retrieval with tenant-scoped candidates."""

import argparse
import hashlib
import json
import math
import re
import sqlite3
import time
from collections import Counter

DOCS = [
    ("north", "N-1", "Prepare bank statements for the funding declaration"),
    ("north", "N-2", "Submit academic transcripts and grade reports"),
    ("north", "N-3", "Find the location of the intake appointment"),
    ("north", "N-4", "Update your application deadline or start date"),
    ("harbor", "H-1", "Funding declaration and proof of money"),
    ("harbor", "H-2", "Cancel an appointment and rebook an interview"),
    ("harbor", "H-3", "Submit your passport and identity materials"),
    ("harbor", "H-4", "Tuition payment schedule and deposit dates"),
]
QUERIES = [
    ("north", "proof of money", "N-1"),
    ("north", "grade report", "N-2"),
    ("north", "intake appointment", "N-3"),
    ("north", "when does my program begin", "N-4"),
    ("harbor", "funding declaration", "H-1"),
    ("harbor", "move my interview", "H-2"),
    ("harbor", "passport identity", "H-3"),
    ("harbor", "deposit dates", "H-4"),
]
PHRASES = {
    "proof of money": "bank funding statement",
    "when does my program begin": "start date deadline",
    "move my interview": "rebook appointment",
    "grade report": "grade reports",
}
STOP = {"the", "of", "and", "or", "to", "your", "my", "an", "a", "is", "for"}


def tokens(text):
    return [term for term in re.findall(r"[a-z0-9]+", text.lower()) if term not in STOP]


def feature_vector(text):
    expanded = text.lower()
    for phrase, replacement in PHRASES.items():
        expanded = expanded.replace(phrase, replacement)
    terms = tokens(expanded)
    grams = [*terms, *[a + " " + b for a, b in zip(terms, terms[1:])]]
    return Counter(
        int.from_bytes(hashlib.blake2b(term.encode(), digest_size=4).digest(), "big") % 4096
        for term in grams
    )


def cosine(a, b):
    denominator = math.sqrt(sum(x * x for x in a.values()) * sum(x * x for x in b.values()))
    return sum(x * b[key] for key, x in a.items()) / denominator if denominator else 0


def setup(db):
    db.execute("CREATE TABLE docs (tenant TEXT, id TEXT, body TEXT, PRIMARY KEY(tenant,id))")
    db.executemany("INSERT INTO docs VALUES (?,?,?)", DOCS)


def search(db, tenant, query, mode, limit=3):
    result = []
    vector = feature_vector(query)
    query_terms = set(tokens(query))
    for doc_id, body in db.execute("SELECT id,body FROM docs WHERE tenant=?", (tenant,)):
        score = (
            cosine(vector, feature_vector(body))
            if mode == "vector"
            else len(query_terms.intersection(tokens(body)))
        )
        if score > 0:
            result.append((score, doc_id))
    return [doc_id for _, doc_id in sorted(result, key=lambda item: (-item[0], item[1]))[:limit]]


def dense(text):
    sparse = feature_vector(text)
    return json.dumps([sparse[i] for i in range(4096)], separators=(",", ":"))


def compare_postgres(dsn):
    import psycopg

    with psycopg.connect(dsn, connect_timeout=5) as db:
        db.execute("CREATE EXTENSION IF NOT EXISTS vector")
        db.execute(
            "CREATE TEMP TABLE vector_docs (tenant text, id text, features vector(4096), PRIMARY KEY(tenant,id))"
        )
        with db.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO vector_docs VALUES (%s,%s,%s::vector)",
                [(tenant, doc_id, dense(body)) for tenant, doc_id, body in DOCS],
            )
        runs = []
        for tenant, query, expected in QUERIES:
            vector = dense(query)
            rows = db.execute(
                """SELECT id FROM (SELECT id, features <=> %s::vector AS distance
                FROM vector_docs WHERE tenant=%s) ranked WHERE distance < 1 ORDER BY distance,id LIMIT 3""",
                (vector, tenant),
            ).fetchall()
            runs.append(
                {
                    "tenant": tenant,
                    "query": query,
                    "expected": expected,
                    "top3": [row[0] for row in rows],
                }
            )
        return {
            "engine": "pgvector exact",
            "recall_at_3": sum(row["expected"] in row["top3"] for row in runs) / len(runs),
            "queries": runs,
        }


def demo():
    with sqlite3.connect(":memory:") as db:
        setup(db)
        report = {}
        for mode in ("keyword", "vector"):
            runs, latencies = [], []
            for tenant, query, expected in QUERIES:
                start = time.perf_counter_ns()
                ids = search(db, tenant, query, mode)
                latencies.append((time.perf_counter_ns() - start) / 1_000_000)
                runs.append({"tenant": tenant, "query": query, "expected": expected, "top3": ids})
            report[mode] = {
                "recall_at_3": sum(row["expected"] in row["top3"] for row in runs) / len(runs),
                "p95_ms": round(sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1], 4),
                "queries": runs,
            }
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pg-dsn", help="Optional connection string for this repo’s local pgvector container"
    )
    args = parser.parse_args()
    result = demo()
    if args.pg_dsn:
        result["pgvector"] = compare_postgres(args.pg_dsn)
        if [x["top3"] for x in result["pgvector"]["queries"]] != [
            x["top3"] for x in result["vector"]["queries"]
        ]:
            raise RuntimeError("Application ranking and exact SQL ranking disagree")
    print(json.dumps(result, indent=2))

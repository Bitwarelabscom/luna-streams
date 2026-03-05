#!/usr/bin/env python3
"""Export training data from Luna Chat and MemoryCore databases.

Runs SQL queries via docker exec against the PostgreSQL containers
and outputs JSON files to training/data_prep/raw/.

Usage:
    python -m training.data_prep.export_luna_data
"""

import json
import subprocess
import sys
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"

# (filename, container, database, user, query, expected_approx_rows)
QUERIES = [
    (
        "messages.json",
        "luna-postgres",
        "luna_chat",
        "luna",
        """
        SELECT id, session_id, role, content, tokens_used, model, created_at
        FROM messages
        ORDER BY created_at
        """,
        3145,
    ),
    (
        "message_embeddings.json",
        "luna-postgres",
        "luna_chat",
        "luna",
        """
        SELECT message_id, emotional_valence, attention_score
        FROM message_embeddings
        ORDER BY message_id
        """,
        1840,
    ),
    (
        "sessions.json",
        "luna-postgres",
        "luna_chat",
        "luna",
        """
        SELECT id, title, mode, created_at, rolling_summary
        FROM sessions
        ORDER BY created_at
        """,
        244,
    ),
    (
        "user_facts.json",
        "luna-postgres",
        "luna_chat",
        "luna",
        """
        SELECT category, fact_key, fact_value, confidence
        FROM user_facts
        ORDER BY category, fact_key
        """,
        360,
    ),
    (
        "memory_nodes.json",
        "memorycore-postgres",
        "memorycore",
        "memorycore",
        """
        SELECT id, node_type, node_label, origin, activation_strength,
               emotional_intensity, created_at
        FROM memory_nodes
        ORDER BY created_at
        """,
        5388,
    ),
    (
        "memory_edges.json",
        "memorycore-postgres",
        "memorycore",
        "memorycore",
        """
        SELECT source_node_id, target_node_id, edge_type, weight, activation_count
        FROM memory_edges
        ORDER BY activation_count DESC
        LIMIT 50000
        """,
        50000,
    ),
    (
        "session_summaries.json",
        "memorycore-postgres",
        "memorycore",
        "memorycore",
        """
        SELECT session_id, duration, interaction_count,
               patterns, breakthroughs, timestamp
        FROM session_summaries
        ORDER BY timestamp
        """,
        486,
    ),
]


def run_query(container: str, database: str, user: str, query: str) -> list[dict]:
    """Execute a SQL query via docker exec and return results as dicts."""
    # Use psql with JSON output format
    sql = f"""
    SELECT json_agg(t) FROM ({query.strip().rstrip(';')}) t;
    """

    cmd = [
        "docker", "exec", container,
        "psql", "-U", user, "-d", database,
        "-t", "-A",  # tuples only, unaligned
        "-c", sql,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        return []

    output = result.stdout.strip()
    if not output or output == "null" or output == "":
        return []

    try:
        data = json.loads(output)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        print(f"  First 200 chars: {output[:200]}")
        return []


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Luna Streams - Training Data Export")
    print("=" * 50)

    total_rows = 0
    for filename, container, database, user, query, expected in QUERIES:
        print(f"\nExporting {filename}...")
        rows = run_query(container, database, user, query)
        count = len(rows)
        total_rows += count

        out_path = RAW_DIR / filename
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2, default=str)

        status = "OK" if count > 0 else "EMPTY"
        delta = f" (expected ~{expected})" if abs(count - expected) > expected * 0.2 else ""
        print(f"  {status}: {count} rows{delta} -> {out_path}")

    print(f"\n{'=' * 50}")
    print(f"Total: {total_rows} rows exported to {RAW_DIR}")


if __name__ == "__main__":
    main()

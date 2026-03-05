#!/usr/bin/env python3
"""Build event sequences from raw exported data.

Converts raw DB exports into chronological StructuredEvent sequences
using the same compact encoding format as the live system.

Usage:
    python -m training.data_prep.build_event_sequences
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
OUTPUT = Path(__file__).parent / "sequences.jsonl"


def load_json(filename: str) -> list[dict]:
    path = RAW_DIR / filename
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return []
    with open(path) as f:
        return json.load(f)


def compact_encode(event: dict) -> str:
    """Compact text encoding matching UserModelStream.event_to_tokens().

    Format: type_code source entities topics sentiment importance summary_snippet
    """
    type_codes = {
        "memory_entry": "mem_e",
        "entity_update": "ent_u",
        "edge_update": "edge_u",
        "conversation_meta": "conv_m",
    }
    source_codes = {
        "conversation": "conv",
        "agent_dialogue": "agent",
        "neuralsleep": "sleep",
        "system": "sys",
    }

    parts = [
        type_codes.get(event.get("event_type", ""), "unk"),
        source_codes.get(event.get("source", ""), "?"),
    ]

    entities = event.get("entities", [])
    if entities:
        parts.append(",".join(entities[:3]))

    topics = event.get("topic_tags", [])
    if topics:
        parts.append(",".join(topics[:2]))

    sentiment = event.get("sentiment", 0.0)
    if sentiment != 0.0:
        parts.append(f"{sentiment:.1f}")

    importance = event.get("importance", 0.5)
    if importance != 0.5:
        parts.append(f"{importance:.1f}")

    summary = event.get("summary", "")
    if summary:
        parts.append(summary[:30].strip())

    return " ".join(parts)


def build_node_lookup(nodes: list[dict]) -> dict[int, str]:
    """Build ID -> label lookup for memory nodes."""
    return {n["id"]: n.get("node_label", f"node_{n['id']}") for n in nodes}


def parse_timestamp(ts) -> datetime:
    """Parse various timestamp formats. Returns naive UTC datetime."""
    if not ts:
        return datetime.min
    if isinstance(ts, datetime):
        # Strip tzinfo to keep everything naive-UTC comparable
        return ts.replace(tzinfo=None)
    if isinstance(ts, str):
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f%z",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ]:
            try:
                dt = datetime.strptime(ts, fmt)
                return dt.replace(tzinfo=None)
            except ValueError:
                continue
    return datetime.min


def main():
    print("Luna Streams - Build Event Sequences")
    print("=" * 50)

    # Load raw data
    messages = load_json("messages.json")
    embeddings = load_json("message_embeddings.json")
    sessions = load_json("sessions.json")
    nodes = load_json("memory_nodes.json")
    edges = load_json("memory_edges.json")
    summaries = load_json("session_summaries.json")
    user_facts = load_json("user_facts.json")

    # Build lookups
    embedding_map = {e["message_id"]: e for e in embeddings}
    node_labels = build_node_lookup(nodes)

    print(f"\nLoaded: {len(messages)} messages, {len(embeddings)} embeddings, "
          f"{len(nodes)} nodes, {len(edges)} edges, {len(summaries)} summaries")

    # Convert messages to events
    all_events = []

    for msg in messages:
        emb = embedding_map.get(msg.get("id"))
        sentiment = emb.get("emotional_valence", 0.0) if emb else 0.0
        importance = emb.get("attention_score", 0.5) if emb else 0.5

        # Extract entities from content (simple heuristic - capitalized words)
        content = msg.get("content", "")
        summary = content[:60].strip() if content else ""

        event = {
            "timestamp": msg.get("created_at", ""),
            "event_type": "memory_entry",
            "source": "conversation",
            "session_id": msg.get("session_id"),
            "entities": [],
            "topic_tags": [],
            "sentiment": float(sentiment) if sentiment else 0.0,
            "importance": float(importance) if importance else 0.5,
            "summary": summary,
        }
        all_events.append(event)

    # Convert memory nodes to entity_update events
    for node in nodes:
        event = {
            "timestamp": node.get("created_at", ""),
            "event_type": "entity_update",
            "source": node.get("origin", "system"),
            "session_id": None,
            "entities": [node.get("node_label", "")],
            "topic_tags": [node.get("node_type", "")],
            "sentiment": float(node.get("emotional_intensity", 0.0) or 0.0),
            "importance": float(node.get("activation_strength", 0.5) or 0.5),
            "summary": "",
        }
        # Normalize source
        if event["source"] not in ("conversation", "agent_dialogue", "neuralsleep", "system"):
            event["source"] = "system"
        all_events.append(event)

    # Convert top edges to edge_update events
    for edge in edges[:10000]:  # limit to top 10K by activation
        src_label = node_labels.get(edge.get("source_node_id"), "unknown")
        tgt_label = node_labels.get(edge.get("target_node_id"), "unknown")
        event = {
            "timestamp": "",  # edges don't have timestamps, will be sorted last
            "event_type": "edge_update",
            "source": "system",
            "session_id": None,
            "entities": [src_label, tgt_label],
            "topic_tags": [edge.get("edge_type", "related")],
            "sentiment": 0.0,
            "importance": min(float(edge.get("weight", 0.5) or 0.5), 1.0),
            "summary": "",
        }
        all_events.append(event)

    # Sort by timestamp
    all_events.sort(key=lambda e: parse_timestamp(e.get("timestamp", "")))

    print(f"Total events: {len(all_events)}")

    # Group by session (session_id) for session-based events, rest go to "global"
    session_groups = defaultdict(list)
    for event in all_events:
        sid = event.get("session_id") or "global"
        session_groups[sid].append(event)

    # Build sequences
    sequences = []
    for sid, events in session_groups.items():
        tokens = [compact_encode(e) for e in events]
        sequences.append({
            "session_id": str(sid),
            "sequence": tokens,
            "events": events,
            "length": len(events),
        })

    # Write JSONL
    with open(OUTPUT, "w") as f:
        for seq in sequences:
            f.write(json.dumps(seq, default=str) + "\n")

    # Stats
    lengths = [s["length"] for s in sequences]
    type_counts = defaultdict(int)
    for e in all_events:
        type_counts[e["event_type"]] += 1

    print(f"\nSequences: {len(sequences)}")
    print(f"Avg length: {sum(lengths) / len(lengths):.1f}")
    print(f"Min/Max length: {min(lengths)}/{max(lengths)}")
    print(f"Event types: {dict(type_counts)}")
    print(f"\nOutput: {OUTPUT}")


if __name__ == "__main__":
    main()

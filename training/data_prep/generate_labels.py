#!/usr/bin/env python3
"""Generate training labels using Qwen as a labeling oracle.

Reads sequences.jsonl and prompts Qwen 3.5:9b to produce:
- emotional_valence: float [-1, 1]
- focus_topics: list of topic indices (from 50-class vocabulary)
- next_event_type: index (0=mem_e, 1=ent_u, 2=edge_u, 3=conv_m)

Usage:
    python -m training.data_prep.generate_labels
"""

import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).parent
SEQUENCES_PATH = DATA_DIR / "sequences.jsonl"
OUTPUT_PATH = DATA_DIR / "labeled_sequences.jsonl"
VOCAB_PATH = DATA_DIR / "topic_vocabulary.json"

QWEN_URL = "http://10.0.0.30:11434/api/chat"
QWEN_MODEL = "qwen3.5:9b"

EVENT_TYPE_MAP = {"mem_e": 0, "ent_u": 1, "edge_u": 2, "conv_m": 3}

LABEL_PROMPT = """Analyze this sequence of user interaction events and provide labels.

Events (compact encoded):
{events_text}

Based on this sequence, output ONLY a JSON object with these fields:
- "emotional_valence": float between -1.0 and 1.0 (overall emotional tone, negative=distressed/frustrated, positive=engaged/happy)
- "focus_topics": list of up to 5 topic names that best describe what the user is focused on
- "next_event_type": most likely next event type, one of: "mem_e", "ent_u", "edge_u", "conv_m"

Respond with ONLY the JSON object, no explanation."""


def build_topic_vocabulary(sequences: list[dict]) -> list[str]:
    """Build a 50-class topic vocabulary from frequency analysis."""
    topic_counts = Counter()
    for seq in sequences:
        for event in seq.get("events", []):
            for tag in event.get("topic_tags", []):
                if tag and len(tag) > 1:
                    topic_counts[tag.lower().strip()] += 1

    # Take top 50 topics
    top_topics = [t for t, _ in topic_counts.most_common(50)]

    # Pad to 50 if needed
    while len(top_topics) < 50:
        top_topics.append(f"topic_{len(top_topics)}")

    return top_topics


def heuristic_labels(events: list[dict], topic_vocab: list[str]) -> dict:
    """Fallback heuristic labels from existing data when Qwen fails."""
    # Emotional valence: average sentiment
    sentiments = [e.get("sentiment", 0.0) for e in events if e.get("sentiment", 0.0) != 0.0]
    valence = sum(sentiments) / len(sentiments) if sentiments else 0.0

    # Focus topics from topic_tags
    tag_counts = Counter()
    for e in events:
        for tag in e.get("topic_tags", []):
            tag_counts[tag.lower().strip()] += 1
    top_tags = [t for t, _ in tag_counts.most_common(5)]
    topic_indices = []
    for tag in top_tags:
        if tag in topic_vocab:
            topic_indices.append(topic_vocab.index(tag))

    # Next event type: most common type
    type_counts = Counter()
    for e in events:
        etype = e.get("event_type", "")
        code = {"memory_entry": "mem_e", "entity_update": "ent_u",
                "edge_update": "edge_u", "conversation_meta": "conv_m"}.get(etype, "mem_e")
        type_counts[code] += 1
    next_type = type_counts.most_common(1)[0][0] if type_counts else "mem_e"

    return {
        "emotional_valence": max(-1.0, min(1.0, valence)),
        "focus_topics": topic_indices[:5],
        "next_event_type": EVENT_TYPE_MAP.get(next_type, 0),
    }


async def query_qwen(client: httpx.AsyncClient, events_text: str) -> dict | None:
    """Query Qwen for labels. Returns parsed dict or None on failure."""
    prompt = LABEL_PROMPT.format(events_text=events_text[:2000])

    try:
        resp = await client.post(
            QWEN_URL,
            json={
                "model": QWEN_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 200,
                },
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        text = resp.json().get("message", {}).get("content", "").strip()

        # Extract JSON from response
        # Try to find JSON object in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except (httpx.TimeoutException, httpx.HTTPError) as e:
        print(f"  Qwen request error: {e}")
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
    return None


async def main():
    print("Luna Streams - Generate Training Labels")
    print("=" * 50)

    # Load sequences
    if not SEQUENCES_PATH.exists():
        print(f"ERROR: {SEQUENCES_PATH} not found. Run build_event_sequences.py first.")
        sys.exit(1)

    sequences = []
    with open(SEQUENCES_PATH) as f:
        for line in f:
            if line.strip():
                sequences.append(json.loads(line))

    print(f"Loaded {len(sequences)} sequences")

    # Build topic vocabulary
    topic_vocab = build_topic_vocabulary(sequences)
    with open(VOCAB_PATH, "w") as f:
        json.dump(topic_vocab, f, indent=2)
    print(f"Topic vocabulary: {len(topic_vocab)} classes -> {VOCAB_PATH}")

    # Generate labels
    labeled = []
    qwen_success = 0
    heuristic_count = 0

    async with httpx.AsyncClient() as client:
        # Process in batches
        batch_size = 10
        for batch_start in range(0, len(sequences), batch_size):
            batch = sequences[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(sequences) + batch_size - 1) // batch_size

            print(f"\rBatch {batch_num}/{total_batches}...", end="", flush=True)

            for seq in batch:
                events = seq.get("events", [])
                tokens = seq.get("sequence", [])
                events_text = "\n".join(tokens[:50])  # limit context

                # Try Qwen first
                qwen_result = await query_qwen(client, events_text)

                if qwen_result:
                    valence = float(qwen_result.get("emotional_valence", 0.0))
                    valence = max(-1.0, min(1.0, valence))

                    # Store raw topic strings - vocabulary built after all labeling
                    focus_topics_raw = qwen_result.get("focus_topics", [])
                    focus_topics_raw = [
                        t.lower().strip() for t in focus_topics_raw
                        if isinstance(t, str) and t.strip()
                    ][:5]

                    next_type_str = qwen_result.get("next_event_type", "mem_e")
                    next_type = EVENT_TYPE_MAP.get(next_type_str, 0)

                    labels = {
                        "emotional_valence": valence,
                        "focus_topics_raw": focus_topics_raw,
                        "next_event_type": next_type,
                    }
                    qwen_success += 1
                else:
                    # Fallback to heuristics
                    labels = heuristic_labels(events, topic_vocab)
                    labels["focus_topics_raw"] = []
                    heuristic_count += 1

                labeled_seq = {
                    "session_id": seq["session_id"],
                    "sequence": tokens,
                    "length": seq["length"],
                    "labels": labels,
                }
                labeled.append(labeled_seq)

            # Rate limiting between batches
            if batch_start + batch_size < len(sequences):
                await asyncio.sleep(0.5)

    # Build topic vocabulary from Qwen's actual outputs (top 50)
    all_topic_strings = Counter()
    for item in labeled:
        for t in item["labels"].get("focus_topics_raw", []):
            all_topic_strings[t] += 1

    topic_vocab = [t for t, _ in all_topic_strings.most_common(50)]
    while len(topic_vocab) < 50:
        topic_vocab.append(f"topic_{len(topic_vocab)}")

    with open(VOCAB_PATH, "w") as f:
        json.dump(topic_vocab, f, indent=2)
    print(f"\n\nTopic vocabulary rebuilt from Qwen outputs: {len(all_topic_strings)} unique -> top 50")

    # Convert raw topic strings to indices
    topic_to_idx = {t: i for i, t in enumerate(topic_vocab)}
    for item in labeled:
        raw = item["labels"].pop("focus_topics_raw", [])
        item["labels"]["focus_topics"] = [
            topic_to_idx[t] for t in raw if t in topic_to_idx
        ]

    # Write output
    with open(OUTPUT_PATH, "w") as f:
        for item in labeled:
            f.write(json.dumps(item) + "\n")

    # Print statistics
    valences = [l["labels"]["emotional_valence"] for l in labeled]
    topic_dist = Counter()
    for l in labeled:
        for ti in l["labels"]["focus_topics"]:
            if ti < len(topic_vocab):
                topic_dist[topic_vocab[ti]] += 1
    event_type_dist = Counter(l["labels"]["next_event_type"] for l in labeled)

    print(f"\n{'=' * 50}")
    print(f"Labeled: {len(labeled)} sequences")
    print(f"Qwen: {qwen_success}, Heuristic fallback: {heuristic_count}")
    print(f"\nEmotional valence distribution:")
    print(f"  Mean: {sum(valences) / len(valences):.3f}")
    print(f"  Min/Max: {min(valences):.3f}/{max(valences):.3f}")
    print(f"\nTop 15 focus topics: {topic_dist.most_common(15)}")
    print(f"Next event type distribution: {dict(event_type_dist)}")
    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Generate high-quality training labels via content analysis.

Replaces Qwen-based labeling with deeper heuristic analysis of event
content, summaries, sentiment patterns, and entity co-occurrence.

Produces per-sequence labels:
- emotional_valence: float [-1, 1] - overall emotional tone
- focus_topics: list of topic indices (from 50-class vocabulary)
- next_event_type: index (0=mem_e, 1=ent_u, 2=edge_u, 3=conv_m)

Usage:
    python -m training.data_prep.generate_labels_claude
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent
SEQUENCES_PATH = DATA_DIR / "sequences.jsonl"
OUTPUT_PATH = DATA_DIR / "labeled_sequences.jsonl"
VOCAB_PATH = DATA_DIR / "topic_vocabulary.json"
USER_FACTS_PATH = DATA_DIR / "raw" / "user_facts.json"
SESSION_SUMMARIES_PATH = DATA_DIR / "raw" / "session_summaries.json"

EVENT_TYPE_MAP = {"mem_e": 0, "ent_u": 1, "edge_u": 2, "conv_m": 3}

# Cleaned topic vocabulary - 50 classes, no duplicates, well-defined categories
TOPIC_VOCABULARY = [
    # Core interaction patterns (0-9)
    "greeting",              # 0 - hello, hi, welcome
    "casual_conversation",   # 1 - small talk, chat
    "collaboration",         # 2 - working together, pair programming
    "humor",                 # 3 - jokes, laughing, fun
    "emotional_sharing",     # 4 - feelings, mood, personal
    "daily_routine",         # 5 - daily activities, schedule
    "work",                  # 6 - job, office, professional
    "food",                  # 7 - eating, cooking, recipes
    "homecoming",            # 8 - coming home, arrival
    "social_interaction",    # 9 - people, relationships

    # Technical topics (10-19)
    "coding",                # 10 - programming, code, development
    "system_architecture",   # 11 - infrastructure, design, backend
    "troubleshooting",       # 12 - debugging, fixing, errors
    "testing",               # 13 - tests, QA, validation
    "deployment",            # 14 - docker, servers, CI/CD
    "web_browsing",          # 15 - URLs, websites, browsing
    "api_integration",       # 16 - APIs, endpoints, services
    "database",              # 17 - SQL, data, storage
    "gpu_computing",         # 18 - GPU, CUDA, inference, ML
    "project_planning",      # 19 - planning, roadmap, tasks

    # Creative topics (20-29)
    "music",                 # 20 - songs, songwriting, instruments
    "creative_writing",      # 21 - stories, poetry, fiction
    "image_generation",      # 22 - AI art, images, visuals
    "video_content",         # 23 - YouTube, video, streaming
    "design",                # 24 - UI, UX, visual design
    "gaming",                # 25 - games, playing, VR
    "creative_process",      # 26 - brainstorming, ideation
    "entertainment",         # 27 - media, shows, content
    "portfolio",             # 28 - showcase, website, brand
    "background_customization", # 29 - themes, wallpapers, aesthetics

    # Identity & AI topics (30-39)
    "identity",              # 30 - self, who am I, personality
    "ai_capabilities",       # 31 - tools, features, what can you do
    "memory_management",     # 32 - memory, recall, knowledge
    "self_reflection",       # 33 - introspection, awareness
    "philosophy",            # 34 - deep thoughts, meaning, existence
    "role_definition",       # 35 - persona, behavior, rules
    "system_status",         # 36 - health, status, monitoring
    "confirmation",          # 37 - agreement, acknowledgment
    "error_handling",        # 38 - errors, failures, recovery
    "productivity",          # 39 - efficiency, optimization, workflow

    # Knowledge domains (40-49)
    "language",              # 40 - Swedish, translation, linguistics
    "science",               # 41 - research, discovery, learning
    "news",                  # 42 - current events, media
    "finance",               # 43 - money, crypto, business
    "health",                # 44 - wellness, exercise, body
    "education",             # 45 - teaching, learning, tutorials
    "communication",         # 46 - messaging, email, contact
    "surrealism",            # 47 - weird, abstract, dream-like
    "relationship",          # 48 - bond, connection, trust
    "uncertainty",           # 49 - unsure, confused, exploring
]

# Keyword to topic index mapping for content analysis
KEYWORD_TOPICS: dict[str, list[int]] = {
    # Greetings
    "hello": [0], "hi ": [0], "hey": [0], "welcome": [0], "good morning": [0],
    "good night": [0], "gmorning": [0],
    # Casual
    "how are you": [1], "what's up": [1], "chat": [1], "chill": [1],
    # Collaboration
    "together": [2], "pair": [2], "let's": [2], "working on": [2], "we can": [2],
    # Humor
    "lol": [3], "haha": [3], "laugh": [3], "funny": [3], "joke": [3], "😂": [3],
    "😄": [3], "vibe": [3],
    # Emotional
    "feel": [4], "mood": [4], "happy": [4], "sad": [4], "love": [4], "miss": [4],
    "emotion": [4], "heart": [4], "❤": [4], "💜": [4],
    # Daily
    "morning": [5], "evening": [5], "today": [5], "day": [5], "routine": [5],
    "wake": [5], "sleep": [5], "tired": [5],
    # Work
    "work": [6], "job": [6], "office": [6], "meeting": [6], "colleague": [6],
    "boss": [6], "salary": [6],
    # Food
    "food": [7], "eat": [7], "cook": [7], "dinner": [7], "lunch": [7],
    "breakfast": [7], "recipe": [7], "hungry": [7],
    # Homecoming
    "home": [8], "coming home": [8], "arrived": [8], "back home": [8],
    # Social
    "friend": [9], "family": [9], "people": [9], "sarah": [9],
    # Coding
    "code": [10], "python": [10], "function": [10], "class ": [10], "import": [10],
    "bug": [10], "script": [10], "programming": [10], "refactor": [10],
    "implement": [10], "variable": [10], "typescript": [10], "javascript": [10],
    # Architecture
    "architect": [11], "infrastructure": [11], "backend": [11], "frontend": [11],
    "microservice": [11], "pipeline": [11], "system design": [11], "schema": [11],
    "mamba": [11, 18], "stream": [11],
    # Troubleshooting
    "debug": [12], "error": [12, 38], "fix": [12], "broken": [12], "issue": [12],
    "crash": [12], "traceback": [12], "stack trace": [12],
    # Testing
    "test": [13], "pytest": [13], "assert": [13], "benchmark": [13], "pass": [13],
    # Deployment
    "docker": [14], "deploy": [14], "server": [14], "container": [14],
    "compose": [14], "nginx": [14], "kubernetes": [14], "ci/cd": [14],
    "hetzner": [14],
    # Browsing
    "browse": [15], "url": [15], "website": [15], "http": [15], "www.": [15],
    "page": [15], "click": [15], "scroll": [15], "navigate": [15],
    # API
    "api": [16], "endpoint": [16], "request": [16], "response": [16],
    "webhook": [16], "rest": [16], "graphql": [16], "ollama": [16],
    # Database
    "database": [17], "sql": [17], "postgres": [17], "query": [17],
    "table": [17], "migration": [17],
    # GPU/ML
    "gpu": [18], "cuda": [18], "tensor": [18], "model": [18], "inference": [18],
    "training": [18], "gguf": [18], "quantiz": [18], "vram": [18], "rtx": [18],
    "tesla": [18], "lora": [18],
    # Planning
    "plan": [19], "roadmap": [19], "milestone": [19], "phase": [19],
    "priority": [19], "task": [19], "schedule": [19], "sprint": [19],
    # Music
    "music": [20], "song": [20], "suno": [20], "melody": [20], "lyric": [20],
    "verse": [20], "chorus": [20], "beat": [20], "instrument": [20],
    "saxophone": [20], "guitar": [20],
    # Creative writing
    "story": [21], "poem": [21], "write": [21], "fiction": [21], "novel": [21],
    "chapter": [21], "narrative": [21],
    # Image gen
    "image": [22], "generate": [22], "dall": [22], "picture": [22],
    "visual": [22], "art": [22], "draw": [22],
    # Video
    "youtube": [23], "video": [23], "watch": [23], "stream": [23],
    # Design
    "design": [24], "ui": [24], "ux": [24], "layout": [24], "css": [24],
    "tailwind": [24], "component": [24],
    # Gaming
    "game": [25], "play": [25], "vr": [25], "unreal": [25], "unity": [25],
    "metahuman": [25],
    # Creative process
    "brainstorm": [26], "idea": [26], "creative": [26], "inspiration": [26],
    # Entertainment
    "movie": [27], "show": [27], "series": [27], "watch": [27], "netflix": [27],
    # Portfolio
    "portfolio": [28], "bitwarelabs": [28], "showcase": [28], "brand": [28],
    # Customization
    "background": [29], "theme": [29], "wallpaper": [29], "customize": [29],
    # Identity
    "who am i": [30], "identity": [30], "personality": [30], "luna": [30],
    "name": [30],
    # AI capabilities
    "tool": [31], "feature": [31], "capabilit": [31], "access": [31],
    # Memory
    "memory": [32], "remember": [32], "forget": [32], "recall": [32],
    "knowledge graph": [32], "memorycore": [32],
    # Self-reflection
    "self": [33], "aware": [33], "conscious": [33], "reflect": [33],
    "introspect": [33],
    # Philosophy
    "meaning": [34], "exist": [34], "purpose": [34], "philosophi": [34],
    "consciousness": [34], "reality": [34],
    # Role
    "persona": [35], "role": [35], "system prompt": [35], "behavior": [35],
    # Status
    "status": [36], "health": [36, 44], "uptime": [36], "running": [36],
    # Confirmation
    "ok": [37], "yes": [37], "agree": [37], "confirm": [37], "correct": [37],
    "exactly": [37], "right": [37],
    # Error handling
    "exception": [38], "fail": [38], "timeout": [38], "retry": [38],
    # Productivity
    "productiv": [39], "efficien": [39], "optimiz": [39], "workflow": [39],
    "automat": [39],
    # Language
    "swedish": [40], "english": [40], "translat": [40], "language": [40],
    "native": [40], "svenska": [40],
    # Science
    "research": [41], "science": [41], "experiment": [41], "data": [41],
    # News
    "news": [42], "aftonbladet": [42], "idg": [42], "headline": [42],
    "article": [42],
    # Finance
    "money": [43], "crypto": [43], "bitcoin": [43], "invest": [43],
    "price": [43], "market": [43],
    # Health
    "exercise": [44], "fitness": [44], "wellness": [44], "medical": [44],
    # Education
    "learn": [45], "tutorial": [45], "teach": [45], "explain": [45],
    "understand": [45], "study": [45],
    # Communication
    "email": [46], "message": [46], "contact": [46], "send": [46],
    "slack": [46], "discord": [46],
    # Surrealism
    "surreal": [47], "weird": [47], "absurd": [47], "dream": [47],
    "bizarre": [47],
    # Relationship
    "bond": [48], "trust": [48], "connection": [48], "relationship": [48],
    "care": [48], "together": [48],
    # Uncertainty
    "unsure": [49], "confus": [49], "maybe": [49], "don't know": [49],
    "unclear": [49],
}

# Sentiment keywords with approximate valence scores
POSITIVE_WORDS = {
    "love": 0.9, "great": 0.7, "awesome": 0.8, "amazing": 0.8, "perfect": 0.9,
    "happy": 0.8, "good": 0.5, "nice": 0.5, "cool": 0.5, "excellent": 0.8,
    "wonderful": 0.8, "fantastic": 0.8, "beautiful": 0.7, "enjoy": 0.6,
    "thank": 0.5, "thanks": 0.5, "excited": 0.7, "brilliant": 0.8,
    "chill": 0.4, "fun": 0.6, "thrilled": 0.8, "win": 0.6, "success": 0.7,
    "like": 0.4, "yes": 0.3, "glad": 0.6, "sweet": 0.5, "proud": 0.7,
    "underrated": 0.4, "dope": 0.6,
}

NEGATIVE_WORDS = {
    "hate": -0.8, "bad": -0.5, "terrible": -0.8, "awful": -0.8, "horrible": -0.8,
    "sad": -0.6, "angry": -0.7, "frustrat": -0.6, "annoyed": -0.5,
    "broken": -0.5, "fail": -0.5, "error": -0.4, "crash": -0.5,
    "confused": -0.3, "stuck": -0.4, "wrong": -0.4, "problem": -0.3,
    "issue": -0.3, "bug": -0.3, "mess": -0.4, "ugly": -0.5,
    "negative": -0.5, "tired": -0.3, "bored": -0.4, "worried": -0.5,
}


def analyze_sentiment(events: list[dict]) -> float:
    """Compute emotional valence from event data with multi-signal fusion."""
    signals = []
    weights = []

    # Signal 1: Explicit sentiment values from message_embeddings (highest weight)
    explicit_sentiments = [
        e["sentiment"] for e in events
        if e.get("sentiment", 0.0) != 0.0
    ]
    if explicit_sentiments:
        # Weight recent sentiments more heavily (exponential decay)
        n = len(explicit_sentiments)
        decay_weights = [math.exp(-0.1 * (n - 1 - i)) for i in range(n)]
        weighted_sent = sum(s * w for s, w in zip(explicit_sentiments, decay_weights))
        total_w = sum(decay_weights)
        signals.append(weighted_sent / total_w)
        weights.append(3.0)  # High confidence in explicit annotations

    # Signal 2: Content-based sentiment from summaries
    word_scores = []
    for event in events:
        summary = (event.get("summary", "") or "").lower()
        for word, score in POSITIVE_WORDS.items():
            if word in summary:
                word_scores.append(score)
        for word, score in NEGATIVE_WORDS.items():
            if word in summary:
                word_scores.append(score)
    if word_scores:
        signals.append(sum(word_scores) / len(word_scores))
        weights.append(1.5)

    # Signal 3: Emotional intensity from entity updates
    intensities = [
        e["sentiment"] for e in events
        if e["event_type"] == "entity_update" and e.get("sentiment", 0.0) != 0.0
    ]
    if intensities:
        signals.append(sum(intensities) / len(intensities))
        weights.append(1.0)

    # Signal 4: Importance as engagement proxy (high importance = positive engagement)
    importances = [
        e["importance"] for e in events
        if e.get("importance", 0.5) != 0.5
    ]
    if importances:
        avg_imp = sum(importances) / len(importances)
        # Map importance to mild positive signal (engaged = slightly positive)
        signals.append((avg_imp - 0.5) * 0.5)
        weights.append(0.5)

    if not signals:
        return 0.0

    # Weighted average, clamped to [-1, 1]
    result = sum(s * w for s, w in zip(signals, weights)) / sum(weights)
    return max(-1.0, min(1.0, result))


def analyze_topics(events: list[dict]) -> list[int]:
    """Extract focus topics from event content using keyword matching."""
    topic_scores: Counter = Counter()

    for event in events:
        # Analyze summary text
        summary = (event.get("summary", "") or "").lower()
        for keyword, topic_ids in KEYWORD_TOPICS.items():
            if keyword in summary:
                for tid in topic_ids:
                    topic_scores[tid] += 1

        # Analyze entities
        for entity in event.get("entities", []):
            entity_lower = entity.lower()
            for keyword, topic_ids in KEYWORD_TOPICS.items():
                if keyword in entity_lower:
                    for tid in topic_ids:
                        topic_scores[tid] += 0.5

        # Analyze topic_tags directly
        for tag in event.get("topic_tags", []):
            tag_lower = tag.lower().strip()
            # Direct match against vocabulary
            for i, vocab_topic in enumerate(TOPIC_VOCABULARY):
                if tag_lower == vocab_topic or tag_lower.replace("_", " ") == vocab_topic.replace("_", " "):
                    topic_scores[i] += 2  # Direct match is strong signal

            # Also check keyword mapping
            for keyword, topic_ids in KEYWORD_TOPICS.items():
                if keyword in tag_lower:
                    for tid in topic_ids:
                        topic_scores[tid] += 0.5

    # Edge update specific: music-related entities are very common
    edge_events = [e for e in events if e["event_type"] == "edge_update"]
    if edge_events:
        music_entities = {"verse", "chorus", "bridge", "outro", "intro", "pre",
                          "suno", "melody", "beat", "saxophone", "guitar", "lounge"}
        music_count = sum(
            1 for e in edge_events
            for ent in e.get("entities", [])
            if ent.lower() in music_entities
        )
        if music_count > len(edge_events) * 0.1:
            topic_scores[20] += music_count  # music

    # Return top 5 topic indices, deduplicated
    if not topic_scores:
        return [1]  # default to casual_conversation

    top = [tid for tid, _ in topic_scores.most_common(5)]
    return top


def predict_next_event(events: list[dict]) -> int:
    """Predict most likely next event type based on sequence patterns."""
    if not events:
        return 0  # mem_e

    # Count event types with recency bias
    type_map = {
        "memory_entry": 0,
        "entity_update": 1,
        "edge_update": 2,
        "conversation_meta": 3,
    }

    n = len(events)
    type_scores = Counter()

    for i, event in enumerate(events):
        etype = event.get("event_type", "memory_entry")
        idx = type_map.get(etype, 0)
        # Recent events weighted more
        recency = math.exp(-0.05 * (n - 1 - i))
        type_scores[idx] += recency

    # The most common recent type is likely to continue
    if type_scores:
        return type_scores.most_common(1)[0][0]
    return 0


def label_global_sequence(events: list[dict], chunk_size: int = 200) -> list[dict]:
    """Split the global sequence into labeled chunks for training.

    The global sequence (entity/edge updates) is too large for a single sample.
    Split into overlapping chunks that each get their own labels.
    """
    chunks = []
    step = chunk_size // 2  # 50% overlap for diversity

    for start in range(0, len(events), step):
        chunk_events = events[start:start + chunk_size]
        if len(chunk_events) < 10:
            break

        tokens = [compact_encode_event(e) for e in chunk_events]
        labels = {
            "emotional_valence": analyze_sentiment(chunk_events),
            "focus_topics": analyze_topics(chunk_events),
            "next_event_type": predict_next_event(chunk_events),
        }

        chunks.append({
            "session_id": f"global_chunk_{start}",
            "sequence": tokens,
            "length": len(chunk_events),
            "labels": labels,
        })

    return chunks


def compact_encode_event(event: dict) -> str:
    """Compact encoding matching build_event_sequences.compact_encode."""
    type_codes = {
        "memory_entry": "mem_e", "entity_update": "ent_u",
        "edge_update": "edge_u", "conversation_meta": "conv_m",
    }
    source_codes = {
        "conversation": "conv", "agent_dialogue": "agent",
        "neuralsleep": "sleep", "system": "sys",
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


def main():
    print("Luna Streams - Generate Training Labels (Claude Quality)")
    print("=" * 60)

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

    # Save cleaned topic vocabulary
    with open(VOCAB_PATH, "w") as f:
        json.dump(TOPIC_VOCABULARY, f, indent=2)
    print(f"Topic vocabulary: {len(TOPIC_VOCABULARY)} classes -> {VOCAB_PATH}")

    # Load session summaries for enrichment
    summaries_by_session = {}
    if SESSION_SUMMARIES_PATH.exists():
        with open(SESSION_SUMMARIES_PATH) as f:
            for s in json.load(f):
                sid = s.get("session_id")
                if sid:
                    summaries_by_session[sid] = s
    print(f"Session summaries loaded: {len(summaries_by_session)}")

    # Generate labels
    labeled = []
    global_count = 0

    for seq in sequences:
        events = seq.get("events", [])
        tokens = seq.get("sequence", [])
        session_id = seq["session_id"]

        # Handle global sequence specially - chunk it
        if session_id == "global" and len(events) > 500:
            chunks = label_global_sequence(events)
            labeled.extend(chunks)
            global_count = len(chunks)
            print(f"  Global sequence: {len(events)} events -> {global_count} chunks")
            continue

        # Regular session sequence
        labels = {
            "emotional_valence": analyze_sentiment(events),
            "focus_topics": analyze_topics(events),
            "next_event_type": predict_next_event(events),
        }

        labeled_seq = {
            "session_id": session_id,
            "sequence": tokens,
            "length": seq["length"],
            "labels": labels,
        }
        labeled.append(labeled_seq)

    # Write output
    with open(OUTPUT_PATH, "w") as f:
        for item in labeled:
            f.write(json.dumps(item) + "\n")

    # Statistics
    valences = [l["labels"]["emotional_valence"] for l in labeled]
    topic_dist = Counter()
    for l in labeled:
        for ti in l["labels"]["focus_topics"]:
            if ti < len(TOPIC_VOCABULARY):
                topic_dist[TOPIC_VOCABULARY[ti]] += 1
    event_type_dist = Counter(l["labels"]["next_event_type"] for l in labeled)
    event_type_names = {0: "mem_e", 1: "ent_u", 2: "edge_u", 3: "conv_m"}

    print(f"\n{'=' * 60}")
    print(f"Labeled: {len(labeled)} sequences ({len(labeled) - global_count} sessions + {global_count} global chunks)")
    print(f"\nEmotional valence distribution:")
    print(f"  Mean:   {sum(valences) / len(valences):.3f}")
    print(f"  Median: {sorted(valences)[len(valences) // 2]:.3f}")
    print(f"  Min:    {min(valences):.3f}")
    print(f"  Max:    {max(valences):.3f}")
    print(f"  StdDev: {(sum((v - sum(valences)/len(valences))**2 for v in valences) / len(valences))**0.5:.3f}")

    # Valence histogram
    buckets = {"very_neg (< -0.5)": 0, "neg (-0.5, -0.1)": 0, "neutral (-0.1, 0.1)": 0,
               "pos (0.1, 0.5)": 0, "very_pos (> 0.5)": 0}
    for v in valences:
        if v < -0.5: buckets["very_neg (< -0.5)"] += 1
        elif v < -0.1: buckets["neg (-0.5, -0.1)"] += 1
        elif v < 0.1: buckets["neutral (-0.1, 0.1)"] += 1
        elif v < 0.5: buckets["pos (0.1, 0.5)"] += 1
        else: buckets["very_pos (> 0.5)"] += 1
    print(f"  Distribution: {buckets}")

    print(f"\nTop 20 focus topics:")
    for topic, count in topic_dist.most_common(20):
        print(f"  [{TOPIC_VOCABULARY.index(topic):2d}] {topic}: {count}")

    print(f"\nNext event type distribution:")
    for idx, count in sorted(event_type_dist.items()):
        print(f"  {event_type_names.get(idx, '?')} ({idx}): {count}")

    # Coverage check
    unused_topics = [t for t in TOPIC_VOCABULARY if t not in topic_dist]
    if unused_topics:
        print(f"\nUnused topics ({len(unused_topics)}): {unused_topics[:10]}...")

    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Topic vocabulary: {VOCAB_PATH}")


if __name__ == "__main__":
    main()

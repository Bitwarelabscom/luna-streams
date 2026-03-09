#!/usr/bin/env python3
"""LoRA fine-tune Mamba for user modeling.

Trains mamba-370m-hf with LoRA adapters + MLP output heads.
Designed to run on RTX 3080 (10GB VRAM) at 10.0.0.30.

Usage:
    python train.py [--config configs/stream1_user.yaml] [--epochs 10]

Post-training:
    1. Merge LoRA weights
    2. Convert to GGUF Q8_0
    3. Export MLP heads as safetensors
    4. Copy to Hetzner /opt/luna-streams/models/
"""

import argparse
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, random_split
from safetensors.torch import save_file as save_safetensors


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "configs/stream1_user.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MambaSequenceDataset(Dataset):
    """Dataset of tokenized event sequences with labels."""

    def __init__(self, data_path: str, tokenizer, max_len: int = 512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(data_path) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Concatenate all compact tokens into a single string
                    text = " | ".join(item.get("sequence", []))
                    labels = item.get("labels", {})
                    if text and labels:
                        self.samples.append((text, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, labels = self.samples[idx]

        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # Labels
        valence = torch.tensor([labels.get("emotional_valence", 0.0)], dtype=torch.float32)

        # Focus topics as multi-hot vector (50 classes)
        topic_indices = labels.get("focus_topics", [])
        topics = torch.zeros(50, dtype=torch.float32)
        for ti in topic_indices:
            if 0 <= ti < 50:
                topics[ti] = 1.0

        # Next event type (4 classes)
        next_type = labels.get("next_event_type", 0)
        next_type = torch.tensor(next_type, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "valence_label": valence,
            "topics_label": topics,
            "next_type_label": next_type,
        }


# ---------------------------------------------------------------------------
# MLP Output Heads
# ---------------------------------------------------------------------------

class EmotionalValenceHead(nn.Module):
    """hidden_dim -> 256 -> 1, tanh output."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class FocusTopicsHead(nn.Module):
    """hidden_dim -> 256 -> 50, raw logits (BCE with logits for stability)."""
    def __init__(self, hidden_dim: int, n_topics: int = 50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_topics),
        )

    def forward(self, x):
        return self.net(x)  # raw logits, no sigmoid


class NextEventHead(nn.Module):
    """hidden_dim -> 128 -> 4, softmax for next event prediction."""
    def __init__(self, hidden_dim: int, n_types: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_types),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size buffer for experience replay to prevent forgetting."""

    def __init__(self, capacity: int = 1000):
        self.buffer = deque(maxlen=capacity)

    def add(self, batch):
        self.buffer.append(batch)

    def sample(self, n: int):
        n = min(n, len(self.buffer))
        return random.sample(list(self.buffer), n)


def extract_hidden_state(model, input_ids, attention_mask):
    """Extract the last hidden state from the Mamba model."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    # Last hidden state, last token position
    # Find the last non-padding token for each sequence
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_size = input_ids.shape[0]
    hidden_states = outputs.hidden_states[-1]  # (batch, seq, hidden)

    # Gather last token's hidden state
    last_hidden = torch.stack([
        hidden_states[i, seq_lengths[i].long()] for i in range(batch_size)
    ])
    return last_hidden, outputs


def train_epoch(
    model, heads, train_loader, optimizer, device, config, replay_buffer, epoch
):
    model.train()
    for head in heads.values():
        head.train()

    total_loss = 0
    n_batches = 0
    aux_weight = config["training"].get("aux_loss_weight", 0.1)
    replay_ratio = config["training"].get("replay_mix_ratio", 0.1)
    grad_accum = config["training"].get("gradient_accumulation", 1)
    use_fp16 = config["training"].get("fp16", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        valence_labels = batch["valence_label"].to(device)
        topics_labels = batch["topics_label"].to(device)
        next_type_labels = batch["next_type_label"].to(device)

        # Forward through Mamba (with autocast for fp16)
        with torch.amp.autocast("cuda", enabled=use_fp16):
            hidden, outputs = extract_hidden_state(model, input_ids, attention_mask)

            # Causal LM loss
            lm_logits = outputs.logits
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            lm_loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0,  # padding
            )

            # Head losses (cast hidden to float32 for heads)
            hidden_f32 = hidden.float()
            valence_pred = heads["emotional_valence"](hidden_f32)
            valence_loss = nn.functional.mse_loss(valence_pred, valence_labels)

            topics_logits = heads["focus_topics"](hidden_f32)
            topics_loss = nn.functional.binary_cross_entropy_with_logits(topics_logits, topics_labels)

            next_pred = heads["next_event"](hidden_f32)
            next_loss = nn.functional.cross_entropy(next_pred, next_type_labels)

            # Combined loss scaled for gradient accumulation
            loss = (lm_loss + valence_loss + topics_loss + aux_weight * next_loss).float()
            loss = loss / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step optimizer every grad_accum batches
        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + [p for h in heads.values() for p in h.parameters()],
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + [p for h in heads.values() for p in h.parameters()],
                    max_norm=1.0,
                )
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum  # undo the scaling for logging
        n_batches += 1

        # Add to replay buffer
        if random.random() < replay_ratio:
            replay_buffer.add({
                k: v.detach().cpu() for k, v in batch.items()
            })

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, heads, val_loader, device, config):
    model.eval()
    for head in heads.values():
        head.eval()

    total_loss = 0
    total_valence_mae = 0
    total_next_correct = 0
    total_next_count = 0
    n_batches = 0
    aux_weight = config["training"].get("aux_loss_weight", 0.1)

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        valence_labels = batch["valence_label"].to(device)
        topics_labels = batch["topics_label"].to(device)
        next_type_labels = batch["next_type_label"].to(device)

        hidden, outputs = extract_hidden_state(model, input_ids, attention_mask)

        # LM loss
        lm_logits = outputs.logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        lm_loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=0,
        )

        valence_pred = heads["emotional_valence"](hidden)
        valence_loss = nn.functional.mse_loss(valence_pred, valence_labels)
        total_valence_mae += (valence_pred - valence_labels).abs().mean().item()

        topics_logits = heads["focus_topics"](hidden)
        topics_loss = nn.functional.binary_cross_entropy_with_logits(topics_logits, topics_labels)

        next_pred = heads["next_event"](hidden)
        next_loss = nn.functional.cross_entropy(next_pred, next_type_labels)
        total_next_correct += (next_pred.argmax(dim=1) == next_type_labels).sum().item()
        total_next_count += next_type_labels.shape[0]

        loss = (lm_loss + valence_loss + topics_loss + aux_weight * next_loss).float()
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    valence_mae = total_valence_mae / max(n_batches, 1)
    next_acc = total_next_correct / max(total_next_count, 1)

    return {
        "val_loss": avg_loss,
        "valence_mae": valence_mae,
        "next_event_accuracy": next_acc,
    }


def export_heads_numpy(heads: dict, output_path: str):
    """Export trained head weights for numpy-only inference on deployment server."""
    state = {}
    for name, head in heads.items():
        for i, layer in enumerate(head.net):
            if isinstance(layer, nn.Linear):
                state[f"{name}.layer{i}.weight"] = layer.weight.data.cpu()
                state[f"{name}.layer{i}.bias"] = layer.bias.data.cpu()

    save_safetensors(state, output_path)
    print(f"Exported head weights: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Mamba for user modeling")
    parser.add_argument("--config", default="configs/stream1_user.yaml")
    parser.add_argument("--data", default="data_prep/labeled_sequences.jsonl")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    model_name = config["model"]["base"]
    hidden_dim = config["model"]["hidden_dim"]
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # fp16 for 2.8B to fit in 10GB VRAM, fp32 for 370M (Mamba can NaN in fp16 on some configs)
    use_fp16 = config["training"].get("fp16", False) and device.type == "cuda"
    model_dtype = torch.float16 if use_fp16 else torch.float32
    print(f"Model dtype: {model_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
    ).to(device)

    # Apply LoRA
    lora_config = LoraConfig(
        r=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing to reduce VRAM
    model.gradient_checkpointing_enable()

    # Create output heads
    heads = {
        "emotional_valence": EmotionalValenceHead(hidden_dim).to(device),
        "focus_topics": FocusTopicsHead(hidden_dim).to(device),
        "next_event": NextEventHead(hidden_dim).to(device),
    }

    # Dataset
    max_epochs = args.epochs or config["training"].get("max_epochs", 10)
    max_seq_len = config["training"].get("max_seq_length", 512)

    print(f"Loading dataset from {args.data}...")
    dataset = MambaSequenceDataset(args.data, tokenizer, max_len=max_seq_len)
    print(f"Dataset: {len(dataset)} sequences")

    # 80/20 split by session
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = config["training"].get("batch_size", 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Optimizer
    all_params = list(model.parameters()) + [
        p for h in heads.values() for p in h.parameters()
    ]
    optimizer = torch.optim.AdamW(
        all_params,
        lr=config["training"].get("learning_rate", 2e-4),
        weight_decay=config["training"].get("weight_decay", 0.01),
    )

    # Replay buffer
    replay_buffer = ReplayBuffer(
        capacity=config["training"].get("replay_buffer_size", 1000)
    )

    # Training loop with early stopping
    patience = config["training"].get("early_stopping_patience", 3)
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nTraining for up to {max_epochs} epochs (patience={patience})...")
    print("-" * 60)

    for epoch in range(max_epochs):
        train_loss = train_epoch(
            model, heads, train_loader, optimizer, device, config, replay_buffer, epoch
        )
        val_metrics = validate(model, heads, val_loader, device, config)

        print(
            f"Epoch {epoch + 1}/{max_epochs} - "
            f"train_loss: {train_loss:.4f}, "
            f"val_loss: {val_metrics['val_loss']:.4f}, "
            f"valence_mae: {val_metrics['valence_mae']:.4f}, "
            f"next_acc: {val_metrics['next_event_accuracy']:.3f}"
        )

        # Early stopping
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            patience_counter = 0
            # Save best checkpoint
            model.save_pretrained(str(output_dir / "best_lora"))
            export_heads_numpy(heads, str(output_dir / "best_heads.safetensors"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print("-" * 60)
    print(f"Best val_loss: {best_val_loss:.4f}")

    # Merge LoRA and save
    print("\nMerging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_path = output_dir / "merged_model"
    merged_model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))
    print(f"Merged model saved: {merged_path}")

    # Export final heads
    export_heads_numpy(heads, str(output_dir / "mlp_heads.safetensors"))

    print(f"""
Post-training steps:
  1. Convert to GGUF F16: python convert_hf_to_gguf.py {merged_path} --outfile output/model-f16.gguf --outtype f16
  2. Quantize to Q8_0:     llama-quantize output/model-f16.gguf output/model-q8_0.gguf Q8_0
  3. Deploy GGUF + heads to models/ directory
""")


if __name__ == "__main__":
    main()

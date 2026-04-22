from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import Dict
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification


BASE_MODEL_DEFAULT = "facebook/convnextv2-tiny-22k-224"
DATA_DIR_DEFAULT = "data/dataset_split"
OUTPUT_DIR_DEFAULT = "models/classifier"
EPOCHS_DEFAULT = 5
BATCH_SIZE_DEFAULT = 8
LEARNING_RATE = 5e-5


def resolve_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parents[1] / resolved
    return resolved


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_splits(data_dir: Path):
    dataset = load_dataset("imagefolder", data_dir=str(data_dir))
    if "train" not in dataset:
        raise ValueError(f"Missing 'train' split in {data_dir}")

    val_split_name = None
    for candidate in ("val", "validation"):
        if candidate in dataset:
            val_split_name = candidate
            break
    if val_split_name is None:
        raise ValueError(f"Missing validation split in {data_dir} (expected 'val' or 'validation')")

    return dataset["train"], dataset[val_split_name]


def build_label_maps(train_split) -> tuple[list[str], Dict[str, int], Dict[int, str]]:
    class_names = sorted(train_split.features["label"].names)
    label2id = {name: idx for idx, name in enumerate(class_names)}
    id2label = {idx: name for name, idx in label2id.items()}
    return class_names, label2id, id2label


def remap_labels(dataset, label2id: Dict[str, int]):
    original_features = dataset.features["label"]

    def _map_example(example):
        class_name = original_features.int2str(example["label"])
        return {"label": label2id[class_name]}

    return dataset.map(_map_example)


def build_collate_fn(processor):
    def collate_fn(batch):
        images = []
        labels = []
        for item in batch:
            image = item["image"]
            if isinstance(image, Image.Image):
                image = image.convert("RGB")
            else:
                image = Image.fromarray(image).convert("RGB")
            images.append(image)
            labels.append(int(item["label"]))

        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": label_tensor}

    return collate_fn


def save_artifacts(output_dir: Path, model, processor) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


def evaluate(model, data_loader, device) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return (correct / total) * 100.0 if total else 0.0


def train_one_epoch(model, data_loader, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    total_examples = 0

    for batch in tqdm(data_loader, desc="Training", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_examples += batch_size

    return running_loss / total_examples if total_examples else 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an image classifier on the split dataset.")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR_DEFAULT)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--base-model", type=str, default=BASE_MODEL_DEFAULT)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)

    train_split, val_split = load_splits(data_dir)
    class_names, label2id, id2label = build_label_maps(train_split)
    train_split = remap_labels(train_split, label2id)
    val_split = remap_labels(val_split, label2id)

    processor = AutoImageProcessor.from_pretrained(args.base_model)
    model = AutoModelForImageClassification.from_pretrained(
        args.base_model,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.config.id2label = id2label
    model.config.label2id = label2id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collate_fn = build_collate_fn(processor)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_split,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_accuracy = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}/{args.epochs} — loss: {train_loss:.4f} — val_accuracy: {val_accuracy:.2f}%")

        checkpoint_dir = output_dir / f"epoch-{epoch}"
        save_artifacts(checkpoint_dir, model, processor)

    final_dir = output_dir / "final"
    save_artifacts(final_dir, model, processor)

    print("Training complete. Model saved to models/classifier/final/")
    print("To use this model, set classifier_model_path in config/default.yaml")


if __name__ == "__main__":
    main()


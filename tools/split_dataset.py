from __future__ import annotations
import random
import shutil
from pathlib import Path
from typing import Dict, List


class DatasetSplitter:
    def __init__(
        self,
        src_dir: str,
        dst_dir: str,
        val_ratio=0.15,
        test_ratio=0.10,
        seed=42,
    ):
        self.src_dir = Path(src_dir)
        self.dst_dir = Path(dst_dir)
        if not self.src_dir.is_absolute():
            self.src_dir = Path(__file__).resolve().parents[1] / self.src_dir
        if not self.dst_dir.is_absolute():
            self.dst_dir = Path(__file__).resolve().parents[1] / self.dst_dir

        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.seed = int(seed)
        self.extensions = {".jpg", ".jpeg", ".png"}

    def _list_classes(self) -> List[Path]:
        if not self.src_dir.exists() or not self.src_dir.is_dir():
            raise ValueError(f"Source dataset directory not found: {self.src_dir}")
        class_dirs = sorted([p for p in self.src_dir.iterdir() if p.is_dir()])
        if not class_dirs:
            raise ValueError(f"No class directories found in: {self.src_dir}")
        return class_dirs

    def _list_images_for_class(self, class_dir: Path) -> List[Path]:
        files = [
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.extensions
        ]
        return sorted(files)

    def _copy_split_files(self, files: List[Path], split_name: str, class_name: str) -> int:
        split_dir = self.dst_dir / split_name / class_name
        split_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for src_file in files:
            dst_file = split_dir / src_file.name
            shutil.copy2(src_file, dst_file)
            copied += 1
        return copied

    def _print_summary_table(self, summary: Dict[str, Dict[str, int]]):
        headers = ["Class", "Train", "Val", "Test", "Total"]
        rows = []
        for class_name in sorted(summary.keys()):
            train_n = summary[class_name]["train"]
            val_n = summary[class_name]["val"]
            test_n = summary[class_name]["test"]
            rows.append([class_name, train_n, val_n, test_n, train_n + val_n + test_n])

        widths = [
            max(len(str(row[idx])) for row in [headers] + rows)
            for idx in range(len(headers))
        ]

        def _fmt(row):
            return " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))

        print("\nDataset Split Summary")
        print(_fmt(headers))
        print("-+-".join("-" * w for w in widths))
        for row in rows:
            print(_fmt(row))

    def split(self) -> Dict[str, Dict[str, int]]:
        if self.val_ratio < 0 or self.test_ratio < 0:
            raise ValueError("val_ratio and test_ratio must be >= 0")
        if (self.val_ratio + self.test_ratio) >= 1:
            raise ValueError("val_ratio + test_ratio must be < 1")

        class_dirs = self._list_classes()
        summary: Dict[str, Dict[str, int]] = {}

        random.seed(self.seed)

        for class_dir in class_dirs:
            class_name = class_dir.name
            images = self._list_images_for_class(class_dir)

            if len(images) < 10:
                raise ValueError(
                    f"Class '{class_name}' has {len(images)} images; minimum required is 10"
                )

            random.shuffle(images)

            total = len(images)
            val_count = int(total * self.val_ratio)
            test_count = int(total * self.test_ratio)
            train_count = total - val_count - test_count

            train_files = images[:train_count]
            val_files = images[train_count:train_count + val_count]
            test_files = images[train_count + val_count:]

            copied_train = self._copy_split_files(train_files, "train", class_name)
            copied_val = self._copy_split_files(val_files, "val", class_name)
            copied_test = self._copy_split_files(test_files, "test", class_name)

            summary[class_name] = {
                "train": copied_train,
                "val": copied_val,
                "test": copied_test,
            }

        self._print_summary_table(summary)
        return summary


def main():
    src = "data/dataset"
    dst = "data/dataset_split"
    splitter = DatasetSplitter(src_dir=src, dst_dir=dst)
    splitter.split()
    print("Split complete. Ready for training.")


if __name__ == "__main__":
    main()


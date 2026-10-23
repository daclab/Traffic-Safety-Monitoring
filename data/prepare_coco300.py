"""Prepare Qualcomm's deterministic 300-image COCO val2017 subset."""

import json
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
COCO = PROJECT / "datasets" / "coco"
ANNOTATIONS = COCO / "annotations" / "instances_val2017.json"
SOURCE_IMAGES = COCO / "images" / "val2017"
OUTPUT = Path(__file__).resolve().parent / "coco300"

metadata = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
images = sorted(metadata["images"], key=lambda item: item["id"])
if len(images) != 5000:
    raise RuntimeError(f"Expected 5000 COCO val images, found {len(images)}")

# qai-hub-models BaseDataset.get_dataloader(num_samples=300) uses this stride.
selected = images[0 : (len(images) // 300) * 300 : len(images) // 300]
if len(selected) != 300:
    raise RuntimeError(f"Expected 300 selected images, found {len(selected)}")

image_output = OUTPUT / "images"
image_output.mkdir(parents=True, exist_ok=True)
for number, item in enumerate(selected, 1):
    source = SOURCE_IMAGES / item["file_name"]
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, image_output / item["file_name"])
    if number % 50 == 0:
        print(f"Prepared {number}/300")

manifest = {
    "protocol": "Qualcomm deterministic COCO val2017 300-sample stride",
    "total_val_images": 5000,
    "stride": 16,
    "images": [
        {
            "id": item["id"],
            "file_name": item["file_name"],
            "width": item["width"],
            "height": item["height"],
        }
        for item in selected
    ],
}
(OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Subset ready: {OUTPUT}")


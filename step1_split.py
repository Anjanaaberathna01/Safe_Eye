import os
import random
from PIL import Image, ImageOps

SOURCE_DIR = "data_set"
OUTPUT_DIR = "final_dataset"

CLASSES = [
    "eye_open",
    "eye_close",
    "yawn",
    "no_yawn",
    "drowsy",
    "non_drowsy",
]

IMG_SIZE = (224, 224)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
MAX_PER_CLASS = 2000
SEED = 42

random.seed(SEED)


def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def main():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    for cls in CLASSES:
        cls_folder = os.path.join(SOURCE_DIR, cls)
        if not os.path.exists(cls_folder):
            print(f"❌ '{cls_folder}' is missing. Skipping class '{cls}'.")
            continue

        valid_files = []
        for fname in os.listdir(cls_folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                fpath = os.path.join(cls_folder, fname)
                if is_valid_image(fpath):
                    valid_files.append(fpath)
                else:
                    print(f"❌ Corrupt image skipped: {fpath}")

        random.shuffle(valid_files)

        if len(valid_files) > MAX_PER_CLASS:
            valid_files = valid_files[:MAX_PER_CLASS]

        n = len(valid_files)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)

        splits_data = {
            "train": valid_files[:train_end],
            "val": valid_files[train_end:val_end],
            "test": valid_files[val_end:],
        }

        print(
            f"✅ {cls:10s} -> Used: {n:4d} | "
            f"Train: {len(splits_data['train']):4d} | "
            f"Val: {len(splits_data['val']):3d} | "
            f"Test: {len(splits_data['test']):3d}"
        )

        for split_name, files in splits_data.items():
            for idx, src_path in enumerate(files):
                try:
                    with Image.open(src_path) as img:
                        img = ImageOps.exif_transpose(img)
                        rgb_img = img.convert("RGB").resize(IMG_SIZE, Image.Resampling.BILINEAR)
                        out_name = f"{cls}_{idx:05d}.jpg"
                        out_path = os.path.join(OUTPUT_DIR, split_name, cls, out_name)
                        rgb_img.save(out_path, "JPEG", quality=92)
                except Exception as e:
                    print(f"⚠️ Error saving {src_path}: {e}")

    print("\n🎉 Step 1 complete! 'final_dataset' is ready.")


if __name__ == "__main__":
    main()
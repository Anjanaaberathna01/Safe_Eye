import os
import random
from PIL import Image, ImageOps

TRAIN_DIR = "final_dataset/train"
CLASSES = [
    "eye_open",
    "eye_close",
    "yawn",
    "no_yawn",
    "drowsy",
    "non_drowsy",
]
TARGET_TRAIN_COUNT = 1600
SEED = 42

random.seed(SEED)


def augment_image(img: Image.Image) -> Image.Image:
    mode = random.choice(["flip", "rot_left", "rot_right", "flip_rot"])

    if mode == "flip":
        return ImageOps.mirror(img)
    elif mode == "rot_left":
        angle = random.uniform(5, 15)
        return img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    elif mode == "rot_right":
        angle = random.uniform(-15, -5)
        return img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    else:
        flipped = ImageOps.mirror(img)
        angle = random.uniform(-12, 12)
        return flipped.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))


def count_images(folder):
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


def main():
    for cls in CLASSES:
        cls_path = os.path.join(TRAIN_DIR, cls)
        if not os.path.exists(cls_path):
            print(f"⚠️ Folder not found: {cls_path}")
            continue

        existing_files = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        current_count = len(existing_files)

        if current_count == 0:
            print(f"⚠️ No images in {cls_path}")
            continue

        if current_count >= TARGET_TRAIN_COUNT:
            print(f"✅ {cls:10s}: Already has {current_count} images (target: {TARGET_TRAIN_COUNT}).")
            continue

        needed = TARGET_TRAIN_COUNT - current_count
        print(f"🔄 {cls:10s}: {current_count} -> Generating {needed} augmented images...")

        for i in range(needed):
            src_name = random.choice(existing_files)
            src_path = os.path.join(cls_path, src_name)
            try:
                with Image.open(src_path) as img:
                    rgb_img = img.convert("RGB")
                    aug_img = augment_image(rgb_img)
                    out_name = f"aug_{cls}_{i:05d}.jpg"
                    out_path = os.path.join(cls_path, out_name)
                    aug_img.save(out_path, "JPEG", quality=92)
            except Exception as e:
                print(f"❌ Error augmenting {src_path}: {e}")

        print(f"🎉 {cls:10s}: Final train count = {count_images(cls_path)}")

    print("\n✅ Step 2 complete! All 6 train classes are balanced.")


if __name__ == "__main__":
    main()
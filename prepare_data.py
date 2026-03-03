"""
prepare_data.py — Split flat miniImageNet into train / val / test.

miniImageNet on Kaggle is a flat directory of 100 class folders.

Modes
-----
fsdm (few-shot DM, default):
    Splits classes into 60 / 16 / 20 train / val / test sets using symlinks.
    Train + val are used for few-shot meta-training; test classes are held out.

vanilla:
    Keeps all 100 classes in both train and test.
    - Train : 100 classes × 500 images per class  = 50 000 images
    - Test  : 100 classes × 100 images per class  = 10 000 images
    Images are sampled from each class folder, shuffled with --seed.
    Files are symlinked (or copied with --copy).

Usage (Kaggle notebook):

  # fsdm mode (few-shot split)
  !python prepare_data.py \\
      --mode fsdm \\
      --src /kaggle/input/datasets/arjunashok33/miniimagenet \\
      --dst /kaggle/working/miniimagenet_fsdm \\
      --train 60 --val 16 --test 20

  # vanilla mode (standard classification split)
  !python prepare_data.py \\
      --mode vanilla \\
      --src /kaggle/input/datasets/arjunashok33/miniimagenet \\
      --dst /kaggle/working/miniimagenet_vanilla \\
      --train_per_class 500 --test_per_class 100
"""

import os
import shutil
import argparse
import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _link_or_copy(src: str, dst: str, copy: bool) -> None:
    """Create a symlink dst → src, or copy src to dst."""
    if os.path.exists(dst):
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


# ---------------------------------------------------------------------------
# mode implementations
# ---------------------------------------------------------------------------

def run_fsdm(args) -> None:
    """Few-shot DM mode: split classes into train / val / test subsets."""
    all_classes = sorted([
        d for d in os.listdir(args.src)
        if os.path.isdir(os.path.join(args.src, d))
    ])
    n_total = len(all_classes)
    n_need = args.train + args.val + args.test
    print(f"[fsdm] Found {n_total} classes in {args.src}")
    assert n_total >= n_need, f"Need {n_need} classes but found only {n_total}."

    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(n_total)[:n_need]
    splits = {
        'train': [all_classes[i] for i in indices[:args.train]],
        'val':   [all_classes[i] for i in indices[args.train:args.train + args.val]],
        'test':  [all_classes[i] for i in indices[args.train + args.val:n_need]],
    }

    for split_name, class_list in splits.items():
        split_dir = os.path.join(args.dst, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for cls_name in class_list:
            src_path = os.path.join(args.src, cls_name)
            dst_path = os.path.join(split_dir, cls_name)
            if os.path.exists(dst_path):
                continue
            if args.copy:
                shutil.copytree(src_path, dst_path)
            else:
                os.symlink(os.path.abspath(src_path), dst_path)
        print(f"  {split_name}: {len(class_list)} classes → {split_dir}")

    print(f"\nDone! Use --data_dir {args.dst} when running train.py")
    print(f"\n--- Split details (seed={args.seed}) ---")
    for split_name, class_list in splits.items():
        preview = ', '.join(class_list[:5])
        suffix = '...' if len(class_list) > 5 else ''
        print(f"{split_name} ({len(class_list)}): {preview}{suffix}")


def run_vanilla(args) -> None:
    """Vanilla mode: 100 classes shared by train and test, fixed images-per-class.

    Train : 100 classes × args.train_per_class images  (default 500)
    Test  : 100 classes × args.test_per_class  images  (default 100)
    Total images required per class: train_per_class + test_per_class ≤ images in folder.
    """
    all_classes = sorted([
        d for d in os.listdir(args.src)
        if os.path.isdir(os.path.join(args.src, d))
    ])
    n_total = len(all_classes)
    print(f"[vanilla] Found {n_total} classes in {args.src}")

    rng = np.random.RandomState(args.seed)

    train_total = 0
    test_total = 0

    for cls_name in all_classes:
        cls_src = os.path.join(args.src, cls_name)
        # Collect all image files in the class folder (non-recursive)
        images = sorted([
            f for f in os.listdir(cls_src)
            if os.path.isfile(os.path.join(cls_src, f))
            and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        ])
        n_imgs = len(images)
        n_need = args.train_per_class + args.test_per_class
        assert n_imgs >= n_need, (
            f"Class '{cls_name}' has only {n_imgs} images but need "
            f"{n_need} ({args.train_per_class} train + {args.test_per_class} test)."
        )

        # Shuffle and assign
        perm = rng.permutation(n_imgs)
        train_imgs = [images[i] for i in perm[:args.train_per_class]]
        test_imgs  = [images[i] for i in perm[args.train_per_class:n_need]]

        for split_name, img_list in [('train', train_imgs), ('test', test_imgs)]:
            split_cls_dir = os.path.join(args.dst, split_name, cls_name)
            os.makedirs(split_cls_dir, exist_ok=True)
            for img_name in img_list:
                src_img = os.path.join(cls_src, img_name)
                dst_img = os.path.join(split_cls_dir, img_name)
                _link_or_copy(src_img, dst_img, args.copy)

        train_total += len(train_imgs)
        test_total  += len(test_imgs)

    print(f"  train : {n_total} classes × {args.train_per_class} = {train_total} images → {os.path.join(args.dst, 'train')}")
    print(f"  test  : {n_total} classes × {args.test_per_class}  = {test_total}  images → {os.path.join(args.dst, 'test')}")
    print(f"\nDone! Use --data_dir {args.dst} when running train.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Prepare miniImageNet in fsdm or vanilla mode.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--mode', choices=['fsdm', 'vanilla'], default='fsdm',
                        help='Split mode: "fsdm" (few-shot class split) or "vanilla" (standard classification split).')
    parser.add_argument('--src', required=True, help='Source directory with class folders.')
    parser.add_argument('--dst', required=True, help='Destination directory for split output.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--copy', action='store_true', help='Copy files instead of symlinking.')

    # fsdm-specific
    fsdm_group = parser.add_argument_group('fsdm mode options')
    fsdm_group.add_argument('--train', type=int, default=60, help='[fsdm] Number of training classes.')
    fsdm_group.add_argument('--val',   type=int, default=16, help='[fsdm] Number of validation classes.')
    fsdm_group.add_argument('--test',  type=int, default=20, help='[fsdm] Number of test classes.')

    # vanilla-specific
    van_group = parser.add_argument_group('vanilla mode options')
    van_group.add_argument('--train_per_class', type=int, default=500,
                           help='[vanilla] Images per class in the train split.')
    van_group.add_argument('--test_per_class',  type=int, default=100,
                           help='[vanilla] Images per class in the test split.')

    args = parser.parse_args()

    if args.mode == 'fsdm':
        run_fsdm(args)
    else:
        run_vanilla(args)


if __name__ == '__main__':
    main()

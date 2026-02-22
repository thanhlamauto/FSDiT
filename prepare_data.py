"""
prepare_data.py — Split flat miniImageNet into train / val / test.

miniImageNet on Kaggle is a flat directory of 100 class folders.
This script splits them into 64 / 16 / 20 (or custom) and creates
symlinks so the original data isn't duplicated.

Usage (Kaggle notebook):
    !python prepare_data.py \
        --src /kaggle/input/datasets/arjunashok33/miniimagenet \
        --dst /kaggle/working/miniimagenet_split \
        --train 60 --val 16 --test 20
"""

import os
import shutil
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Split miniImageNet classes into train/val/test.')
    parser.add_argument('--src', required=True, help='Source directory with class folders.')
    parser.add_argument('--dst', required=True, help='Destination directory for split.')
    parser.add_argument('--train', type=int, default=60, help='Number of training classes.')
    parser.add_argument('--val', type=int, default=16, help='Number of validation classes.')
    parser.add_argument('--test', type=int, default=20, help='Number of test classes.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--copy', action='store_true', help='Copy files instead of symlinking.')
    args = parser.parse_args()

    # List all class directories
    all_classes = sorted([
        d for d in os.listdir(args.src)
        if os.path.isdir(os.path.join(args.src, d))
    ])
    n_total = len(all_classes)
    n_need = args.train + args.val + args.test
    print(f"Found {n_total} classes in {args.src}")
    assert n_total >= n_need, f"Need {n_need} classes but found only {n_total}."

    # Shuffle and split
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(n_total)[:n_need]
    splits = {
        'train': [all_classes[i] for i in indices[:args.train]],
        'val':   [all_classes[i] for i in indices[args.train:args.train + args.val]],
        'test':  [all_classes[i] for i in indices[args.train + args.val:n_need]],
    }

    # Create split directories with symlinks (or copies)
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
    
    # Print class assignments for reproducibility
    print(f"\n--- Split details (seed={args.seed}) ---")
    for split_name, class_list in splits.items():
        print(f"{split_name} ({len(class_list)}): {', '.join(class_list[:5])}{'...' if len(class_list) > 5 else ''}")


if __name__ == '__main__':
    main()

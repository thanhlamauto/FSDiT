import os
import random
import numpy as np

# Set environment variables for Kaggle TPU v5e-8
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Hide TPU from TensorFlow to prevent Segfaults when big_vision imports TF
# MUST BE DONE BEFORE IMPORTING JAX
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

import jax
import jax.numpy as jnp
from PIL import Image
# Import from the existing repository
from run_experiments import setup_siglip_jax, encode_images_jax, init_fsdit, generate_sample

def main():
    # 1. Paths
    img_dir = "/kaggle/input/datasets/arjunashok33/miniimagenet/n01558993"
    ckpt_path = "/kaggle/input/models/lucastnguyen/dit-few-shot/flax/default/1/ckpt_step_0050000.pkl"
    out_path = "/kaggle/working/generated_output.png"

    # 2. Pick 5 random images
    all_imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(all_imgs) < 5:
        raise ValueError(f"Need at least 5 images in {img_dir}")
    
    selected_imgs = random.sample(all_imgs, 5)
    print("Selected 5 random images:")
    for img_p in selected_imgs:
        print(f" - {img_p}")

    # 3. Load SigLIP2 Model
    print("\nLoading google/siglip2-base-patch16-224-jax (via big_vision)...")
    siglip_model, siglip_params, txt_tokenizer = setup_siglip_jax()

    # 4. Pass each image through the model to get CLS (pooled) and patch (seq) tokens
    print("\nExtracting CLS and Patch tokens for each image...")
    all_seq = []
    all_pooled = []
    
    # We pass them sequentially as requested ("cho lần lượt 5 ảnh đi qua")
    for img_path in selected_imgs:
        seq, pooled = encode_images_jax([img_path], siglip_model, siglip_params)
        all_seq.append(seq[0])       # shape: (196, 768)
        all_pooled.append(pooled[0]) # shape: (768,)
        
    all_seq = np.stack(all_seq)       # (5, 196, 768)
    all_pooled = np.stack(all_pooled) # (5, 768)
    print(f"Extracted Patch tokens shape: {all_seq.shape}")
    print(f"Extracted CLS tokens shape: {all_pooled.shape}")

    # 5. Pool mean the CLS tokens
    mean_pool = np.mean(all_pooled, axis=0, keepdims=True) # (1, 768)
    
    # We reshape seq tokens to match DiT's expected condition format (1, 5*196, 768)
    mean_seq = all_seq.reshape(1, -1, 768) # (1, 980, 768)
    
    print(f"\nCondition CLS Token (Pooled Mean) shape: {mean_pool.shape}")
    print(f"Condition Patch Tokens shape: {mean_seq.shape}")

    # 6. Initialize DiT with checkpoint
    print(f"\nInitializing Few-Shot DiT with checkpoint: {ckpt_path}...")
    trainer, cfg, decode_fn = init_fsdit(ckpt_path)

    # 7. Generate image using the condition
    print("\nGenerating image with DiT... (Running on TPU v5e-8)")
    seed = random.randint(0, 10000)
    latent = generate_sample(trainer, cfg, mean_pool, mean_seq, seed=seed)
    
    # 8. Decode, print, and save the generated image
    img_arr = decode_fn(latent)
    generated_img = Image.fromarray((img_arr * 255).astype(np.uint8))
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    generated_img.save(out_path)
    print(f"\nDone! Image successfully generated and saved to: {out_path}")
    
    # Note: On Kaggle, you can visualize the saved image via matplotlib/display
    # display(generated_img)

if __name__ == "__main__":
    main()

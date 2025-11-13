"""
download_dataset.py - Download sample images for training

This script downloads sample images from free image sources
that you can use to train the steganalysis model.
"""

import os
import requests
from tqdm import tqdm
from PIL import Image
import io


def download_unsplash_images(num_images=100, output_dir="training_images"):
    """
    Download random images from Unsplash API
    
    Note: You need an Unsplash API key for this.
    Alternative: Use the Picsum method below (no API key needed)
    """
    print("For Unsplash, you need an API key.")
    print("Using Lorem Picsum instead (no API key required)...\n")
    download_picsum_images(num_images, output_dir)


def download_picsum_images(num_images=100, output_dir="training_images", width=512, height=512):
    """
    Download random images from Lorem Picsum (no API key needed)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {num_images} images to '{output_dir}'...")
    print(f"Image size: {width}x{height}\n")
    
    successful = 0
    failed = 0
    
    for i in tqdm(range(num_images), desc="Downloading"):
        try:
            # Random image from Picsum
            url = f"https://picsum.photos/{width}/{height}?random={i}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # Save image
                img = Image.open(io.BytesIO(response.content))
                img_path = os.path.join(output_dir, f"image_{i:04d}.png")
                img.save(img_path, "PNG", quality=95)
                successful += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            continue
    
    print(f"\n✅ Downloaded {successful} images successfully")
    if failed > 0:
        print(f"⚠️  Failed to download {failed} images")
    print(f"\nImages saved to: {output_dir}/")


def download_from_multiple_sources(num_images=100, output_dir="training_images"):
    """
    Download images from multiple free sources
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sources = [
        "https://picsum.photos/512/512",  # Lorem Picsum
        "https://source.unsplash.com/random/512x512",  # Unsplash (limited)
        "https://loremflickr.com/512/512",  # LoremFlickr
    ]
    
    print(f"Downloading {num_images} images from multiple sources...")
    print(f"Output directory: {output_dir}\n")
    
    successful = 0
    
    for i in tqdm(range(num_images), desc="Downloading"):
        try:
            # Rotate through sources
            source = sources[i % len(sources)]
            response = requests.get(source, timeout=10)
            
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                # Resize to consistent size
                img = img.resize((512, 512), Image.LANCZOS)
                img_path = os.path.join(output_dir, f"image_{i:04d}.png")
                img.save(img_path, "PNG", quality=95)
                successful += 1
        except:
            continue
    
    print(f"\n✅ Downloaded {successful} images")


def use_local_images(input_dir, output_dir="training_images", target_count=100):
    """
    Copy and prepare local images for training
    """
    if not os.path.exists(input_dir):
        print(f"❌ Directory not found: {input_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(valid_extensions)
    ]
    
    if len(image_files) == 0:
        print(f"❌ No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images in {input_dir}")
    print(f"Preparing images for training...\n")
    
    for i, img_file in enumerate(tqdm(image_files[:target_count])):
        try:
            img_path = os.path.join(input_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            
            # Resize to consistent size
            img = img.resize((512, 512), Image.LANCZOS)
            
            output_path = os.path.join(output_dir, f"image_{i:04d}.png")
            img.save(output_path, "PNG", quality=95)
        except:
            continue
    
    print(f"\n✅ Prepared {min(len(image_files), target_count)} images")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download training images")
    parser.add_argument(
        "--method",
        type=str,
        default="picsum",
        choices=["picsum", "multiple", "local"],
        help="Download method: picsum, multiple sources, or use local images"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=100,
        help="Number of images to download"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training_images",
        help="Output directory for images"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="my_images",
        help="Local directory with images (for --method local)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Image Dataset Downloader")
    print("="*60 + "\n")
    
    if args.method == "picsum":
        download_picsum_images(args.num_images, args.output_dir)
    elif args.method == "multiple":
        download_from_multiple_sources(args.num_images, args.output_dir)
    elif args.method == "local":
        use_local_images(args.local_dir, args.output_dir, args.num_images)
    
    print("\n" + "="*60)
    print("✅ Dataset preparation complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"  1. Check images in: {args.output_dir}/")
    print(f"  2. Run training: python train_model.py")
    print()
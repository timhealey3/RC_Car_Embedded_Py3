import os
import pandas as pd
from pathlib import Path

def get_existing_images(csv_path):
    """Get a set of image filenames that are already in the training dataset."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return set(df['img'].values)
    return set()

def find_training_images(training_dir):
    """Find all training images in the training directory."""
    training_dir = Path(training_dir)
    return [img.name for img in training_dir.glob('training_*.jpg')]

def append_new_images(csv_path, training_dir):
    """Append new training images to the dataset."""
    # Get existing and new images
    existing_images = get_existing_images(csv_path)
    all_images = find_training_images(training_dir)
    new_images = [img for img in all_images if img not in existing_images]
    
    if not new_images:
        print("No new images to add to the training dataset.")
        return 0
    
    # Create DataFrame for new entries
    new_data = pd.DataFrame([
        {'img': img, 'forward': 1, 'steering_angle': 0}
        for img in new_images
    ])
    
    # Append to existing CSV or create new one
    if os.path.exists(csv_path):
        new_data.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        new_data.to_csv(csv_path, index=False)
    
    print(f"Added {len(new_images)} new images to the training dataset.")
    return len(new_images)

if __name__ == "__main__":
    # Define paths
    script_dir = Path(__file__).parent
    training_dir = script_dir / 'training'
    csv_path = script_dir / 'training_dataset.csv'
    
    # Ensure training directory exists
    training_dir.mkdir(exist_ok=True)
    
    # Append new images
    count = append_new_images(csv_path, training_dir)
    print(f"Processing complete. {count} images were added.")

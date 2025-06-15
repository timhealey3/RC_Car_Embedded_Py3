import os
import pandas as pd
from pathlib import Path
import cv2

def find_training_images(training_dir):
    """Find all training images in the training directory."""
    training_dir = Path(training_dir)
    return [img.name for img in training_dir.glob('training_*.jpg')]

def flipping(img, steering_angle):
    image = cv2.flip(img, 1)
    if steering_angle == 1:
        steering_angle = -1
    elif steering_angle == -1:
        steering_angle = 1
    return image, steering_angle

def append_new_images(csv_path, training_dir):
    """Append new training images to the dataset."""
    # Get existing and new images
    all_images = find_training_images(training_dir)
    training_df = pd.read_csv(csv_path)
    for x in range(len(training_df)):
        image_name = training_df.iloc[x]['img']
        img = cv2.imread(os.path.join(training_dir, training_df.iloc[x]['img']))
        img, steering_angle = flipping(img, training_df.iloc[x]['steering_angle'])
        if steering_angle != 0:
            # Split filename and extension, then add '_flipped' before extension
            img_name = os.path.splitext(training_df.iloc[x]['img'])
            flipped_img_name = f"{img_name[0]}_flipped{img_name[1]}"
            cv2.imwrite(os.path.join(training_dir, flipped_img_name), img)  # Save flipped image with new name
            # Create DataFrame for new entries with the new image name
            new_data = pd.DataFrame([
                {'img': flipped_img_name, 'forward': 1, 'steering_angle': steering_angle}
            ])
            
            # Append to existing CSV or create new one
            if os.path.exists(csv_path):
                new_data.to_csv(csv_path, mode='a', header=False, index=False)
                
            print(f"Added {len(new_data)} new images to the training dataset.")

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

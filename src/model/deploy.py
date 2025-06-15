from model import NeuralNetwork
import os
import cv2
import torch
import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(200, 66)):
    """Preprocess the image to match training format"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    
    # Convert BGR to RGB (if needed)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply same preprocessing as in training
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    # Convert to PyTorch tensor format: (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    return torch.FloatTensor(image).unsqueeze(0)  # Add batch dimension

def main():
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork().to(device)
    
    # Load trained weights
    model_path = os.path.join(os.path.dirname(__file__), '../../models/models', 'models', 'latest_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Example image path - replace with your actual image path
    image_dir = os.path.join(os.path.dirname(__file__), '..', 'camera', 'training')
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found at {image_dir}")
    
    # Get list of images in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        raise FileNotFoundError(f"No images found in {image_dir}")
    
    # Use the first available image
    image_path = os.path.join(image_dir, image_files[0])
    print(f"Using image: {image_path}")
    
    try:
        # Preprocess and predict
        input_tensor = preprocess_image(image_path).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        print(f"Raw output: {output}")
        print(f"Probabilities: {probabilities}")
        print(f"Predicted class: {predicted_class} (0: left, 1: straight, 2: right)")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
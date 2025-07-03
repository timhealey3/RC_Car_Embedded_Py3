from model import NeuralNetwork
import os
import cv2
import torch
import numpy as np
import random

def preprocess_image(image_path):
    """Preprocess the image to match training format"""
    # Read image
    image = cv2.imread(image_path)
    np_img = np.array(image)
    height = np_img.shape[0]
    np_img = np_img[height // 3 + 30:-310, :, :]
    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    np_img = cv2.GaussianBlur(np_img, (3, 3), 0)
    np_img = cv2.resize(np_img, (200, 66))
    np_img = np_img / 255
    return np_img

def main():
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork().to(device)
    
    # Load trained weights
    model_path = os.path.join(os.path.dirname(__file__), '../../models', 'latest_model.pth')
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
    random_num = random.randint(0, 2000)
    image_path = os.path.join(image_dir, image_files[random_num])
    print(f"Using image: {image_path}")
    
    try:
        # Preprocess and predict
        input_tensor = torch.tensor(preprocess_image(image_path), dtype=torch.float32)
        if input_tensor.dim() == 4 and input_tensor.shape[-1] in [1, 3]:  # NHWC to NCHW if needed
            input_tensor = input_tensor.permute(0, 3, 1, 2).contiguous()
        
        # Convert to float32 if needed
        if input_tensor.dtype != torch.float32:
            input_tensor = input_tensor.float()
        
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
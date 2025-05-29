import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from ultrasound_classifier import ShallowCNN, HandcraftedFeatureExtractor
import os


def predict_single_image(model_path, image_path, class_names):
    """
    Predict the class of a single ultrasound image
    
    Args:
        model_path: Path to the trained model (.pth file)
        image_path: Path to the image to classify
        class_names: List of class names
    
    Returns:
        predicted_class: The predicted class name
        confidence: The confidence score
    """
    
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShallowCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()
    
    return predicted_class, confidence_score


def extract_features_demo(image_path):
    """
    Demonstrate feature extraction from an ultrasound image
    
    Args:
        image_path: Path to the image
    
    Returns:
        features: Extracted handcrafted features
    """
    
    # Initialize feature extractor
    extractor = HandcraftedFeatureExtractor()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32) / 255.0
    
    # Extract features
    features = extractor.extract_all_features(image)
    
    print(f"Extracted {len(features)} handcrafted features:")
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"Feature mean: {features.mean():.4f}")
    print(f"Feature std: {features.std():.4f}")
    
    return features


def demo_predictions():
    """
    Demonstrate predictions on sample images from the test set
    """
    
    model_path = 'ultrasound_classifier.pth'
    test_dir = './datasets/img/test'
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    print("=== Ultrasound Image Classification Demo ===\n")
    
    # Test on a few sample images
    for class_name in ['bladder', 'kidney']:
        class_dir = os.path.join(test_dir, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            # Test on first 3 images of each class
            for i, img_name in enumerate(images[:3]):
                img_path = os.path.join(class_dir, img_name)
                
                print(f"Testing image: {img_path}")
                print(f"True class: {class_name}")
                
                # Make prediction
                predicted_class, confidence = predict_single_image(model_path, img_path)
                
                print(f"Predicted class: {predicted_class}")
                print(f"Confidence: {confidence:.4f}")
                print(f"Correct: {'✓' if predicted_class == class_name else '✗'}")
                
                # Extract features
                print("\nFeature extraction:")
                features = extract_features_demo(img_path)
                
                print("-" * 50)
    
    print("\nDemo complete!")


def predict_new_image(image_path, class_names):
    """
    Predict the class of a new image provided by the user
    
    Args:
        image_path: Path to the new image
    """
    
    model_path = 'ultrasound_classifier.pth'
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    
    if not os.path.exists(image_path):
        print(f"Image file {image_path} not found.")
        return
    
    print(f"Analyzing image: {image_path}")
    
    # Make prediction
    predicted_class, confidence = predict_single_image(model_path, image_path, class_names)
    
    print(f"Predicted organ: {predicted_class}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.1f}%)")
    
    # Extract and display features
    print("\nExtracted features:")
    features = extract_features_demo(image_path)


if __name__ == "__main__":
    import sys

    class_names = ['bladder', 'kidney', 'liver', 'gallbladder', 'spleen', 'bowel']
    
    if len(sys.argv) > 1:
        # If an image path is provided as argument, predict that image
        image_path = sys.argv[1]
        predict_new_image(image_path, class_names)
    else:
        # Otherwise, run the demo on test images
        demo_predictions() 
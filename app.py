import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import os

def load_model_and_classes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/best_model.pth"
    
    if not os.path.exists(model_path):
        return None, None, "Model not found. Please train the model first by running `python train.py`."
        
    try:
        checkpoint = torch.load(model_path, map_location=device)
        classes = checkpoint['classes']
        
        model = get_model(num_classes=len(classes), pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model, classes, None
    except Exception as e:
        return None, None, f"Error loading model: {e}"

# Load global model once
model, classes, error_msg = load_model_and_classes()

def predict(image):
    if error_msg:
        return {"Error": 1.0}
    if image is None:
         return {"No image input!": 1.0}   
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Transforms that exactly match training/validation transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Ensure image is in RGB
    image = image.convert("RGB")
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    # Return dictionary of top-K results to Gradio Label interface
    results = {classes[i]: float(prob) for i, prob in enumerate(probabilities)}
    return results

if __name__ == "__main__":
    print("Starting Gradio Web UI...")
    
    interface_title = "🌿 Plant Disease Detection System"
    interface_desc = (
        "Upload an image of a plant leaf to identify potential diseases using our Convolutional Neural Network."
        "\nIf you haven't trained it yet, download the dataset using `python download_data.py` and run `python train.py`."
    )
    
    interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload Leaf Image"),
        outputs=gr.Label(num_top_classes=3, label="Predictions"),
        title=interface_title,
        description=interface_desc,
        theme="huggingface"
    )
    
    # For local inference, allow listening on 0.0.0.0
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)

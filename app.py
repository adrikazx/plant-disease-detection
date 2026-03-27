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
    
    # Utilizing a modern Blocks layout (default theme to prevent loading errors)
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # 🌱 Advanced Plant Disease AI Diagnostic System
            Welcome to your **ResNet18-powered diagnostic tool**. Please upload a high-resolution image of a plant leaf below to detect anomalies and identify potential diseases with top-3 confidence scores.
            """
        )
        
        with gr.Row():
            # Left column for the input
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Leaf Image here ➔")
                with gr.Row():
                    clear_btn = gr.Button("Clear Image")
                    submit_btn = gr.Button("🔍 Diagnose Plant", variant="primary")
            
            # Right column for the AI's output
            with gr.Column(scale=1):
                label_output = gr.Label(num_top_classes=3, label="AI Confidence Levels")
                
        # Link the buttons to their actions!
        submit_btn.click(fn=predict, inputs=image_input, outputs=label_output)
        clear_btn.click(fn=lambda: None, inputs=None, outputs=image_input)
        
    # Launch on all interfaces with a public internet link!
    interface.launch(server_name="127.0.0.1", server_port=7860, share=True)

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel  # Adjust based on your model type

# Download the model from Hugging Face
model_name = "Shitao/omnigen-v1"  # Replace with your actual model name
snapshot_download(model_name)

# Load the model
model = AutoModel.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode

# Create a dummy input tensor with the appropriate shape
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust shape as needed

# Export the model to ONNX format
onnx_path = 'your_model.onnx'
torch.onnx.export(model, dummy_input, onnx_path, export_params=True)
print('Model exported to ONNX format successfully!')

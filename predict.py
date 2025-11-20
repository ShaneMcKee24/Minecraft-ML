# prediction.py
import torch
from PIL import Image
from torchvision import transforms
import main  # import the main.py script

def predict_image(img_path):
    model = main.model        # reference the model object created in main.py
    device = main.device      # reference the device
    class_map = main.class_map

    idx_to_class = {v:k for k,v in class_map.items()}

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    print(f"Predicted: {idx_to_class[pred.item()]}")

input_image_path = "Tester.png"
predict_image(input_image_path)
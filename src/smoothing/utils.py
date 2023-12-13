import torch
import torchvision.models as models
from torchvision.transforms import functional as F
from torchvision.ops import nms
from PIL import ImageDraw, ImageFont


def load_model(weights_path):
    device = torch.device('cpu')
    # Load a pre-trained Faster R-CNN model or your own trained model
    model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(num_classes=14)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, device


def predict_and_nms(model, image, device, confidence_threshold=0.5, nms_threshold=0.3):
    # Transform and add batch dimension
    img = F.to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(img)

    # Filter out low-confidence predictions
    pred_boxes = prediction[0]['boxes']
    pred_scores = prediction[0]['scores']
    pred_labels = prediction[0]['labels']

    high_conf_indices = pred_scores > confidence_threshold
    high_conf_boxes = pred_boxes[high_conf_indices]
    high_conf_scores = pred_scores[high_conf_indices]
    high_conf_labels = pred_labels[high_conf_indices]

    # Apply NMS
    keep_indices = nms(high_conf_boxes, high_conf_scores, nms_threshold)

    final_boxes = high_conf_boxes[keep_indices].cpu()
    final_labels = high_conf_labels[keep_indices].cpu()
    final_scores = high_conf_scores[keep_indices].cpu()

    return final_boxes, final_labels, final_scores


def visualize_predictions(image, boxes, labels, label_map):
    # Convert PIL image to draw context
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label in zip(boxes, labels):
        x, y, x2, y2 = box
        label_name = label_map.get(label.item(), 'Unknown')

        # Draw rectangle
        draw.rectangle([(x, y), (x2, y2)], outline='red', width=2)

        # Draw label
        text_size = font.getsize(label_name)
        draw.rectangle([(x, y - text_size[1]), (x + text_size[0], y)], fill='red')
        draw.text((x, y - text_size[1]), label_name, fill='white', font=font)

    return image

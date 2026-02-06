import argparse
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import os
import time

from src.model import get_model_instance_segmentation


def get_transform():
    return T.Compose(
        [
            T.ToTensor(),
        ]
    )


def load_model(model_path, num_classes, device):
    model = get_model_instance_segmentation(num_classes)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Model path {model_path} not found. Using initialized weights (random)."
        )

    model.to(device)
    model.eval()
    return model


def predict_image(model, image_path, device, threshold=0.5):
    img = Image.open(image_path).convert("RGB")
    transform = get_transform()
    img_tensor = transform(img).to(device)

    with torch.no_grad():
        start_time = time.time()
        # Model expects a list of tensors
        prediction = model([img_tensor])
        end_time = time.time()

    print(f"Inference time: {end_time - start_time:.4f}s")

    # Process prediction
    pred = prediction[0]

    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()

    # Filter by threshold
    keep = scores >= threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    return img, boxes, labels, scores


def draw_prediction(img, boxes, labels, scores):
    draw = ImageDraw.Draw(img)
    # Mapping: 0=background (not used), 1=with_mask, 2=without_mask, 3=incorrect
    label_map = {1: "With Mask", 2: "Without Mask", 3: "Incorrect Mask"}
    color_map = {1: "green", 2: "red", 3: "orange"}

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box
        label_text = label_map.get(label, f"Class {label}")
        color = color_map.get(label, "blue")

        # Draw box
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=3)

        # Draw label
        text = f"{label_text}: {score:.2f}"

        # Calculate text size using textbbox (for Pillow >= 9.2.0) or generic estimation
        if hasattr(draw, "textbbox"):
            text_bbox = draw.textbbox((xmin, ymin), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        else:
            text_w, text_h = draw.textsize(text, font=font)

        draw.rectangle(
            [(xmin, ymin - text_h - 4), (xmin + text_w + 4, ymin)], fill=color
        )
        draw.text((xmin + 2, ymin - text_h - 2), text, fill="white", font=font)

    return img


def main():
    parser = argparse.ArgumentParser(description="Face Mask Detection Inference")
    parser.add_argument(
        "--input", required=True, help="Path to input image or directory"
    )
    parser.add_argument(
        "--model", default="models/model_best.pth", help="Path to trained model weights"
    )
    parser.add_argument("--output", default="output", help="Directory to save results")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Confidence threshold"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    # 3 Classes + Background
    num_classes = 4
    device = torch.device(args.device)

    model = load_model(args.model, num_classes, device)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    inputs = []
    if os.path.isdir(args.input):
        for f in os.listdir(args.input):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                inputs.append(os.path.join(args.input, f))
    else:
        inputs.append(args.input)

    for img_path in inputs:
        print(f"Processing {img_path}...")
        try:
            orig_img, boxes, labels, scores = predict_image(
                model, img_path, device, args.threshold
            )
            result_img = draw_prediction(orig_img, boxes, labels, scores)

            filename = os.path.basename(img_path)
            save_path = os.path.join(args.output, f"pred_{filename}")
            result_img.save(save_path)
            print(f"Saved result to {save_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    main()

import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from ultralytics import YOLO


def get_faster_rcnn_mobilenet(num_classes):
    """
    Faster R-CNN with MobileNetV3-Large 320 FPN backbone.
    Fast training, good for mobile/edge.
    """
    model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_faster_rcnn_resnet50(num_classes):
    """
    Faster R-CNN with ResNet50 FPN backbone.
    Standard high-accuracy benchmark.
    """
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_retinanet_resnet50(num_classes):
    """
    RetinaNet with ResNet50 FPN backbone.
    Single-stage detector (like YOLO), good trade-off.
    """
    model = retinanet_resnet50_fpn(pretrained=True)
    # Replace head
    in_features = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes

    # We need to rebuild the Classification Head mostly
    # torchvision makes this a bit tricky compared to Faster R-CNN
    # But for simple class change, we re-init the whole head usually or just the final layer logic
    # Easier way:
    model = retinanet_resnet50_fpn(
        num_classes=num_classes,
        pretrained=False,
        weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
    )
    return model


class YOLOWrapper:
    """
    Wrapper for YOLOv8 using Ultralytics.
    Note: YOLOv8 follows a separate API (model.train, model.predict)
    and does not return standard torch loss dicts in the same way for custom loops.

    This wrapper provides the model object that can be trained using .train() method.
    """

    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)

    def train(self, data_yaml, epochs=10):
        return self.model.train(data=data_yaml, epochs=epochs)

    def __call__(self, *args, **kwargs):
        # Pass through to pure model if needed, but usually we use .predict
        return self.model(*args, **kwargs)


def get_model(model_name, num_classes):
    """
    Factory function to get model instance.

    Args:
        model_name (str): One of ['fasterrcnn_mobilenet', 'fasterrcnn_resnet50', 'retinanet', 'yolov8n', 'yolov8s']
        num_classes (int): Number of classes (including background for torch models)
    """
    model_name = model_name.lower()

    if model_name == "fasterrcnn_mobilenet":
        return get_faster_rcnn_mobilenet(num_classes)
    elif model_name == "fasterrcnn_resnet50":
        return get_faster_rcnn_resnet50(num_classes)
    elif model_name == "retinanet":
        return get_retinanet_resnet50(num_classes)
    elif "yolo" in model_name:
        # e.g. yolov8n, yolov8s.pt
        if not model_name.endswith(".pt"):
            model_name += ".pt"
        print(
            f"Loading YOLO model: {model_name}. Note: Use .train() method directly similar to notebook usage."
        )
        return YOLOWrapper(model_name)
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. Available: fasterrcnn_mobilenet, fasterrcnn_resnet50, retinanet, yolov8n"
        )


# Backwards compatibility
def get_model_instance_segmentation(num_classes):
    return get_faster_rcnn_mobilenet(num_classes)

import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO
import mlflow
import math
import sys
import os
import time
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


# Helper function for manual training loop inside wrapper
def train_one_epoch(
    model, optimizer, data_loader, device, epoch, print_freq=10, scaler=None
):
    model.train()
    metric_logger = {}  # Simple dict logger
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    # Use tqdm for progress bar
    pbar = tqdm(data_loader, desc=header)

    epoch_losses = []

    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Mixed Precision Context
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        epoch_losses.append(loss_value)
        pbar.set_postfix({"loss": f"{loss_value:.4f}"})

    # Return average loss for the epoch
    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    return {"loss": avg_loss}


def match_boxes(pred_boxes, true_boxes, iou_threshold=0.5):
    """
    Matches predicted boxes to true boxes based on IoU.
    Returns a list of (pred_idx, true_idx) tuples.
    """
    if len(pred_boxes) == 0 or len(true_boxes) == 0:
        return []

    ious = torchvision.ops.box_iou(pred_boxes, true_boxes)
    matches = []

    # Greedy matching
    # Sort by IoU to prioritize best matches? Or just iterate.
    # Simple greedy: iterate preds, find max IoU with unused true.
    # But usually we map each pred to best true, or true to best pred?
    # Standard metric: For each pred, is there a true box with IoU > thresh?

    # We will iterate through preds and match to available true boxes
    # To avoid double counting true boxes, we track used indices

    # Better greedy strategy: find max IoU in entire matrix, match, remove, repeat.
    # But for simple CM, let's just find max IoU for each pred, and if > thresh, check if true is used.

    # Actually, let's use a simpler per-pred check like in the notebook
    # Note: This is an approximation for CM.
    # Strict mAP requires sorting by confidence. Here we just want "What did this pred match to?"

    # Let's do: For each pred, find max IoU. If > 0.5, it's a candidate.
    # If multiple preds match same true, only one counts (usually highest conf).
    # Since we didn't sort by score here (though they should be high conf from filtering),
    # let's do a simple greedy match.

    # clone to modify
    ious = ious.clone()

    for _ in range(len(pred_boxes)):
        # Find global max
        if ious.numel() == 0:
            break
        val, idx = ious.flatten().max(0)
        if val < iou_threshold:
            break

        # Convert 1D idx to 2D
        pred_idx = idx // ious.size(1)
        true_idx = idx % ious.size(1)

        matches.append((pred_idx.item(), true_idx.item()))

        # Mask out row and col
        ious[pred_idx, :] = -1
        ious[:, true_idx] = -1

    return matches


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision()

    # First pass: Compute Loss (Requires Train Mode for R-CNN)
    model.train()
    losses = []
    for images, targets in tqdm(data_loader, desc="Validation (Loss)"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses.append(sum(loss for loss in loss_dict.values()).item())

    avg_loss = sum(losses) / len(losses) if losses else 0

    # Second pass: Compute Metrics (Requires Eval Mode)
    model.eval()

    all_preds_cls = []
    all_true_cls = []

    for images, targets in tqdm(data_loader, desc="Validation (Metrics)"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        preds = model(images)

        # Filter predictions for metric calculation (Speed up & Fix mAP 0.0 issue)
        filtered_preds = []
        for p in preds:
            # Standard COCO threshold is often 0.05 confidence for evaluation
            # Limiting to top 100 detections per image is also standard
            keep = p["scores"] > 0.05
            filtered_preds.append(
                {
                    "boxes": p["boxes"][keep],
                    "scores": p["scores"][keep],
                    "labels": p["labels"][keep],
                }
            )

        metric.update(filtered_preds, targets)

        # Confusion Matrix Accumulation
        for i, pred in enumerate(preds):
            target = targets[i]

            true_boxes = target["boxes"]
            true_labels = target["labels"]

            pred_boxes = pred["boxes"]
            pred_labels = pred["labels"]
            pred_scores = pred["scores"]

            # Filter low confidence for CM
            keep = pred_scores > 0.5
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]

            matches = match_boxes(pred_boxes, true_boxes)

            matched_pred_indices = set()
            matched_true_indices = set()

            for p_idx, t_idx in matches:
                all_preds_cls.append(pred_labels[p_idx].item())
                all_true_cls.append(true_labels[t_idx].item())
                matched_pred_indices.add(p_idx)
                matched_true_indices.add(t_idx)

            # False Negatives (Missed matches) -> Target exists, Pred is Background (0)
            for t_idx in range(len(true_labels)):
                if t_idx not in matched_true_indices:
                    all_true_cls.append(true_labels[t_idx].item())
                    all_preds_cls.append(0)

            # False Positives (Spurious) -> Pred exists, True is Background (0)
            for p_idx in range(len(pred_labels)):
                if p_idx not in matched_pred_indices:
                    all_true_cls.append(0)
                    all_preds_cls.append(pred_labels[p_idx].item())

    m_metrics = metric.compute()

    results = {"val_loss": avg_loss}

    # Confusion Matrix Plotting
    if all_true_cls and all_preds_cls:
        # Determine unique labels dynamically
        unique_labels = sorted(list(set(all_true_cls) | set(all_preds_cls)))
        max_id = max(unique_labels) if unique_labels else 0
        cm_labels = list(range(max_id + 1))

        cm = confusion_matrix(all_true_cls, all_preds_cls, labels=cm_labels)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        # Save to temp
        import tempfile

        # Create temp file, close it so we can read/write by name safely if needed,
        # but here we just need name. delete=False to keep it for logging.
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)  # Close file descriptor

        plt.savefig(path)
        plt.close(fig)
        results["cm_path"] = path

    # Add mAP metrics to results
    for k, v in m_metrics.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                results[k] = v.item()
        else:
            results[k] = v
    return results


# Wrapper for TorchVision Models to match YOLO API
class TorchVisionWrapper:
    def __init__(self, model, model_name="fasterrcnn"):
        self.model = model
        self.model_name = model_name

    def to(self, device):
        self.model.to(device)
        return self

    def train(
        self,
        data_loader,
        test_loader,
        epochs,
        device,
        project=None,
        name=None,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
        step_size=3,
        gamma=0.1,
        optimizer_name="SGD",
        **kwargs,
    ):
        """
        Standardized training loop for TorchVision Object Detection models.
        """
        print(f"Starting training for {self.model_name}...")

        # Log Hyperparameters
        mlflow.log_param("lr", lr)
        mlflow.log_param("momentum", momentum)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("step_size", step_size)
        mlflow.log_param("gamma", gamma)
        mlflow.log_param("optimizer_name", optimizer_name)

        # Optimizer & Scheduler (Standard defaults)
        params = [p for p in self.model.parameters() if p.requires_grad]

        if optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                params, lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        elif optimizer_name == "Adam":
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Simple step LR
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

        results = {}

        # Scaler for AMP
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

        for epoch in range(1, epochs + 1):
            # Train one epoch
            train_metrics = train_one_epoch(
                self.model,
                optimizer,
                data_loader,
                device,
                epoch,
                print_freq=10,
                scaler=scaler,
            )

            lr_scheduler.step()

            # Log Train Metrics
            mlflow.log_metric("train_loss", train_metrics["loss"], step=epoch)
            print(f"Epoch {epoch + 1}/{epochs} Train Loss: {train_metrics['loss']:.4f}")

            # Validation
            if test_loader:
                val_metrics = evaluate(self.model, test_loader, device)

                # Log CM artifact if it exists
                if "cm_path" in val_metrics:
                    try:
                        mlflow.log_artifact(
                            val_metrics["cm_path"], artifact_path="confusion_matrices"
                        )
                        os.remove(val_metrics["cm_path"])  # cleanup
                    except Exception as e:
                        print(f"Failed to log CM: {e}")
                    del val_metrics["cm_path"]  # Remove from metrics dict

                # Log all metrics
                for k, v in val_metrics.items():
                    mlflow.log_metric(k, v, step=epoch)

                # Print key metrics
                map_val = val_metrics.get("map", 0)
                map_50 = val_metrics.get("map_50", 0)
                print(
                    f"Epoch {epoch + 1}/{epochs} Val Loss: {val_metrics['val_loss']:.4f} | mAP: {map_val:.4f} | mAP50: {map_50:.4f}"
                )
                results = val_metrics

        # Save model
        if project and name:
            save_dir = os.path.join(project, name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "weights.pth")
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        return results

    def val(self, **kwargs):
        return {}

    def eval(self):
        self.model.eval()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def get_faster_rcnn_mobilenet(num_classes):
    """
    Faster R-CNN with MobileNetV3-Large 320 FPN backbone.
    Fast training, good for mobile/edge.
    """
    model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)

    # Handle potentially different structure of cls_score (Linear vs Sequential/Conv)
    cls_score = model.roi_heads.box_predictor.cls_score
    if hasattr(cls_score, "in_features"):
        in_features = cls_score.in_features
    elif hasattr(cls_score, "in_channels"):
        in_features = cls_score.in_channels
    elif isinstance(cls_score, torch.nn.Sequential):
        if hasattr(cls_score[0], "in_features"):
            in_features = cls_score[0].in_features
        elif hasattr(cls_score[0], "in_channels"):
            in_features = cls_score[0].in_channels
        else:
            raise AttributeError(
                f"Could not determine in_features from {type(cls_score[0])}"
            )
    else:
        raise AttributeError(f"Could not determine in_features from {type(cls_score)}")

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_faster_rcnn_resnet50(num_classes):
    """
    Faster R-CNN with ResNet50 FPN backbone.
    Standard high-accuracy benchmark.
    """
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Handle potentially different structure of cls_score (Linear vs Sequential/Conv)
    cls_score = model.roi_heads.box_predictor.cls_score
    if hasattr(cls_score, "in_features"):
        in_features = cls_score.in_features
    elif hasattr(cls_score, "in_channels"):
        in_features = cls_score.in_channels
    elif isinstance(cls_score, torch.nn.Sequential):
        if hasattr(cls_score[0], "in_features"):
            in_features = cls_score[0].in_features
        elif hasattr(cls_score[0], "in_channels"):
            in_features = cls_score[0].in_channels
        else:
            raise AttributeError(
                f"Could not determine in_features from {type(cls_score[0])}"
            )
    else:
        raise AttributeError(f"Could not determine in_features from {type(cls_score)}")

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_retinanet_resnet50(num_classes):
    """
    RetinaNet with ResNet50 FPN backbone.
    Single-stage detector (like YOLO), good trade-off.
    """
    model = retinanet_resnet50_fpn(pretrained=True)

    # Handle different torchvision versions where conv might be Conv2d or Conv2dNormActivation
    first_layer = model.head.classification_head.conv[0]
    if hasattr(first_layer, "in_channels"):
        in_features = first_layer.in_channels
    elif isinstance(first_layer, torch.nn.Sequential):  # Conv2dNormActivation
        in_features = first_layer[0].in_channels
    else:
        # Fallback if structure is unexpected
        try:
            in_features = first_layer[0].in_channels
        except:
            raise AttributeError(
                f"Could not determine in_channels from {type(first_layer)}"
            )

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

    def train(self, **kwargs):
        return self.model.train(**kwargs)

    def val(self, **kwargs):
        return self.model.val(**kwargs)

    def __call__(self, *args, **kwargs):
        # Pass through to pure model if needed, but usually we use .predict
        # Handle cases where args might be dict (training mode for R-CNN style calls? No, YOLO handles differently)
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
        model = get_faster_rcnn_mobilenet(num_classes)
        return TorchVisionWrapper(model, model_name)
    elif model_name == "fasterrcnn_resnet50":
        model = get_faster_rcnn_resnet50(num_classes)
        return TorchVisionWrapper(model, model_name)
    elif model_name == "retinanet":
        model = get_retinanet_resnet50(num_classes)
        return TorchVisionWrapper(model, model_name)
    elif "yolo" in model_name:
        # e.g. yolov8n, yolov8s.pt
        if not model_name.endswith(".pt"):
            model_name += ".pt"
        print(f"Loading YOLO model: {model_name}. Use .train() method.")
        return YOLOWrapper(model_name)
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. Available: fasterrcnn_mobilenet, fasterrcnn_resnet50, retinanet, yolov8n"
        )


# Backwards compatibility
def get_model_instance_segmentation(num_classes):
    return get_faster_rcnn_mobilenet(num_classes)

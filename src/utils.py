import os
import shutil
import xml.etree.ElementTree as ET
import yaml
from tqdm import tqdm


def convert_to_yolo_format(data_dir, label_map, target_yaml_path="data.yaml"):
    """
    Converts a dataset with XML annotations (VOC-like) to YOLO format.
    Structure expected:
    data_dir/
      images/
      annotations/

    Creates:
    data_dir/
      labels/ (YOLO txt files)
    data.yaml
    """
    images_dir = os.path.join(data_dir, "images")
    xmls_dir = os.path.join(data_dir, "annotations")
    labels_dir = os.path.join(data_dir, "labels")

    os.makedirs(labels_dir, exist_ok=True)

    # Reverse label map for name lookup if needed, but here we just need name -> id
    # YOLO keys are 0-indexed integers.
    # The existing label_map in dataset.py might be 1-indexed for Faster R-CNN (Background=0).
    # YOLO doesn't use a background class idx 0. It uses 0..N-1.
    # So we need to remap or ensure consistency.
    # Let's assume input label_map is {'mask': 1, ...}. We will map 1->0, 2->1 etc for YOLO if strict 0-indexing is needed.
    # Actually, let's just create a direct map for YOLO:
    # 0: with_mask, 1: without_mask, 2: mask_weared_incorrect

    yolo_map = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 2}

    xml_files = [f for f in os.listdir(xmls_dir) if f.endswith(".xml")]

    print(f"Converting {len(xml_files)} files to YOLO format...")

    for xml_file in tqdm(xml_files):
        tree = ET.parse(os.path.join(xmls_dir, xml_file))
        root = tree.getroot()

        # Get image dimensions from XML usually
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        filename = root.find("filename").text
        # Ensure filename matches correct extension if not consistent, but usually it is.
        # But we need the txt filename
        txt_filename = os.path.splitext(xml_file)[0] + ".txt"

        with open(os.path.join(labels_dir, txt_filename), "w") as f:
            for obj in root.findall("object"):
                name = obj.find("name").text
                if name not in yolo_map:
                    continue

                cls_id = yolo_map[name]

                xmlbox = obj.find("bndbox")
                xmin = float(xmlbox.find("xmin").text)
                xmax = float(xmlbox.find("xmax").text)
                ymin = float(xmlbox.find("ymin").text)
                ymax = float(xmlbox.find("ymax").text)

                # Normalize
                b_center_x = (xmin + xmax) / 2.0 / w
                b_center_y = (ymin + ymax) / 2.0 / h
                b_width = (xmax - xmin) / w
                b_height = (ymax - ymin) / h

                f.write(
                    f"{cls_id} {b_center_x:.6f} {b_center_y:.6f} {b_width:.6f} {b_height:.6f}\n"
                )

    # Create data.yaml
    # We need to split train/val. YOLO usually takes paths to directories or txt files with paths.
    # For simplicity in this hybrid setup, we can point 'train' and 'val' to the same folder
    # and let YOLO split or just validate on same (not ideal ML but keeps dir structure simple compliant with request).
    # OR better: The user's notebook splits logic in code (Dataset class).
    # YOLO `train` command usually expects a structure.
    # Let's define the yaml to point to the images dir.

    data_config = {
        "path": os.path.abspath(data_dir),  # dataset root dir
        "train": "images",  # relative to 'path'
        "val": "images",  # relative to 'path', usage dependent on splitting
        "names": {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrect"},
    }

    with open(target_yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"YOLO format conversion complete. YAML saved to {target_yaml_path}")
    return os.path.abspath(target_yaml_path)
    print(f"YOLO format conversion complete. YAML saved to {target_yaml_path}")
    return os.path.abspath(target_yaml_path)


def stratified_yolo_split(dataset, output_root, split_ratios=(0.7, 0.2, 0.1), seed=42):
    """
    Performs object-based greedy stratified split and prepares YOLO directory structure.

    Args:
        dataset: FaceMaskDataset instance
        output_root: Root directory for YOLO dataset
        split_ratios: Tuple of (train, val, test) ratios

    Returns:
        (train_indices, val_indices, test_indices)
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    # 0. Setup Directories
    splits = ["train", "val", "test"]
    for s in splits:
        os.makedirs(os.path.join(output_root, "images", s), exist_ok=True)
        os.makedirs(os.path.join(output_root, "labels", s), exist_ok=True)

    # 1. Analyze Dataset (Count objects per image)
    print("Analyzing dataset for stratified split...")
    image_infos = []
    class_counts = {}  # Total counts per class

    # We need to map class names to IDs consistent with our config
    # Internal map:
    label_map = dataset.label_map
    # YOLO map (0-indexed):
    yolo_map = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 2}

    for idx in tqdm(range(len(dataset)), desc="Scanning for Stratification"):
        xml_filename = dataset.xmls[idx]
        xml_path = os.path.join(dataset.annot_dir, xml_filename)

        info = {"idx": idx, "counts": {0: 0, 1: 0, 2: 0}, "total": 0}

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                name = obj.find("name").text
                if name in yolo_map:
                    cid = yolo_map[name]
                    info["counts"][cid] += 1
                    info["total"] += 1
                    class_counts[cid] = class_counts.get(cid, 0) + 1
        except Exception:
            pass

        image_infos.append(info)

    # 2. Assign Splits (Greedy Strategy)
    # Sort images by number of objects (descending) and then by rarity (optional refinement)
    # For simplicity, shuffling then greedy allocation works well for large datasets
    # But for imbalance, sorting by rarest class count is better.
    # Let's simple shuffle first to avoid file-order bias, then greedy allocate.
    random.shuffle(image_infos)

    # Track current counts in each split
    split_counts = {
        "train": {0: 0, 1: 0, 2: 0},
        "val": {0: 0, 1: 0, 2: 0},
        "test": {0: 0, 1: 0, 2: 0},
    }

    # Target ratios
    target_ratios = {
        "train": split_ratios[0],
        "val": split_ratios[1],
        "test": split_ratios[2],
    }

    # Indices lists
    split_indices = {"train": [], "val": [], "test": []}

    print("Allocating images to splits...")
    for info in tqdm(image_infos, desc="Stratifying"):
        idx = info["idx"]
        counts = info["counts"]

        # Identify classes present in this image
        present_classes = [c for c, n in counts.items() if n > 0]

        best_split = None

        if not present_classes:
            # If no objects, assign based on split capacity (fill the one most behind in image count)
            # Normalized by target ratio
            def get_capacity_score(s):
                # Avoid division by zero
                ratio = target_ratios[s] if target_ratios[s] > 0 else 0.001
                # current_fill = num_images / ratio
                return len(split_indices[s]) / ratio

            best_split = min(splits, key=get_capacity_score)
        else:
            # Assign to the split that is least saturated for the classes in this image
            # Saturation = current_count / desired_count
            def get_saturation_score(s):
                scores = []
                for c in present_classes:
                    total_c = class_counts.get(c, 0)
                    if total_c == 0:
                        continue

                    desired = total_c * target_ratios[s]
                    if desired < 1:
                        desired = 1

                    current = split_counts[s][c]
                    scores.append(current / desired)

                # Average saturation for relevant classes
                return sum(scores) / len(scores) if scores else 0

            best_split = min(splits, key=get_saturation_score)

        # Assign
        split_indices[best_split].append(idx)
        for c in [0, 1, 2]:
            split_counts[best_split][c] += counts.get(c, 0)

    # 3. Process Files (Copy & Convert)
    for s in splits:
        indices = split_indices[s]

        img_dest = os.path.join(output_root, "images", s)
        lbl_dest = os.path.join(output_root, "labels", s)

        for idx in tqdm(indices, desc=f"Processing {s}"):
            img_filename = dataset.imgs[idx]
            xml_filename = dataset.xmls[idx]

            src_img = os.path.join(dataset.img_dir, img_filename)
            src_xml = os.path.join(dataset.annot_dir, xml_filename)

            # Copy Image
            if os.path.exists(src_img):
                shutil.copy(src_img, os.path.join(img_dest, img_filename))

            # Convert XML
            if os.path.exists(src_xml):
                try:
                    tree = ET.parse(src_xml)
                    root = tree.getroot()
                    size = root.find("size")
                    w = int(size.find("width").text)
                    h = int(size.find("height").text)

                    txt_name = os.path.splitext(img_filename)[0] + ".txt"
                    with open(os.path.join(lbl_dest, txt_name), "w") as f:
                        for obj in root.findall("object"):
                            name = obj.find("name").text
                            if name in yolo_map:
                                cid = yolo_map[name]
                                bnd = obj.find("bndbox")
                                xmin = float(bnd.find("xmin").text)
                                xmax = float(bnd.find("xmax").text)
                                ymin = float(bnd.find("ymin").text)
                                ymax = float(bnd.find("ymax").text)

                                x = (xmin + xmax) / 2.0 / w
                                y = (ymin + ymax) / 2.0 / h
                                bw = (xmax - xmin) / w
                                bh = (ymax - ymin) / h

                                f.write(f"{cid} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")
                except:
                    pass

    # 4. Generate data.yaml (Correct Structure)
    data_yaml = {
        "path": os.path.abspath(output_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrect"},
        "nc": 3,
    }

    yaml_path = os.path.join(output_root, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"Stratified Split Complete. YAML at {yaml_path}")
    return split_indices["train"], split_indices["val"], split_indices["test"]

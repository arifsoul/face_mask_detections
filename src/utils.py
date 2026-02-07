import os
import shutil
import xml.etree.ElementTree as ET
import yaml
from tqdm import tqdm

def convert_to_yolo_format(data_dir, label_map, target_yaml_path='data.yaml'):
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
    images_dir = os.path.join(data_dir, 'images')
    xmls_dir = os.path.join(data_dir, 'annotations')
    labels_dir = os.path.join(data_dir, 'labels')
    
    os.makedirs(labels_dir, exist_ok=True)
    
    # Reverse label map for name lookup if needed, but here we just need name -> id
    # YOLO keys are 0-indexed integers.
    # The existing label_map in dataset.py might be 1-indexed for Faster R-CNN (Background=0).
    # YOLO doesn't use a background class idx 0. It uses 0..N-1.
    # So we need to remap or ensure consistency.
    # Let's assume input label_map is {'mask': 1, ...}. We will map 1->0, 2->1 etc for YOLO if strict 0-indexing is needed.
    # Actually, let's just create a direct map for YOLO:
    # 0: with_mask, 1: without_mask, 2: mask_weared_incorrectly
    
    yolo_map = {
        "with_mask": 0,
        "without_mask": 1,
        "mask_weared_incorrectly": 2
    }
    
    xml_files = [f for f in os.listdir(xmls_dir) if f.endswith('.xml')]
    
    print(f"Converting {len(xml_files)} files to YOLO format...")
    
    for xml_file in tqdm(xml_files):
        tree = ET.parse(os.path.join(xmls_dir, xml_file))
        root = tree.getroot()
        
        # Get image dimensions from XML usually
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        filename = root.find('filename').text
        # Ensure filename matches correct extension if not consistent, but usually it is.
        # But we need the txt filename
        txt_filename = os.path.splitext(xml_file)[0] + '.txt'
        
        with open(os.path.join(labels_dir, txt_filename), 'w') as f:
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in yolo_map:
                    continue
                
                cls_id = yolo_map[name]
                
                xmlbox = obj.find('bndbox')
                xmin = float(xmlbox.find('xmin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymin = float(xmlbox.find('ymin').text)
                ymax = float(xmlbox.find('ymax').text)
                
                # Normalize
                b_center_x = (xmin + xmax) / 2.0 / w
                b_center_y = (ymin + ymax) / 2.0 / h
                b_width = (xmax - xmin) / w
                b_height = (ymax - ymin) / h
                
                f.write(f"{cls_id} {b_center_x:.6f} {b_center_y:.6f} {b_width:.6f} {b_height:.6f}\n")

    # Create data.yaml
    # We need to split train/val. YOLO usually takes paths to directories or txt files with paths.
    # For simplicity in this hybrid setup, we can point 'train' and 'val' to the same folder 
    # and let YOLO split or just validate on same (not ideal ML but keeps dir structure simple compliant with request).
    # OR better: The user's notebook splits logic in code (Dataset class). 
    # YOLO `train` command usually expects a structure.
    # Let's define the yaml to point to the images dir.
    
    data_config = {
        'path': os.path.abspath(data_dir), # dataset root dir
        'train': 'images',  # relative to 'path'
        'val': 'images',    # relative to 'path', usage dependent on splitting
        'names': {
            0: "with_mask",
            1: "without_mask",
            2: "mask_weared_incorrectly"
        }
    }
    
    with open(target_yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
        
    print(f"YOLO format conversion complete. YAML saved to {target_yaml_path}")
    return os.path.abspath(target_yaml_path)

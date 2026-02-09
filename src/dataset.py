import os
import torch
import torch.utils.data
from PIL import Image
import xml.etree.ElementTree as ET


class FaceMaskDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        # Assume directory structure:
        # root/
        #   images/
        #   annotations/

        self.img_dir = os.path.join(root, "images")
        self.annot_dir = os.path.join(root, "annotations")

        # Robust matching of images and xmls
        if not os.path.exists(self.img_dir) or not os.path.exists(self.annot_dir):
            self.imgs = []
            self.xmls = []
        else:
            all_imgs = sorted(os.listdir(self.img_dir))
            all_xmls = sorted(os.listdir(self.annot_dir))

            # Map basename to filename
            img_map = {
                os.path.splitext(f)[0]: f
                for f in all_imgs
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            }
            xml_map = {
                os.path.splitext(f)[0]: f
                for f in all_xmls
                if f.lower().endswith(".xml")
            }

            # Intersection
            common_names = sorted(list(set(img_map.keys()) & set(xml_map.keys())))

            self.imgs = [img_map[n] for n in common_names]
            self.xmls = [xml_map[n] for n in common_names]

            if len(self.imgs) != len(self.xmls):
                print(
                    f"Warning: Root {root} has mismatch: Imgs={len(all_imgs)}, XMLs={len(all_xmls)}, Matched={len(self.imgs)}"
                )

        # Label mapping
        self.label_map = {
            "with_mask": 1,
            "without_mask": 2,
            "mask_weared_incorrect": 3,
        }

    def __getitem__(self, idx):
        # Load images and masks
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        xml_path = os.path.join(self.annot_dir, self.xmls[idx])

        img = Image.open(img_path).convert("RGB")

        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            name = obj.find("name").text.strip()
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            # Keep labels compatible with Faster R-CNN target format
            if name in self.label_map:
                labels.append(self.label_map[name])
                boxes.append([xmin, ymin, xmax, ymax])

        # Handle cases with no valid boxes
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            # Fix: torchvision transforms only take image
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

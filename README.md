# Face Mask Detection

This project works on the [Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) to detect whether a person is wearing a mask, not wearing one, or wearing it incorrectly.

## Project Structure

- `data/`: Place dataset here.
  - `images/`: Contains image files.
  - `annotations/`: Contains XML annotation files.
- `notebooks/`: Jupyter notebooks for training and exploration.
- `src/`: Source code modules.
- `models/`: Directory for saving trained models.
- `inference.py`: Script for running inference on new images.

## Setup

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection).
   - Extract it so that you have `images` and `annotations` folders inside `data/`.
   - Structure should look like:

     ```
     data/
     ├── annotations/
     │   ├── maksssksksss0.xml
     │   └── ...
     └── images/
         ├── maksssksksss0.png
         └── ...
     ```

## Usage

### Training

Open `notebooks/Face_Mask_Detection.ipynb` and run all cells.

### Inference

Run the inference script on an image or folder:

```bash
python inference.py --input path/to/image.png
```

Or specify a model path:

```bash
python inference.py --input path/to/image.png --model models/best_model.pth
```

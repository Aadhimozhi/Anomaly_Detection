# Anomaly Localization on MVTec-AD

This project implements an anomaly detection and localization system on the **MVTec Anomaly Detection (MVTec-AD)** dataset using a **Convolutional Autoencoder (CAE)**. The model is trained unsupervised using only normal samples, and it localizes anomalies via reconstruction error.

## Project Structure

.
├── main.py                 # Full pipeline script (train, test, visualize)
├── mvtec_dataset.py        # Dataset loader (included in main.py currently)
├── model.py                # Convolutional autoencoder definition (included)
├── README.md               # Project documentation
└── MVTecDataset/
    └── Dataset/
        └── cable/
            ├── train/good/...
            ├── test/good/...
            ├── test/broken_wire/...
            └── ground_truth/broken_wire/...

## Model Architecture

**Convolutional Autoencoder:**

- **Encoder:**
  - Conv2D(3→64), Conv2D(64→128), Conv2D(128→256)
- **Decoder:**
  - ConvTranspose2D(256→128), ConvTranspose2D(128→64), ConvTranspose2D(64→3)
- Activation: ReLU (except last → Sigmoid)
- Loss: MSE (Mean Squared Error)

## Dependencies

Install the required packages:

pip install torch torchvision scikit-image matplotlib scikit-learn pillow

## Dataset

Download the **MVTec-AD dataset** from Kaggle:

- Link: https://www.kaggle.com/datasets/ipythonx/mvtec-ad
- Extract it and structure it as:

MVTecDataset/Dataset/<category>/...

Example for `cable` category:

MVTecDataset/Dataset/cable/train/good/
MVTecDataset/Dataset/cable/test/broken_wire/
MVTecDataset/Dataset/cable/test/good/
MVTecDataset/Dataset/cable/ground_truth/broken_wire/

## Running the Pipeline

Run the full training and evaluation pipeline:

python main.py

By default, it trains on the `cable` category.

## Evaluation Metrics

- Pixel-level AUROC: Measures per-pixel anomaly detection accuracy
- Image-level AUROC: Measures overall detection score per image

## Visualizations

The pipeline visualizes the following for 5 test images:

- Original image
- Anomaly heatmap (reconstruction error)
- Binary anomaly mask (after thresholding + post-processing)
- Ground truth mask

## Hyperparameters

| Parameter           | Value     |
|---------------------|-----------|
| Image size          | 256x256   |
| Batch size          | 16 (train), 1 (test) |
| Epochs              | 20        |
| Learning rate       | 1e-4      |
| Loss                | MSE       |
| Thresholding        | Otsu's method |
| Smoothing           | Gaussian filter (σ=2) |
| Morphology          | Binary closing, small object removal (min_size=100) |

## Sample Results

| Metric         | Value (Example) |
|----------------|-----------------|
| Pixel AUROC    | 0.92            |
| Image AUROC    | 0.95            |

## Notes

- Only normal images are used during training.
- The system uses unsupervised learning.
- You can change the category in `run_pipeline()` (e.g., 'hazelnut', 'bottle').

## Author

- Name: Aadhimozhi M
- Organization: Textro AI
- Project Duration: 16/09/2025 – 22/09/2025

## License

This project is for educational and research purposes. Attribution required for public sharing.

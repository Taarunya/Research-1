FusionNet: Deep Learning Based Multi-Sensor Satellite Data Fusion for Land Cover Mapping

Authors: Taarunya Aggarwal · Pranay Mahajan · Bhawani Singh Rathore
Department of Computer Science, Manipal University Jaipur, Rajasthan, India

📄 Abstract
Accurate land cover mapping is essential for environmental monitoring, urban planning, and sustainable resource management. This project proposes FusionNet, a multi-sensor deep learning framework that fuses RGB satellite imagery with pseudo-NDVI vegetation indices to classify land cover using the EuroSAT dataset. The CNN-based architecture learns spatial and spectral features simultaneously, achieving competitive classification accuracy across 10 land cover classes.

🗂️ Dataset
EuroSAT — Sentinel-2 satellite imagery benchmark dataset
PropertyDetailTotal Images27,000Image Size64 × 64 pixelsClasses10 land cover typesSplit70% train / 15% val / 15% test
Land Cover Classes:
Annual Crop · Forest · Herbaceous Vegetation · Highway · Industrial · Pasture · Permanent Crop · Residential · River · Sea/Lake

🔬 Methodology
1. Data Preprocessing

Normalization and resizing of Sentinel-2 imagery
Pseudo-NDVI computed from red and green bands (approximating NIR):

NDVI = (NIR - Red) / (NIR + Red)
2. Multi-Sensor Data Fusion
RGB bands and pseudo-NDVI channels are fused into a multi-channel input tensor, allowing the model to learn both spatial texture and spectral features simultaneously.
3. FusionNet CNN Architecture
LayerPurposeConvolutional layersHierarchical spatial feature extractionMax-pooling layersDimensionality reductionFully connected layersHigh-level reasoningSoftmax outputClass probability distribution
4. Training

Optimizer: Adam
Loss Function: Categorical Cross-Entropy
Regularization: Early stopping to prevent overfitting


📊 Results
ModelOverall AccuracyBaseline RGB CNN~86%FusionNet (RGB + pseudo-NDVI)~80%

Best classified: Forest, Residential, River, Sea/Lake
Most confused: Herbaceous Vegetation, Pasture, Permanent Crop (visually similar)
FusionNet showed stable convergence and competitive class-wise performance
Results highlight the need for true NIR multispectral bands to fully unlock multi-sensor fusion potential


⚙️ Setup & Usage
1. Install dependencies
bashpip3 install tensorflow numpy matplotlib scikit-learn opencv-python
2. Download EuroSAT dataset
bash# EuroSAT RGB dataset
wget https://madm.dfki.de/files/sentinel/EuroSAT.zip
unzip EuroSAT.zip -d data/
3. Run the pipeline
bashpython3 src/preprocess.py      # Preprocessing + NDVI computation
python3 src/train_fusionnet.py # Train FusionNet model
python3 src/evaluate.py        # Evaluate and generate plots

📁 Project Structure
fusionnet/
├── data/
│   └── EuroSAT/               # EuroSAT satellite images by class
├── src/
│   ├── preprocess.py          # Image normalization + pseudo-NDVI
│   ├── fusion.py              # Multi-channel tensor construction
│   ├── model.py               # FusionNet CNN architecture
│   ├── train_fusionnet.py     # Training pipeline
│   └── evaluate.py            # Metrics + confusion matrix + plots
├── results/                   # Output figures
└── main.py                    # Entry point

📚 Citation
If you use this work, please cite:
Aggarwal, T., Mahajan, P., & Rathore, B.S. (2025). FusionNet: Deep Learning Based 
Multi-Sensor Satellite Data Fusion for Land Cover Mapping. Department of Computer 
Science, Manipal University Jaipur.

📬 Contact
Taarunya Aggarwal — taru.agg05@gmail.com   Pranay Mahajan — pranaymahajan3106@gmail.com 

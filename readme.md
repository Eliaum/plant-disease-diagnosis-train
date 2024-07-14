# Plants diseases diagnosis based on MobileNet and Inception Architectures

## Description

This project uses MobileNet and Inception architectures to develop a deep learning model for diagnosing plant diseases from images. The model is trained on the PlantVillage dataset and can be exported to TensorFlow Lite for deployment on mobile devices.

## Installation

1. Install Python 3.x.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
Download the PlantVillage dataset and place it in the data/plantvillage_dataset folder.
Usage
Data Preparation:
Run the script src/data_preparation.py to prepare the data.
Model Building and Training:
Run the script src/model_building.py to build the models.
Run the script src/model_training.py to train the models with Adam and fine-tune them with RMSprop.
Model Evaluation:
Run the script src/model_evaluation.py to evaluate the model's performance.
Model Export:
Run the script src/model_export.py to export the models to TensorFlow Lite.
Notes
This project was initially developed on Kaggle.
To run the code locally, you'll need to install the required libraries and download the PlantVillage dataset.
The code may need adjustments to run on a different machine.
Contributors
Thierry AZEUFACK 
License
This project is licensed under the University Of Yaounde I - see the Thierry_Azeufack for details.
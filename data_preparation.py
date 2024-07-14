import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

# Chemin vers le jeu de données sur Kaggle
data_dir = '/kaggle/input/plantvillage-dataset/color'

# Création d'un DataFrame pour stocker les chemins d'accès aux images et leurs étiquettes
image_paths = []  # stockes les chemins d'accès aux images et leurs étiquettes correspondantes dans des listes
labels = []
for class_folder in os.listdir(data_dir):  # parcours chaque dossier de classe dans le répertoire des données
    class_path = os.path.join(data_dir, class_folder)
    image_files = os.listdir(class_path)
    for image_file in image_files:
        image_path = os.path.join(class_path, image_file)
        image_paths.append(image_path)
        labels.append(class_folder)

df = pd.DataFrame({'image_path': image_paths, 'label': labels})  # crée un DataFrame Pandas pour stocker les données

# Détection et affichage des doublons
duplicates = df[df['image_path'].duplicated(keep=False)]  # identifie les lignes du DataFrame qui ont le même chemin d'accès à l'image, en incluant les doublons
print(f"Nombre de doublons trouvés : {len(duplicates)}")  # affiche le nombre de doublons trouvés
print(f"Doublons : \n{duplicates}")  # affiche les lignes du DataFrame qui contiennent les doublons

# Suppression des doublons
df.drop_duplicates(subset=['image_path'], inplace=True)  # supprime les doublons du DataFrame en place
print(f"Doublons supprimés. Nombre total d'images : {len(df)}")  # affiche un message de confirmation et le nombre total d'images après la suppression des doublons

# Gestion du déséquilibre des classes par suréchantillonnage
max_size = df['label'].value_counts().max()  # détermine le nombre maximum d'images pour une classe
lst = [df]
for class_index, group in df.groupby('label'):  # parcours chaque classe dans le DataFrame
    lst.append(group.sample(max_size - len(group), replace=True))  # échantillonnage aléatoire des images de chaque classe pour atteindre le nombre maximum d'images (max_size)
df_balanced = pd.concat(lst)  # concaténation des DataFrames pour créer un DataFrame équilibré

# Affichage d'informations sur le DataFrame équilibré
print(f"Nombre total d'images après l'équilibrage : {len(df_balanced)}")  # affiche le nombre total d'images après l'équilibrage des classes
print(f"Nombre de classes : {df_balanced['label'].nunique()}")  # affiche le nombre de classes
print(df_balanced['label'].value_counts())  # affiche le nombre d'images par classe

# Répertoire pour les données prétraitées
processed_data_dir = '/kaggle/working/PlantVillage-processed'
os.makedirs(processed_data_dir, exist_ok=True)

# Normalisation des images (mise à l'échelle des pixels entre 0 et 1)
datagen = ImageDataGenerator(rescale=1./255)

# Augmentation des données (uniquement pour l'entraînement)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,  # Décalage horizontal
    height_shift_range=0.1,  # Décalage vertical
    zoom_range=0.1,
    fill_mode='nearest'
)

# Répartition des données (80% train, 10% val, 10% test)
train_df, temp_df = train_test_split(df_balanced, test_size=0.2, random_state=42, stratify=df_balanced['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

# Fonction pour prétraiter les images pour InceptionV3
def preprocess_image_inception(image_path, target_size=(299, 299)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img)
    if img.shape[-1] == 4:  # Suppression du canal alpha si présent
        img = img[:, :, :3]
    return img

# Fonction pour prétraiter les images pour MobileNetV2
def preprocess_image_mobilenet(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img)
    if img.shape[-1] == 4:  # Suppression du canal alpha si présent
        img = img[:, :, :3]
    return img

# Prétraitement des données avec redimensionnement
for df_split, split_name in zip([train_df, val_df, test_df], ['train', 'val', 'test']):
    print(f"\nPréparation de l'ensemble {split_name}...")
    for _, row in df_split.iterrows():
        class_name = row['label']
        image_name = os.path.basename(row['image_path'])
        src_path = row['image_path']
        dst_path_inception = os.path.join(processed_data_dir, 'inception', split_name, class_name, image_name)
        dst_path_mobilenet = os.path.join(processed_data_dir, 'mobilenet', split_name, class_name, image_name)
        os.makedirs(os.path.dirname(dst_path_inception), exist_ok=True)
        os.makedirs(os.path.dirname(dst_path_mobilenet), exist_ok=True)
        
        # Prétraitement et sauvegarde des images pour InceptionV3
        processed_img_inception = preprocess_image_inception(src_path)
        Image.fromarray(processed_img_inception).save(dst_path_inception)
        
        # Prétraitement et sauvegarde des images pour MobileNetV2
        processed_img_mobilenet = preprocess_image_mobilenet(src_path)
        Image.fromarray(processed_img_mobilenet).save(dst_path_mobilenet)

# Vérification du nombre d'images dans chaque ensemble
for model_name in ['inception', 'mobilenet']:
    for split_name in ['train', 'val', 'test']:
        split_dir = os.path.join(processed_data_dir, model_name, split_name)
        num_images = sum(len(files) for _, _, files in os.walk(split_dir))
        print(f"Ensemble {split_name} pour {model_name} : {num_images} images")
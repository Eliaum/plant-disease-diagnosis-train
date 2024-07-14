import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

# Chemins d'accès aux modèles Adam et RMSprop
model_inception_adam_path = '/kaggle/working/model_inception_adam.h5'
model_mobilenet_adam_path = '/kaggle/working/model_mobilenet_adam.h5'
model_inception_rmsprop_path = '/kaggle/working/model_inception_rmsprop.h5'
model_mobilenet_rmsprop_path = '/kaggle/working/model_mobilenet_rmsprop.h5'

# Chemin d'accès aux données de test
test_data_dir = '/kaggle/working/PlantVillage-processed/'

# Paramètres
batch_size = 32
input_shape_inception = (299, 299, 3)
input_shape_mobilenet = (224, 224, 3)

# Charger les modèles (Adam et RMSprop)
model_inception_adam = load_model(model_inception_adam_path)
model_mobilenet_adam = load_model(model_mobilenet_adam_path)
model_inception_rmsprop = load_model(model_inception_rmsprop_path)
model_mobilenet_rmsprop = load_model(model_mobilenet_rmsprop_path)

# Générateurs de données de test
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator_inception = test_datagen.flow_from_directory(
    os.path.join(test_data_dir, 'inception', 'test'),
    target_size=input_shape_inception[:2],
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important pour l'évaluation
)

test_generator_mobilenet = test_datagen.flow_from_directory(
    os.path.join(test_data_dir, 'mobilenet', 'test'),
    target_size=input_shape_mobilenet[:2],
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important pour l'évaluation
)

# Fonction pour évaluer et afficher les résultats d'un modèle
def evaluate_model(model, generator, model_name):
    print(f"\nÉvaluation {model_name}:")
    results = model.evaluate(generator)
    test_loss = results[0]
    test_acc = results[1]
    print(f"Test accuracy {model_name}:", test_acc)

    test_predictions = model.predict(generator)
    test_pred_labels = np.argmax(test_predictions, axis=1)
    true_labels = generator.classes

    # Calcul de l'AUC multi-classe
    lb = LabelBinarizer()
    true_labels_bin = lb.fit_transform(true_labels)
    test_auc = roc_auc_score(true_labels_bin, test_predictions, multi_class='ovr')
    print(f"AUC {model_name}:", test_auc)

    print(f"\nRapport de classification {model_name}:")
    print(classification_report(true_labels, test_pred_labels, target_names=generator.class_indices.keys()))
    print(f"Matrice de confusion {model_name}:\n", confusion_matrix(true_labels, test_pred_labels))

# Évaluation des modèles
evaluate_model(model_inception_adam, test_generator_inception, "InceptionV3 (Adam)")
evaluate_model(model_mobilenet_adam, test_generator_mobilenet, "MobileNetV2 (Adam)")
evaluate_model(model_inception_rmsprop, test_generator_inception, "InceptionV3 (RMSprop)")
evaluate_model(model_mobilenet_rmsprop, test_generator_mobilenet, "MobileNetV2 (RMSprop)")
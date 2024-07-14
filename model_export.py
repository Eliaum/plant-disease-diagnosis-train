import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Chemins d'accès aux modèles
model_inception_adam_path = '/kaggle/working/model_inception_adam.h5'
model_mobilenet_adam_path = '/kaggle/working/model_mobilenet_adam.h5'
model_inception_rmsprop_path = '/kaggle/working/model_inception_rmsprop.h5'
model_mobilenet_rmsprop_path = '/kaggle/working/model_mobilenet_rmsprop.h5'

# Convertir et sauvegarder chaque modèle
def convert_and_save(model_path, output_filename):
    try:
        model = load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(output_filename, 'wb') as f:
            f.write(tflite_model)
        print(f"Modèle {output_filename} converti avec succès !")
    except Exception as e:
        print(f"Erreur lors de la conversion de {model_path} : {e}")

# Convertir les modèles
try:
    convert_and_save(model_inception_adam_path, 'model_inception_adam.tflite')
    convert_and_save(model_mobilenet_adam_path, 'model_mobilenet_adam.tflite')
    convert_and_save(model_inception_rmsprop_path, 'model_inception_rmsprop.tflite')
    convert_and_save(model_mobilenet_rmsprop_path, 'model_mobilenet_rmsprop.tflite')
    print("Tous les modèles convertis en TensorFlow Lite avec succès !")
except Exception as e:
    print(f"Erreur lors de la conversion des modèles : {e}")
from tensorflow.keras import regularizers
from tensorflow.keras.applications import InceptionV3, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# Paramètres
input_shape_inception = (299, 299, 3)  # InceptionV3
input_shape_mobilenet = (224, 224, 3)  # MobileNetV2
num_classes = 38

# Création des modèles
base_model_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape_inception)
base_model_mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape_mobilenet)

# Gel des couches de base
for layer in base_model_inception.layers:
    layer.trainable = False
for layer in base_model_mobilenet.layers:
    layer.trainable = False

# Ajout des couches de classification 
x_inception = base_model_inception.output
x_inception = GlobalAveragePooling2D()(x_inception)
x_inception = Dropout(0.5)(x_inception)
predictions_inception = Dense(num_classes, activation='softmax')(x_inception)  # Pas de Flatten ni de Dropout pour InceptionV3
model_inception = Model(inputs=base_model_inception.input, outputs=predictions_inception)

x_mobilenet = base_model_mobilenet.output
x_mobilenet = GlobalAveragePooling2D()(x_mobilenet)  # Pas de Flatten ni de Dropout pour MobileNetV2
x_mobilenet = Dropout(0.5)(x_mobilenet)
predictions_mobilenet = Dense(num_classes, activation='softmax')(x_mobilenet)
model_mobilenet = Model(inputs=base_model_mobilenet.input, outputs=predictions_mobilenet)

# Vérification de la structure des modèles
print("Structure du modèle InceptionV3 :")
model_inception.summary()
print("\nStructure du modèle MobileNetV2 :")
model_mobilenet.summary()

# Vérification des paramètres d'entrée
assert model_inception.input_shape[1:] == input_shape_inception, "La taille d'entrée d'InceptionV3 ne correspond pas à la valeur spécifiée."
assert model_mobilenet.input_shape[1:] == input_shape_mobilenet, "La taille d'entrée de MobileNetV2 ne correspond pas à la valeur spécifiée."
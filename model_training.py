import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
import os

# Paramètres
input_shape_inception = (299, 299, 3)  # InceptionV3
input_shape_mobilenet = (224, 224, 3)  # MobileNetV2
batch_size = 32
epochs_adam = 3  # Entraînement initial avec Adam sur 3 epochs
epochs_rmsprop = 5  # Fine-tuning avec RMSprop sur 5 epochs
num_classes = 38
learning_rate = 0.001

# Chargement des modèles pour le fine-tuning
loaded_model_inception = tf.keras.layers.TFSMLayer('/kaggle/working/model_inception_adam', call_endpoint='serving_default')
loaded_model_mobilenet = tf.keras.layers.TFSMLayer('/kaggle/working/model_mobilenet_adam', call_endpoint='serving_default')

# Fine-tuning du modèle InceptionV3
# Créer un nouveau modèle Keras pour le fine-tuning
inputs = tf.keras.Input(shape=(224, 224, 3))
x = loaded_model_inception(inputs)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
fine_tuned_model_inception = tf.keras.Model(inputs=inputs, outputs=outputs)

# Dégeler les couches supérieures pour le fine-tuning
for layer in fine_tuned_model_inception.layers[-2:]:
    layer.trainable = True

# Compiler le modèle fine-tuné
fine_tuned_model_inception.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall', 'f1_score'])

# Fine-tuning avec RMSprop
history_inception_rmsprop = fine_tuned_model_inception.fit_generator(
    train_generator_inception,
    steps_per_epoch=train_generator_inception.samples // batch_size,
    epochs=epochs_rmsprop,
    validation_data=val_generator_inception,
    validation_steps=val_generator_inception.samples // batch_size,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
)

# Fine-tuning du modèle MobileNetV2
# Créer un nouveau modèle Keras pour le fine-tuning
inputs = tf.keras.Input(shape=(224, 224, 3))
x = loaded_model_mobilenet(inputs)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
fine_tuned_model_mobilenet = tf.keras.Model(inputs=inputs, outputs=outputs)

# Dégeler les couches supérieures pour le fine-tuning
for layer in fine_tuned_model_mobilenet.layers[-30:]:
    layer.trainable = True

# Compiler le modèle fine-tuné
fine_tuned_model_mobilenet.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall', 'f1_score'])

# Fine-tuning avec RMSprop
history_mobilenet_rmsprop = fine_tuned_model_mobilenet.fit_generator(
    train_generator_mobilenet,
    steps_per_epoch=train_generator_mobilenet.samples // batch_size,
    epochs=epochs_rmsprop,
    validation_data=val_generator_mobilenet,
    validation_steps=val_generator_mobilenet.samples // batch_size,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
)

# Sauvegarde des modèles entraînés avec RMSprop
fine_tuned_model_inception.save('model_inception_rmsprop.h5')
fine_tuned_model_mobilenet.save('model_mobilenet_rmsprop.h5')
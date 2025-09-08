# dog_breed_classifier.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder
import joblib

# Paths
train_dir = "data/train"
val_dir = "data/val"
test_dir = "data/test"
model_path = "saved_model/dog_breed_classifier.h5"
label_path = "saved_model/label_encoder.pkl"

# Parameters
img_size = 300
batch_size = 32
num_epochs = 20

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='sparse'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='sparse'
)

# Number of classes
num_classes = len(train_gen.class_indices)

# Save class labels
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(list(train_gen.class_indices.keys()))
joblib.dump(label_encoder, label_path)

# Model building
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
preds = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
os.makedirs(os.path.dirname(model_path), exist_ok=True)

checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training
history = model.fit(
    train_gen,
    epochs=num_epochs,
    validation_data=val_gen,
    callbacks=[checkpoint, early_stop]
)

print("✅ Training complete. Model saved at:", model_path)
print("✅ Label encoder saved at:", label_path)

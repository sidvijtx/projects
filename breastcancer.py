# -*- coding: utf-8 -*-
"""breastcancer.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19Bnm9IextKhDaEINkMdwEGx9kVgyeCbC
"""

import kagglehub

# Download latest version
path = kagglehub.dataset_download("hayder17/breast-cancer-detection")

print("Path to dataset files:", path)



import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = path

train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, rotation_range=15)
valid_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)


train_generator = train_datagen.flow_from_directory(
    directory=f"{base_dir}/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

valid_generator = valid_datagen.flow_from_directory(
    directory=f"{base_dir}/valid",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    directory=f"{base_dir}/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)



from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers

base_model = VGG16(weights="imagenet", include_top= False, input_shape=(224, 224, 3))


for layer in base_model.layers[:-1] :
    layer.trainable = False

for layer in base_model.layers[-4:] :
    layer.trainable = True




model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.6),
    Dense(1, activation='sigmoid')
])
optimizer = Adam(learning_rate=0.00001)
model.compile(optimizer= optimizer, loss='binary_crossentropy', metrics=['accuracy'])


model.summary()









history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=40,
    steps_per_epoch=len(train_generator),
    validation_steps=len(valid_generator)
)

train_loss, train_accuracy = model.evaluate(train_generator)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

test_loss, test_accuracy = model.evaluate(valid_generator)
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
model.save("breastcancer.keras")








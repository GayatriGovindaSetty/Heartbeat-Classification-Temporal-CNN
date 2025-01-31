# Imports
import os
import glob
import numpy as np
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
import pandas as pd

# Audio
import librosa
import librosa.display

# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle, class_weight

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import (Input, Conv1D, BatchNormalization, Dropout,
                                     Add, LeakyReLU, Dense, GlobalAveragePooling1D)
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.utils import to_categorical
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Prepare Dataset
dataset = []
for folder in ["/content/drive/MyDrive/Stethoscope project/Deekshitha M/balanced dataset/set_a/**",
               "/content/drive/MyDrive/Stethoscope project/Deekshitha M/balanced dataset/set_b/**"]:
    for filename in glob.iglob(folder):
        if os.path.exists(filename):
            label = os.path.basename(filename).split("_")[0]
            duration = librosa.get_duration(filename=filename)
            if duration >= 3:  # Skip short audio
                slice_size = 3
                iterations = int((duration - slice_size) / (slice_size - 1)) + 1
                initial_offset = (duration - ((iterations * (slice_size - 1)) + 1)) / 2
                if label not in ["Aunlabelledtest", "Bunlabelledtest", "artifact"]:
                    for i in range(iterations):
                        offset = initial_offset + i * (slice_size - 1)
                        label_category = "normal" if label == "normal" else "abnormal"
                        dataset.append({
                            "filename": filename,
                            "label": label_category,
                            "offset": offset
                        })

dataset = pd.DataFrame(dataset)
dataset = shuffle(dataset, random_state=42)

# Visualize Dataset
plt.figure(figsize=(4, 6))
dataset.label.value_counts().plot(kind='bar', title="Dataset Distribution")
plt.show()

# Split Dataset
train, test = train_test_split(dataset, test_size=0.2, random_state=42)

# Feature Extraction with Delta
def extract_features(audio_path, offset):
    y, sr = librosa.load(audio_path, offset=offset, duration=3)
    y = librosa.effects.preemphasis(y)  # Apply pre-emphasis
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
    delta = librosa.feature.delta(mfccs)
    delta_delta = librosa.feature.delta(mfccs, order=2)
    return np.vstack([mfccs, delta, delta_delta])

x_train, x_test = [], []
for idx in tqdm(range(len(train))):
    x_train.append(extract_features(train.filename.iloc[idx], train.offset.iloc[idx]))
for idx in tqdm(range(len(test))):
    x_test.append(extract_features(test.filename.iloc[idx], test.offset.iloc[idx]))

x_train, x_test = np.asarray(x_train), np.asarray(x_test)

# Encode Labels
encoder = LabelEncoder()
encoder.fit(train.label)

y_train = encoder.transform(train.label)
y_test = encoder.transform(test.label)

class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Reshape Data
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define TCN Block
def tcn_block(input_tensor, num_filters, kernel_size, dilation_rate, dropout_rate=0.3):
    x = Conv1D(filters=num_filters, kernel_size=kernel_size, padding='causal',
               dilation_rate=dilation_rate)(input_tensor)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(filters=num_filters, kernel_size=kernel_size, padding='causal',
               dilation_rate=dilation_rate)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    res = Conv1D(filters=num_filters, kernel_size=1, padding='same')(input_tensor)
    x = Add()([x, res])
    x = LeakyReLU(alpha=0.01)(x)
    return x

# Define Model
input_shape = (x_train.shape[1], x_train.shape[2])
inputs = Input(shape=input_shape)

x = tcn_block(inputs, num_filters=64, kernel_size=3, dilation_rate=1)
x = tcn_block(x, num_filters=128, kernel_size=3, dilation_rate=2)
x = tcn_block(x, num_filters=256, kernel_size=3, dilation_rate=4)

x = GlobalAveragePooling1D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(len(encoder.classes_), activation='softmax')(x)

model = Model(inputs, outputs)
model.summary()

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Train Model
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=300,
                    validation_data=(x_test, y_test),
                    class_weight=class_weights_dict,
                    callbacks=[early_stopping, lr_scheduler],
                    shuffle=True)

# Plot Loss and Accuracy
plt.figure(figsize=[14, 10])
plt.subplot(211)
plt.plot(history.history['loss'], '#d62728', linewidth=3.0)
plt.plot(history.history['val_loss'], '#1f77b4', linewidth=3.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)

plt.subplot(212)
plt.plot(history.history['accuracy'], '#d62728', linewidth=3.0)
plt.plot(history.history['val_accuracy'], '#1f77b4', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)

# Evaluate Model
scores = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {scores[0]}")
print(f"Test Accuracy: {scores[1]}")

# Confusion Matrix
predictions = model.predict(x_test, verbose=1)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(predictions, axis=1)
conf_matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Save Model
model.save("heartbeat_classifier_tcn_improved.h5")
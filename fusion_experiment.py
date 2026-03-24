import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = "2750"
IMG_SIZE = 64

classes = [c for c in os.listdir(DATASET_PATH)
           if os.path.isdir(os.path.join(DATASET_PATH, c))]
classes.sort()

X_rgb, X_fusion, y = [], [], []

print("Loading dataset...")

for idx, cls in enumerate(classes):
    folder = os.path.join(DATASET_PATH, cls)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        # RGB
        X_rgb.append(img)

        # Pseudo NDVI
        red = img[:,:,2]
        green = img[:,:,1]
        ndvi = (green - red) / (green + red + 1e-6)
        ndvi = np.expand_dims(ndvi, axis=2)

        fusion_img = np.concatenate((img, ndvi), axis=2)
        X_fusion.append(fusion_img)

        y.append(idx)

X_rgb = np.array(X_rgb)
X_fusion = np.array(X_fusion)
y = to_categorical(np.array(y))

print("Dataset loaded:", X_rgb.shape)

# train test split
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_rgb, y, test_size=0.2)
Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_fusion, y, test_size=0.2)

# ---------------- DATA AUGMENTATION (NEW IMPROVEMENT) ----------------
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# ---------------- MODEL BUILDER (IMPROVED CNN) ----------------
def build_model(channels):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(64,64,channels)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10,activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------------- TRAIN RGB BASELINE ----------------
print("\nTraining RGB CNN...")
rgb_model = build_model(3)

hist_rgb = rgb_model.fit(
    datagen.flow(Xr_train, yr_train, batch_size=32),
    epochs=12,
    validation_data=(Xr_test, yr_test)
)

rgb_acc = rgb_model.evaluate(Xr_test, yr_test)[1]
print("RGB Accuracy:", rgb_acc)

# ---------------- TRAIN FUSIONNET ----------------
print("\nTraining FusionNet...")
fusion_model = build_model(4)

hist_fusion = fusion_model.fit(
    datagen.flow(Xf_train, yf_train, batch_size=32),
    epochs=12,
    validation_data=(Xf_test, yf_test)
)

fusion_acc = fusion_model.evaluate(Xf_test, yf_test)[1]
print("FusionNet Accuracy:", fusion_acc)

# ---------------- GRAPH 1: Accuracy Comparison ----------------
plt.figure()
plt.bar(["RGB CNN","FusionNet"], [rgb_acc, fusion_acc])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.savefig("model_comparison.png")
plt.close()

# ---------------- GRAPH 2: Training Curves ----------------
plt.figure()
plt.plot(hist_rgb.history['accuracy'], label='RGB Train')
plt.plot(hist_rgb.history['val_accuracy'], label='RGB Val')
plt.plot(hist_fusion.history['accuracy'], label='Fusion Train')
plt.plot(hist_fusion.history['val_accuracy'], label='Fusion Val')
plt.title("Training Accuracy Curves")
plt.legend()
plt.savefig("training_curves.png")
plt.close()

# ---------------- GRAPH 3: CONFUSION MATRIX ----------------
pred = fusion_model.predict(Xf_test)
pred_classes = np.argmax(pred, axis=1)
true_classes = np.argmax(yf_test, axis=1)

cm = confusion_matrix(true_classes, pred_classes)
disp = ConfusionMatrixDisplay(cm, display_labels=classes)
disp.plot(xticks_rotation=45)
plt.title("FusionNet Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# ---------------- GRAPH 4: SAMPLE PREDICTIONS ----------------
plt.figure(figsize=(10,10))
for i in range(9):
    idx = np.random.randint(0, len(Xf_test))
    img = Xf_test[idx][:,:,:3]
    pred = np.argmax(fusion_model.predict(Xf_test[idx:idx+1]), axis=1)[0]
    true = np.argmax(yf_test[idx])
    
    plt.subplot(3,3,i+1)
    plt.imshow(img)
    plt.title(f"P:{classes[pred]}\nT:{classes[true]}")
    plt.axis("off")

plt.savefig("sample_predictions.png")
plt.close()

print("\nALL FIGURES GENERATED ✅")

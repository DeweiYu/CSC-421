import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
from PIL import Image
import os
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Step 1: Define shapes and colors
shapes = ['square', 'circle', 'triangle', 'rhombus']
colors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ffa500', '#000000']

# Create directory for dataset
os.makedirs("synthetic_dataset", exist_ok=True)

# Generate a single random shape image
def generate_random_shape():
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    shape_type = np.random.choice(shapes)
    color = np.random.choice(colors)
    size = np.random.uniform(0.1, 0.4)
    x_pos = np.random.uniform(0 + size / 2, 1 - size / 2)
    y_pos = np.random.uniform(0 + size / 2, 1 - size / 2)

    if shape_type == 'square':
        shape = Rectangle((x_pos - size / 2, y_pos - size / 2), size, size, facecolor=color)
    elif shape_type == 'circle':
        shape = Circle((x_pos, y_pos), size / 2, facecolor=color)
    elif shape_type == 'triangle':
        shape = Polygon([(x_pos, y_pos + size / 2), 
                         (x_pos - size / 2, y_pos - size / 2), 
                         (x_pos + size / 2, y_pos - size / 2)], facecolor=color)
    elif shape_type == 'rhombus':
        shape = Polygon([(x_pos, y_pos + size / 2),
                         (x_pos + size / 2, y_pos),
                         (x_pos, y_pos - size / 2),
                         (x_pos - size / 2, y_pos)], facecolor=color)

    ax.add_patch(shape)
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img_resized = Image.fromarray(img).resize((64, 64))

    plt.close(fig)
    return np.array(img_resized), shape_type, color

# Generate dataset
dataset, labels_shape, labels_color = [], [], []
num_samples = 6000

for _ in range(num_samples):
    img, shape_label, color_label = generate_random_shape()
    dataset.append(img)
    labels_shape.append(shape_label)
    labels_color.append(color_label)

np.save('synthetic_dataset/images.npy', np.array(dataset))
np.save('synthetic_dataset/labels_shape.npy', np.array(labels_shape))
np.save('synthetic_dataset/labels_color.npy', np.array(labels_color))

# Visualize samples
images = np.array(dataset)
fig, axs = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axs.ravel()):
    idx = np.random.randint(0, len(images))
    ax.imshow(images[idx])
    ax.set_title(f"{labels_shape[idx]}, {labels_color[idx]}")
    ax.axis('off')
plt.show()

# Extract HOG features
hog_features = []
for img in images:
    feature = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=True)
    hog_features.append(feature)
hog_features = np.array(hog_features)
np.save('synthetic_dataset/hog_features.npy', hog_features)

# Encode labels
le_shape, le_color = LabelEncoder(), LabelEncoder()
y_shape = le_shape.fit_transform(labels_shape)
y_color = le_color.fit_transform(labels_color)

# Split data
X_train, X_test, y_shape_train, y_shape_test = train_test_split(hog_features, y_shape, test_size=0.2, random_state=42)
_, _, y_color_train, y_color_test = train_test_split(hog_features, y_color, test_size=0.2, random_state=42)

# Train SVM for shape classification
svm_shape = SVC(kernel='linear')
svm_shape.fit(X_train, y_shape_train)
y_shape_pred = svm_shape.predict(X_test)

# Train SVM for color classification
svm_color = SVC(kernel='linear')
svm_color.fit(X_train, y_color_train)
y_color_pred = svm_color.predict(X_test)

# Report SVM results
print("Shape Classification Report (SVM):")
print(classification_report(y_shape_test, y_shape_pred, target_names=le_shape.classes_))
print("Color Classification Report (SVM):")
print(classification_report(y_color_test, y_color_pred, target_names=le_color.classes_))

# Deep Learning Classifier
X_dl = images / 255.0  # Normalize
y_shape_dl = to_categorical(y_shape)
y_color_dl = to_categorical(y_color)

# Split data
X_train, X_test, y_shape_train, y_shape_test = train_test_split(X_dl, y_shape_dl, test_size=0.2, random_state=42)
X_train, X_test, y_color_train, y_color_test = train_test_split(X_dl, y_color_dl, test_size=0.2, random_state=42)

# Build and train DNN for shape classification
model_shape = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(le_shape.classes_), activation='softmax')
])
model_shape.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_shape.fit(X_train, y_shape_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate DNN for shape classification
loss, acc = model_shape.evaluate(X_test, y_shape_test)
print(f"Shape Classification Accuracy (DNN): {acc}")

# Build and train DNN for color classification
model_color = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(le_color.classes_), activation='softmax')
])
model_color.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_color.fit(X_train, y_color_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate DNN for color classification
loss, acc = model_color.evaluate(X_test, y_color_test)
print(f"Color Classification Accuracy (DNN): {acc}")

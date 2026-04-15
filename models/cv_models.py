"""Computer Vision module: image preprocessing and simple classification."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import io
import base64
import os
import json
from models.database import execute_db


def generate_sample_image(image_type='building', size=(200, 200)):
    """Generate a simple sample image for CV demonstrations."""
    img = Image.new('RGB', size, color=(135, 206, 235))  # sky blue
    draw = ImageDraw.Draw(img)

    if image_type == 'building':
        # Draw a simplified HDB building
        draw.rectangle([40, 60, 160, 180], fill=(180, 180, 180), outline=(100, 100, 100))
        # Windows
        for row in range(4):
            for col in range(3):
                x = 55 + col * 35
                y = 70 + row * 28
                draw.rectangle([x, y, x + 15, y + 15], fill=(200, 230, 255), outline=(100, 100, 100))
        # Door
        draw.rectangle([85, 150, 115, 180], fill=(139, 90, 43))
        # Ground
        draw.rectangle([0, 180, 200, 200], fill=(34, 139, 34))

    elif image_type == 'park':
        # Green park scene
        draw.rectangle([0, 120, 200, 200], fill=(34, 139, 34))
        # Trees
        for x_pos in [40, 100, 160]:
            draw.rectangle([x_pos - 5, 80, x_pos + 5, 130], fill=(101, 67, 33))
            draw.ellipse([x_pos - 25, 50, x_pos + 25, 100], fill=(0, 128, 0))
        # Path
        draw.line([(0, 160), (200, 140)], fill=(210, 180, 140), width=8)

    elif image_type == 'mrt':
        # MRT station
        draw.rectangle([20, 100, 180, 160], fill=(200, 200, 200), outline=(100, 100, 100))
        draw.rectangle([20, 80, 180, 100], fill=(200, 50, 50))
        # Platform
        draw.rectangle([20, 160, 180, 180], fill=(150, 150, 150))
        # Track lines
        draw.line([(0, 175), (200, 175)], fill=(80, 80, 80), width=3)
        draw.line([(0, 178), (200, 178)], fill=(80, 80, 80), width=3)

    return img


def image_to_base64(img):
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def preprocess_image(img, target_size=(64, 64)):
    """Demonstrate image preprocessing pipeline."""
    steps = {}

    # Step 1: Original
    steps['original'] = {
        'description': 'Original image',
        'size': img.size,
        'mode': img.mode,
        'image': image_to_base64(img)
    }

    # Step 2: Resize
    resized = img.resize(target_size, Image.Resampling.LANCZOS)
    steps['resized'] = {
        'description': f'Resized to {target_size}',
        'size': resized.size,
        'image': image_to_base64(resized)
    }

    # Step 3: Grayscale
    grayscale = resized.convert('L')
    steps['grayscale'] = {
        'description': 'Converted to grayscale',
        'size': grayscale.size,
        'mode': 'L',
        'image': image_to_base64(grayscale)
    }

    # Step 4: Normalize pixel values
    arr = np.array(grayscale, dtype=np.float32) / 255.0
    steps['normalized'] = {
        'description': 'Pixel values normalized to [0, 1]',
        'shape': arr.shape,
        'min': round(float(arr.min()), 4),
        'max': round(float(arr.max()), 4),
        'mean': round(float(arr.mean()), 4),
        'std': round(float(arr.std()), 4)
    }

    # Step 5: Edge detection
    edges = resized.filter(ImageFilter.FIND_EDGES)
    steps['edges'] = {
        'description': 'Edge detection applied',
        'image': image_to_base64(edges)
    }

    # Step 6: Blur
    blurred = resized.filter(ImageFilter.GaussianBlur(radius=2))
    steps['blurred'] = {
        'description': 'Gaussian blur applied (radius=2)',
        'image': image_to_base64(blurred)
    }

    return steps


def extract_color_histogram(img, bins=8):
    """Extract color histogram features from an image."""
    arr = np.array(img)
    histograms = {}

    if img.mode == 'RGB':
        for i, channel in enumerate(['Red', 'Green', 'Blue']):
            hist, bin_edges = np.histogram(arr[:, :, i], bins=bins, range=(0, 256))
            histograms[channel] = {
                'counts': hist.tolist(),
                'bins': [round(float(b), 1) for b in bin_edges.tolist()],
                'mean': round(float(arr[:, :, i].mean()), 2),
                'std': round(float(arr[:, :, i].std()), 2)
            }
    elif img.mode == 'L':
        hist, bin_edges = np.histogram(arr, bins=bins, range=(0, 256))
        histograms['Gray'] = {
            'counts': hist.tolist(),
            'bins': [round(float(b), 1) for b in bin_edges.tolist()]
        }

    return histograms


def simple_image_classifier(img):
    """Simple rule-based classifier based on color features (educational demo)."""
    arr = np.array(img)

    if img.mode != 'RGB':
        img = img.convert('RGB')
        arr = np.array(img)

    avg_r = float(arr[:, :, 0].mean())
    avg_g = float(arr[:, :, 1].mean())
    avg_b = float(arr[:, :, 2].mean())

    green_ratio = avg_g / (avg_r + avg_g + avg_b + 1e-10)
    blue_ratio = avg_b / (avg_r + avg_g + avg_b + 1e-10)
    gray_std = float(np.std([avg_r, avg_g, avg_b]))

    # Simple rule-based classification
    if green_ratio > 0.38:
        prediction = 'park/nature'
        confidence = min(green_ratio * 2, 0.95)
    elif gray_std < 15 and avg_r > 150:
        prediction = 'building/infrastructure'
        confidence = 0.7
    elif blue_ratio > 0.38:
        prediction = 'sky/water'
        confidence = min(blue_ratio * 2, 0.9)
    else:
        prediction = 'urban/mixed'
        confidence = 0.5

    result = {
        'prediction': prediction,
        'confidence': round(confidence, 4),
        'color_features': {
            'avg_red': round(avg_r, 2),
            'avg_green': round(avg_g, 2),
            'avg_blue': round(avg_b, 2),
            'green_ratio': round(green_ratio, 4),
            'blue_ratio': round(blue_ratio, 4),
            'gray_std': round(gray_std, 2)
        },
        'note': 'This is a simplified rule-based classifier for educational purposes. '
                'In production, use CNNs (e.g., ResNet, VGG) trained on labeled data.'
    }

    execute_db(
        "INSERT INTO model_metrics (model_name, model_type, metric_name, metric_value, notes) VALUES (?,?,?,?,?)",
        ('SimpleImageClassifier', 'cv', 'confidence', confidence, prediction)
    )

    return result


def neural_network_pseudocode():
    """Return pseudo-implementation of a CNN for educational purposes."""
    return {
        'title': 'Convolutional Neural Network for Singapore Scene Classification',
        'framework': 'TensorFlow/Keras (pseudo-code)',
        'architecture': [
            {'layer': 'Input', 'shape': '(64, 64, 3)', 'params': 0},
            {'layer': 'Conv2D', 'filters': 32, 'kernel': '3x3', 'activation': 'relu'},
            {'layer': 'MaxPooling2D', 'pool_size': '2x2'},
            {'layer': 'Conv2D', 'filters': 64, 'kernel': '3x3', 'activation': 'relu'},
            {'layer': 'MaxPooling2D', 'pool_size': '2x2'},
            {'layer': 'Conv2D', 'filters': 128, 'kernel': '3x3', 'activation': 'relu'},
            {'layer': 'GlobalAveragePooling2D'},
            {'layer': 'Dense', 'units': 64, 'activation': 'relu'},
            {'layer': 'Dropout', 'rate': 0.5},
            {'layer': 'Dense', 'units': 4, 'activation': 'softmax'}
        ],
        'code': """
# Pseudo-code: CNN for Singapore scene classification
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input(shape=(64, 64, 3)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4, activation='softmax')  # building, park, mrt, urban
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training
# model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=32)

# Notes:
# - Collect labeled images of Singapore scenes (HDB blocks, parks, MRT stations)
# - Augment data with rotation, flipping, brightness adjustments
# - Use transfer learning (ResNet50, MobileNetV2) for better accuracy
# - Export model with model.save() for Flask deployment
""",
        'classes': ['building', 'park', 'mrt_station', 'urban_scene'],
        'training_tips': [
            'Use data augmentation to increase training set size',
            'Transfer learning from ImageNet pre-trained models works well for small datasets',
            'Start with MobileNetV2 for efficiency on local machines',
            'Use early stopping and learning rate scheduling',
            'Minimum ~100 images per class for acceptable accuracy'
        ]
    }

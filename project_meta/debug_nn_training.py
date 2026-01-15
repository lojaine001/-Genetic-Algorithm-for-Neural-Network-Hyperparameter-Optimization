"""
DEBUG SCRIPT - Test Single Neural Network Training
Run this to verify that neural networks CAN actually train
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers

print("="*60)
print("DEBUG: Testing Single Neural Network Training")
print("="*60)

# Load small dataset
print("\n1. Loading Fashion-MNIST (small subset)...")
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Use small subset
X_train = X_train[:1000].astype('float32') / 255.0
y_train = y_train[:1000]
X_test = X_test[:200].astype('float32') / 255.0
y_test = y_test[:200]

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(f"✓ Loaded {len(X_train)} training, {len(X_test)} test samples")

# Build a simple model
print("\n2. Building neural network...")
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model built")
print(f"  Total parameters: {model.count_params():,}")

# Train the model
print("\n3. Training neural network...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_test, y_test),
    verbose=1
)

# Get final results
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Final Training Accuracy:   {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")

if final_val_acc > 0.5:
    print("\n✅ SUCCESS: Neural network IS training properly!")
    print("   The model learned something (>50% accuracy)")
    print("\n   This means the issue in your main code is likely:")
    print("   1. Too few epochs (try 10 instead of 5)")
    print("   2. Dataset too small")
    print("   3. Fitness function not calculating correctly")
else:
    print("\n⚠️  WARNING: Model didn't learn well")
    print("   Accuracy is too low. Check:")
    print("   1. Data preprocessing")
    print("   2. Model architecture")
    print("   3. Training parameters")

print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)

# Show training progression
print("\nTraining Progression:")
for epoch, (train_acc, val_acc) in enumerate(zip(
    history.history['accuracy'], 
    history.history['val_accuracy']
), 1):
    print(f"  Epoch {epoch:2d}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

print("\nIf you see accuracy INCREASING over epochs, neural networks are working!")
print("If accuracy stays flat at ~0.10 (random guessing), there's a problem.")

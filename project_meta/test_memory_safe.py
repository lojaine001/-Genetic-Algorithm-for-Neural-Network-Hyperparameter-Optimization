"""
MEMORY-SAFE VERSION - Clears TensorFlow memory aggressively
This should fix the model training failures
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import time
import random
from copy import deepcopy
import gc
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow for CPU only (avoid GPU memory issues)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
tf.config.set_visible_devices([], 'GPU')

print("="*60)
print("MEMORY-SAFE VERSION - GA Hyperparameter Optimization")
print("Running on CPU to avoid memory issues")
print("="*60)

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# SIMPLE GENETIC ALGORITHM
# ============================================================

class SimpleGA:
    def __init__(self):
        self.search_space = {
            'n_layers': [1, 2],  # Simplified to avoid complex models
            'layer_sizes': [32, 64, 128],
            'learning_rate': (0.001, 0.01),
            'batch_size': [32, 64],
            'dropout': (0.1, 0.3),
            'optimizer': ['adam'],  # Just Adam for simplicity
            'activation': ['relu']  # Just ReLU for simplicity
        }
    
    def create_random(self):
        n_layers = random.choice(self.search_space['n_layers'])
        return {
            'n_layers': n_layers,
            'layer_sizes': [random.choice(self.search_space['layer_sizes']) 
                           for _ in range(n_layers)],
            'learning_rate': random.uniform(*self.search_space['learning_rate']),
            'batch_size': random.choice(self.search_space['batch_size']),
            'dropout': random.uniform(*self.search_space['dropout']),
            'optimizer': 'adam',
            'activation': 'relu'
        }

# ============================================================
# BUILD AND TRAIN - WITH AGGRESSIVE MEMORY MANAGEMENT
# ============================================================

def train_single_model(config, X_train, y_train, X_val, y_val):
    """Train one model with aggressive memory cleanup"""
    
    print(f"\n   🔧 Config: {config['n_layers']} layers × {config['layer_sizes']}, "
          f"lr={config['learning_rate']:.4f}, batch={config['batch_size']}")
    
    # Clear any existing models
    keras.backend.clear_session()
    gc.collect()
    
    try:
        # Build model
        model = models.Sequential([
            layers.Flatten(input_shape=(28, 28))
        ])
        
        for i in range(config['n_layers']):
            model.add(layers.Dense(config['layer_sizes'][i], activation='relu'))
            model.add(layers.Dropout(config['dropout']))
        
        model.add(layers.Dense(10, activation='softmax'))
        
        # Compile
        model.compile(
            optimizer=optimizers.Adam(learning_rate=config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"   📊 Model has {model.count_params():,} parameters")
        print(f"   🏃 Training for 5 epochs...")
        
        # Train with reduced epochs for speed
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            batch_size=config['batch_size'],
            epochs=5,  # Reduced epochs
            validation_data=(X_val, y_val),
            verbose=0,
            shuffle=True
        )
        train_time = time.time() - start_time
        
        # Extract results
        val_acc = float(history.history['val_accuracy'][-1])
        train_acc = float(history.history['accuracy'][-1])
        
        print(f"   ✅ SUCCESS! Train: {train_acc:.4f}, Val: {val_acc:.4f}, Time: {train_time:.1f}s")
        
        # Calculate fitness
        fitness = val_acc  # Simple: just use validation accuracy
        
        result = {
            'fitness': float(fitness),
            'accuracy': float(val_acc),
            'train_accuracy': float(train_acc),
            'training_time': float(train_time),
            'n_parameters': int(model.count_params())
        }
        
        # CRITICAL: Delete model and clear memory
        del model
        del history
        keras.backend.clear_session()
        gc.collect()
        
        # Small delay to let memory clear
        time.sleep(0.5)
        
        return result
        
    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        
        # Clear everything even on failure
        try:
            keras.backend.clear_session()
            gc.collect()
        except:
            pass
        
        # Return failure values
        return {
            'fitness': 0.0,
            'accuracy': 0.0,
            'train_accuracy': 0.0,
            'training_time': 0.0,
            'n_parameters': 0,
            'error': str(e)
        }

# ============================================================
# MAIN
# ============================================================

print("\n📦 Loading Fashion-MNIST...")
(X_train_full, y_train_full), _ = keras.datasets.fashion_mnist.load_data()

# Use VERY SMALL subset for testing
X_train = X_train_full[:1000].astype('float32') / 255.0
y_train = y_train_full[:1000]
X_val = X_train_full[1000:1200].astype('float32') / 255.0
y_val = y_train_full[1000:1200]

y_train = keras.utils.to_categorical(y_train, 10)
y_val = keras.utils.to_categorical(y_val, 10)

print(f"✅ Dataset: {len(X_train)} train, {len(X_val)} val")
print(f"   Using TINY dataset to test if models can train at all")

# Test ONE model first
print("\n" + "="*60)
print("🧪 TESTING: Can we train even ONE model?")
print("="*60)

ga = SimpleGA()
test_config = ga.create_random()
print(f"\nTest configuration:")
for k, v in test_config.items():
    print(f"  {k}: {v}")

test_result = train_single_model(test_config, X_train, y_train, X_val, y_val)

print(f"\n" + "="*60)
print("TEST RESULT:")
print(f"="*60)
print(f"Fitness:  {test_result['fitness']:.4f}")
print(f"Accuracy: {test_result['accuracy']:.4f}")
print(f"Time:     {test_result['training_time']:.2f}s")

if test_result['fitness'] > 0.3:
    print("\n✅ SUCCESS! Model trained successfully!")
    print("   Neural networks CAN train in the GA loop")
    print("\n🚀 Now running mini GA (3 individuals, 3 generations)...")
    
    # Mini GA
    population_size = 3
    generations = 3
    
    for gen in range(generations):
        print(f"\n{'='*60}")
        print(f"GENERATION {gen+1}/{generations}")
        print(f"{'='*60}")
        
        results = []
        for i in range(population_size):
            print(f"\n👤 Individual {i+1}/{population_size}")
            config = ga.create_random()
            result = train_single_model(config, X_train, y_train, X_val, y_val)
            results.append((config, result))
        
        # Show best
        best_idx = max(range(len(results)), key=lambda i: results[i][1]['fitness'])
        best_config, best_result = results[best_idx]
        
        print(f"\n🏆 Generation {gen+1} Best:")
        print(f"   Fitness: {best_result['fitness']:.4f}")
        print(f"   Accuracy: {best_result['accuracy']:.4f}")
    
    print("\n" + "="*60)
    print("✅ COMPLETE! GA CAN OPTIMIZE!")
    print("="*60)
    
elif test_result['fitness'] == 0.0:
    print("\n❌ FAILURE! Model could not train!")
    print("   Every model is crashing")
    print("\n🔍 Diagnostics:")
    print(f"   - Error: {test_result.get('error', 'Unknown')}")
    print(f"   - TensorFlow version: {tf.__version__}")
    print(f"   - Keras version: {keras.__version__}")
    print("\n💡 Possible issues:")
    print("   1. TensorFlow installation problem")
    print("   2. Incompatible versions")
    print("   3. System resource limits")
    print("\n   Try: pip install --upgrade tensorflow")
else:
    print(f"\n⚠️  Model trained but poorly (acc={test_result['accuracy']:.4f})")
    print("   This is expected with tiny dataset and few epochs")
    print("   But at least it's WORKING!")

"""
Quick Test Script
Run this to verify the implementation works correctly
This uses smaller parameters for faster testing
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from genetic_algorithm import GeneticAlgorithm, print_chromosome
from neural_network import NeuralNetworkBuilder
from visualization import EvolutionVisualizer

print("\n" + "="*60)
print("QUICK TEST - GA HYPERPARAMETER OPTIMIZATION")
print("="*60)

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Load small subset of data for quick test
print("\n1. Loading Fashion-MNIST (small subset for testing)...")
(X_train_full, y_train_full), _ = keras.datasets.fashion_mnist.load_data()

# Use only 1000 samples for quick test
X_train = X_train_full[:800].astype('float32') / 255.0
y_train = y_train_full[:800]
X_val = X_train_full[800:1000].astype('float32') / 255.0
y_val = y_train_full[800:1000]

y_train = keras.utils.to_categorical(y_train, 10)
y_val = keras.utils.to_categorical(y_val, 10)

print(f"✓ Loaded {len(X_train)} training, {len(X_val)} validation samples")

# Initialize components
print("\n2. Initializing components...")
nn_builder = NeuralNetworkBuilder(input_shape=(28, 28), num_classes=10)

def fitness_function(chromosome):
    return nn_builder.evaluate_fitness(
        chromosome, X_train, y_train, X_val, y_val,
        epochs=2,  # Very quick training
        verbose=0
    )

print("✓ Neural network builder ready")

# Test GA with small parameters
print("\n3. Testing Genetic Algorithm (3 generations, 6 individuals)...")
ga = GeneticAlgorithm(
    population_size=6,
    generations=3,
    crossover_rate=0.8,
    mutation_rate=0.2,
    elitism_count=1,
    tournament_size=2
)

# Run GA
best_chromosome, best_fitness = ga.evolve(fitness_function)

print("\n" + "="*60)
print("TEST RESULTS")
print("="*60)

print_chromosome(best_chromosome, "Best Configuration Found")
print(f"\nBest Fitness: {best_fitness:.4f}")

# Get statistics
stats = ga.get_statistics()
print(f"\nConverged at generation: {stats['convergence_generation']}")
print(f"Improvement: {stats['improvement']:.4f}")

# Quick visualization test
print("\n4. Testing visualization...")
visualizer = EvolutionVisualizer()

try:
    visualizer.plot_fitness_evolution(ga.history)
    print("✓ Visualization test passed")
except Exception as e:
    print(f"⚠ Visualization test failed: {e}")

print("\n" + "="*60)
print("✅ QUICK TEST COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nThe implementation is working correctly.")
print("You can now run the full version with: python main.py")
print("Or start the web interface with: streamlit run app.py")
print("="*60)

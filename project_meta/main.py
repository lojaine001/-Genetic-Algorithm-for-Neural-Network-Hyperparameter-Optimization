"""
Main Script for GA Hyperparameter Optimization
Execute this file to run the complete project
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import json
from datetime import datetime

# Import custom modules
from genetic_algorithm import GeneticAlgorithm, print_chromosome
from neural_network import NeuralNetworkBuilder, BaselineComparator
from visualization import EvolutionVisualizer


def load_fashion_mnist():
    """
    Load and preprocess Fashion-MNIST dataset
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    print("\n" + "="*60)
    print("LOADING FASHION-MNIST DATASET")
    print("="*60)
    
    # Load data
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    # Normalize to [0, 1]
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Split training into train and validation (80-20 split)
    split_idx = int(0.8 * len(X_train_full))
    X_train = X_train_full[:split_idx]
    y_train = y_train_full[:split_idx]
    X_val = X_train_full[split_idx:]
    y_val = y_train_full[split_idx:]
    
    # Convert labels to one-hot encoding
    y_train = keras.utils.to_categorical(y_train, 10)
    y_val = keras.utils.to_categorical(y_val, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print(f"✓ Dataset loaded successfully")
    print(f"  Training samples:   {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Test samples:       {len(X_test):,}")
    print(f"  Image shape:        {X_train[0].shape}")
    print(f"  Number of classes:  {y_train.shape[1]}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    """
    Main execution function
    """
    print("\n" + "="*60)
    print("GENETIC ALGORITHM FOR HYPERPARAMETER OPTIMIZATION")
    print("Neural Network Tuning on Fashion-MNIST")
    print("="*60)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/plots', exist_ok=True)
    
    print(f"\n✓ Results will be saved to: {results_dir}/")
    
    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist()
    
    # Initialize neural network builder
    nn_builder = NeuralNetworkBuilder(
        input_shape=(28, 28),
        num_classes=10
    )
    
    # Create fitness function
    def fitness_function(chromosome):
        return nn_builder.evaluate_fitness(
            chromosome, 
            X_train, y_train, 
            X_val, y_val,
            epochs=5,  # Quick training for GA
            verbose=0
        )
    
    # ===== RUN GENETIC ALGORITHM =====
    print("\n" + "="*60)
    print("STARTING GENETIC ALGORITHM")
    print("="*60)
    
    ga = GeneticAlgorithm(
        population_size=20,
        generations=15,
        crossover_rate=0.8,
        mutation_rate=0.2,
        elitism_count=2,
        tournament_size=3
    )
    
    # Run evolution
    best_chromosome, best_fitness = ga.evolve(fitness_function)
    
    # Save GA results
    ga.save_history(f'{results_dir}/ga_history.json')
    
    # Print best solution
    print("\n" + "="*60)
    print("GENETIC ALGORITHM - BEST SOLUTION FOUND")
    print("="*60)
    print_chromosome(best_chromosome, "Best Hyperparameters")
    print(f"\nBest Fitness: {best_fitness:.4f}")
    
    # Get GA statistics
    ga_stats = ga.get_statistics()
    
    print(f"\nGeneration of best solution: {ga_stats['convergence_generation']}")
    print(f"Improvement from initial: {ga_stats['improvement']:.4f}")
    
    # ===== TRAIN FINAL MODEL WITH MORE EPOCHS =====
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL (Extended Training)")
    print("="*60)
    
    final_results = nn_builder.evaluate_fitness(
        best_chromosome,
        X_train, y_train,
        X_val, y_val,
        epochs=20,  # More epochs for final model
        verbose=1
    )
    
    print(f"\n✓ Final Model Results:")
    print(f"  Validation Accuracy: {final_results['accuracy']:.4f}")
    print(f"  Training Time: {final_results['training_time']:.2f}s")
    print(f"  Model Parameters: {final_results['n_parameters']:,}")
    
    # ===== BASELINE COMPARISONS =====
    print("\n" + "="*60)
    print("RUNNING BASELINE COMPARISONS")
    print("="*60)
    
    comparator = BaselineComparator(nn_builder)
    
    # Random Search
    random_results = comparator.random_search(
        n_trials=20,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )
    
    # Default Configuration
    default_results = comparator.default_config(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )
    
    # Compare all methods
    ga_results = {
        'best_fitness': best_fitness,
        'best_results': final_results,
        'best_config': best_chromosome
    }
    
    comparison = comparator.compare_all(
        ga_results=ga_results,
        random_results=random_results,
        default_results=default_results
    )
    
    # Save comparison
    with open(f'{results_dir}/comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # ===== CREATE VISUALIZATIONS =====
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    visualizer = EvolutionVisualizer()
    
    # Generate all plots
    visualizer.create_summary_report(
        history=ga.history,
        best_chromosome=best_chromosome,
        comparison=comparison,
        save_dir=f'{results_dir}/plots'
    )
    
    # Architecture visualization
    visualizer.plot_final_model_architecture(
        best_chromosome,
        save_path=f'{results_dir}/plots/best_architecture.png'
    )
    
    # ===== FINAL SUMMARY =====
    print("\n" + "="*60)
    print("PROJECT COMPLETE - SUMMARY")
    print("="*60)
    
    print("\n📊 RESULTS COMPARISON:")
    print(f"  Genetic Algorithm:  Accuracy = {final_results['accuracy']:.4f}")
    print(f"  Random Search:      Accuracy = {random_results['best_results']['accuracy']:.4f}")
    print(f"  Default Config:     Accuracy = {default_results['results']['accuracy']:.4f}")
    
    ga_better_than_random = (final_results['accuracy'] > 
                             random_results['best_results']['accuracy'])
    ga_better_than_default = (final_results['accuracy'] > 
                              default_results['results']['accuracy'])
    
    if ga_better_than_random and ga_better_than_default:
        print("\n✅ SUCCESS: GA found better hyperparameters than both baselines!")
    elif ga_better_than_random or ga_better_than_default:
        print("\n⚠️  PARTIAL SUCCESS: GA outperformed some baselines")
    else:
        print("\n❌ GA did not outperform baselines - may need tuning")
    
    print(f"\n📁 All results saved to: {results_dir}/")
    print("="*60)
    
    return {
        'ga_results': ga_results,
        'random_results': random_results,
        'default_results': default_results,
        'comparison': comparison,
        'results_dir': results_dir
    }


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run main
    results = main()
    
    print("\n✨ Project execution complete!")
    print("Check the results directory for all outputs and visualizations.")

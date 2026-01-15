"""
FIXED VERSION - Main Script for GA Hyperparameter Optimization
This version includes proper error handling and visible results
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from genetic_algorithm import GeneticAlgorithm, print_chromosome
from neural_network_fixed import NeuralNetworkBuilder, BaselineComparator
from visualization import EvolutionVisualizer


def load_fashion_mnist_small():
    """
    Load SMALLER subset of Fashion-MNIST for FASTER, VISIBLE results
    """
    print("\n" + "="*60)
    print("LOADING FASHION-MNIST DATASET (SMALL SUBSET)")
    print("="*60)
    
    # Load data
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    # USE MUCH SMALLER SUBSET FOR VISIBLE RESULTS
    # This makes training fast enough to see differences
    train_size = 5000  # Down from 60,000
    val_size = 1000    # Down from 12,000
    
    X_train = X_train_full[:train_size].astype('float32') / 255.0
    y_train = y_train_full[:train_size]
    X_val = X_train_full[train_size:train_size+val_size].astype('float32') / 255.0
    y_val = y_train_full[train_size:train_size+val_size]
    
    # Small test set
    X_test = X_test[:1000].astype('float32') / 255.0
    y_test = y_test[:1000]
    
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
    print(f"\n⚠️  Using SMALL subset for FAST visible results")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    """
    Main execution function - FIXED VERSION
    """
    print("\n" + "="*60)
    print("GENETIC ALGORITHM FOR HYPERPARAMETER OPTIMIZATION")
    print("Neural Network Tuning on Fashion-MNIST - FIXED VERSION")
    print("="*60)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/plots', exist_ok=True)
    
    print(f"\n✓ Results will be saved to: {results_dir}/")
    
    # Load SMALL dataset for visible results
    X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist_small()
    
    # Initialize neural network builder
    nn_builder = NeuralNetworkBuilder(
        input_shape=(28, 28),
        num_classes=10
    )
    
    # Create fitness function with MORE EPOCHS for visible learning
    def fitness_function(chromosome):
        print(f"  Training with config: layers={chromosome['n_layers']}, "
              f"lr={chromosome['learning_rate']:.6f}, "
              f"optimizer={chromosome['optimizer']}")
        
        result = nn_builder.evaluate_fitness(
            chromosome, 
            X_train, y_train, 
            X_val, y_val,
            epochs=10,  # INCREASED from 5 to see actual learning
            verbose=1   # SHOW training progress
        )
        
        print(f"  → Fitness: {result['fitness']:.4f}, "
              f"Accuracy: {result['accuracy']:.4f}")
        
        return result
    
    # ===== RUN GENETIC ALGORITHM WITH SMALLER POPULATION =====
    print("\n" + "="*60)
    print("STARTING GENETIC ALGORITHM")
    print("="*60)
    
    ga = GeneticAlgorithm(
        population_size=10,  # REDUCED from 20 for faster results
        generations=8,       # REDUCED from 15 for faster completion
        crossover_rate=0.8,
        mutation_rate=0.3,   # INCREASED for more exploration
        elitism_count=2,
        tournament_size=3
    )
    
    # Run evolution
    print("\nStarting evolution...")
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
    
    # ===== TRAIN FINAL MODEL WITH BEST CONFIG =====
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
    
    # ===== SIMPLIFIED BASELINE COMPARISONS =====
    print("\n" + "="*60)
    print("RUNNING BASELINE COMPARISONS")
    print("="*60)
    
    comparator = BaselineComparator(nn_builder)
    
    # Random Search with fewer trials
    print("\n1. Random Search Baseline...")
    random_results = comparator.random_search(
        n_trials=5,  # REDUCED from 20 for speed
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )
    
    # Default Configuration
    print("\n2. Default Configuration Baseline...")
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
        json.dump(comparison, f, indent=2, default=str)
    
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
        print("\n❌ GA did not outperform baselines - may need more generations")
    
    print(f"\n📁 All results saved to: {results_dir}/")
    
    # Print actual numbers to verify they're not zero
    print("\n" + "="*60)
    print("VERIFICATION - ACTUAL VALUES")
    print("="*60)
    print(f"GA Best Fitness (should be >0):     {best_fitness}")
    print(f"GA Best Accuracy (should be >0):    {final_results['accuracy']}")
    print(f"Random Best Accuracy (should be >0): {random_results['best_results']['accuracy']}")
    print(f"Default Accuracy (should be >0):     {default_results['results']['accuracy']}")
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
    
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    
    # Run main
    try:
        results = main()
        print("\n✨ Project execution complete!")
        print("Check the results directory for all outputs and visualizations.")
    except Exception as e:
        print(f"\n❌ ERROR occurred: {str(e)}")
        import traceback
        traceback.print_exc()

"""
Neural Network Builder and Fitness Evaluator
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from typing import Dict, Any, Tuple
import time


class NeuralNetworkBuilder:
    """
    Builds and trains neural networks based on chromosome configuration
    """
    
    def __init__(self, input_shape: Tuple, num_classes: int):
        """
        Initialize builder
        
        Args:
            input_shape: Shape of input data (e.g., (28, 28) for MNIST)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # For fitness normalization
        self.max_training_time = 60.0  # seconds
        self.max_model_complexity = 1e6  # parameters
    
    def build_model(self, chromosome: Dict[str, Any]) -> keras.Model:
        """
        Build a Keras model based on chromosome
        
        Args:
            chromosome: Hyperparameter configuration
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential()
        
        # Input layer - flatten if image data
        if len(self.input_shape) > 1:
            model.add(layers.Flatten(input_shape=self.input_shape))
        else:
            model.add(layers.InputLayer(input_shape=self.input_shape))
        
        # Hidden layers based on chromosome
        n_layers = chromosome['n_layers']
        layer_sizes = chromosome['layer_sizes']
        activation = chromosome['activation']
        dropout = chromosome['dropout']
        
        for i in range(n_layers):
            # Dense layer
            model.add(layers.Dense(
                layer_sizes[i],
                activation=activation,
                name=f'hidden_{i+1}'
            ))
            
            # Dropout for regularization
            if dropout > 0:
                model.add(layers.Dropout(dropout, name=f'dropout_{i+1}'))
        
        # Output layer
        model.add(layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        ))
        
        # Get optimizer
        optimizer = self._get_optimizer(
            chromosome['optimizer'],
            chromosome['learning_rate']
        )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _get_optimizer(self, optimizer_name: str, learning_rate: float):
        """
        Get Keras optimizer
        
        Args:
            optimizer_name: Name of optimizer
            learning_rate: Learning rate
            
        Returns:
            Keras optimizer
        """
        if optimizer_name == 'adam':
            return optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            return optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_name == 'rmsprop':
            return optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def count_parameters(self, model: keras.Model) -> int:
        """
        Count total trainable parameters in model
        
        Args:
            model: Keras model
            
        Returns:
            Number of parameters
        """
        return int(np.sum([
            np.prod(v.get_shape().as_list()) 
            for v in model.trainable_weights
        ]))
    
    def evaluate_fitness(
        self,
        chromosome: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 5,
        verbose: int = 0
    ) -> Dict[str, Any]:
        """
        Build, train, and evaluate model to compute fitness
        
        Args:
            chromosome: Hyperparameter configuration
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of training epochs
            verbose: Verbosity level
            
        Returns:
            Dictionary with fitness and metrics
        """
        try:
            # Build model
            model = self.build_model(chromosome)
            
            # Count parameters
            n_parameters = self.count_parameters(model)
            
            # Train model and measure time
            start_time = time.time()
            
            history = model.fit(
                X_train, y_train,
                batch_size=chromosome['batch_size'],
                epochs=epochs,
                validation_data=(X_val, y_val),
                verbose=verbose,
                # Early stopping to save time
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=2,
                        restore_best_weights=True
                    )
                ]
            )
            
            training_time = time.time() - start_time
            
            # Get best validation accuracy
            val_accuracy = max(history.history['val_accuracy'])
            
            # Normalize metrics for fitness calculation
            norm_accuracy = val_accuracy
            norm_time = 1 - min(training_time / self.max_training_time, 1.0)
            norm_complexity = 1 - min(n_parameters / self.max_model_complexity, 1.0)
            
            # Calculate fitness (weighted combination)
            # Prioritize accuracy, but also reward fast and simple models
            fitness = (
                0.70 * norm_accuracy +      # 70% weight on accuracy
                0.20 * norm_time +           # 20% weight on speed
                0.10 * norm_complexity       # 10% weight on simplicity
            )
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
            
            return {
                'fitness': fitness,
                'accuracy': val_accuracy,
                'training_time': training_time,
                'n_parameters': n_parameters,
                'norm_accuracy': norm_accuracy,
                'norm_time': norm_time,
                'norm_complexity': norm_complexity
            }
            
        except Exception as e:
            # If model fails (e.g., invalid config), return low fitness
            print(f"  ⚠ Model evaluation failed: {str(e)}")
            
            return {
                'fitness': 0.0,
                'accuracy': 0.0,
                'training_time': 0.0,
                'n_parameters': 0,
                'norm_accuracy': 0.0,
                'norm_time': 0.0,
                'norm_complexity': 0.0,
                'error': str(e)
            }


class BaselineComparator:
    """
    Compare GA results with baseline methods
    """
    
    def __init__(self, nn_builder: NeuralNetworkBuilder):
        """
        Initialize comparator
        
        Args:
            nn_builder: Neural network builder instance
        """
        self.nn_builder = nn_builder
    
    def random_search(
        self,
        n_trials: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run random search baseline
        
        Args:
            n_trials: Number of random configurations to try
            X_train, y_train, X_val, y_val: Data
            
        Returns:
            Best configuration and results
        """
        print(f"\nRunning Random Search with {n_trials} trials...")
        
        from genetic_algorithm import GeneticAlgorithm
        
        ga = GeneticAlgorithm()  # Just to use search space
        
        best_fitness = -float('inf')
        best_config = None
        best_results = None
        all_results = []
        
        for i in range(n_trials):
            print(f"\nRandom Trial {i+1}/{n_trials}")
            
            # Generate random configuration
            config = ga.create_random_chromosome()
            
            # Evaluate
            results = self.nn_builder.evaluate_fitness(
                config, X_train, y_train, X_val, y_val
            )
            
            all_results.append({
                'config': config,
                'results': results
            })
            
            print(f"  Fitness: {results['fitness']:.4f} | "
                  f"Accuracy: {results['accuracy']:.4f}")
            
            # Track best
            if results['fitness'] > best_fitness:
                best_fitness = results['fitness']
                best_config = config
                best_results = results
        
        return {
            'method': 'Random Search',
            'best_fitness': best_fitness,
            'best_config': best_config,
            'best_results': best_results,
            'all_results': all_results
        }
    
    def default_config(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate default/standard configuration
        
        Returns:
            Default configuration results
        """
        print("\nEvaluating Default Configuration...")
        
        # Standard configuration
        config = {
            'n_layers': 2,
            'layer_sizes': [128, 64],
            'learning_rate': 0.001,
            'batch_size': 32,
            'dropout': 0.2,
            'optimizer': 'adam',
            'activation': 'relu'
        }
        
        # Evaluate
        results = self.nn_builder.evaluate_fitness(
            config, X_train, y_train, X_val, y_val
        )
        
        print(f"  Fitness: {results['fitness']:.4f} | "
              f"Accuracy: {results['accuracy']:.4f}")
        
        return {
            'method': 'Default Config',
            'config': config,
            'results': results
        }
    
    def compare_all(
        self,
        ga_results: Dict[str, Any],
        random_results: Dict[str, Any],
        default_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare all methods
        
        Returns:
            Comparison summary
        """
        comparison = {
            'Genetic Algorithm': {
                'fitness': ga_results['best_fitness'],
                'accuracy': ga_results['best_results']['accuracy'],
                'training_time': ga_results['best_results']['training_time'],
                'n_parameters': ga_results['best_results']['n_parameters']
            },
            'Random Search': {
                'fitness': random_results['best_fitness'],
                'accuracy': random_results['best_results']['accuracy'],
                'training_time': random_results['best_results']['training_time'],
                'n_parameters': random_results['best_results']['n_parameters']
            },
            'Default Config': {
                'fitness': default_results['results']['fitness'],
                'accuracy': default_results['results']['accuracy'],
                'training_time': default_results['results']['training_time'],
                'n_parameters': default_results['results']['n_parameters']
            }
        }
        
        # Print comparison table
        print("\n" + "="*80)
        print("COMPARISON OF METHODS")
        print("="*80)
        print(f"{'Method':<20} {'Fitness':>12} {'Accuracy':>12} {'Time (s)':>12} {'Parameters':>15}")
        print("-"*80)
        
        for method, metrics in comparison.items():
            print(f"{method:<20} "
                  f"{metrics['fitness']:>12.4f} "
                  f"{metrics['accuracy']:>12.4f} "
                  f"{metrics['training_time']:>12.2f} "
                  f"{metrics['n_parameters']:>15,}")
        
        print("="*80)
        
        return comparison

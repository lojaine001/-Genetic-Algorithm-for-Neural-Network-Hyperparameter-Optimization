"""
Genetic Algorithm for Neural Network Hyperparameter Optimization
Author: Lojaine
ISGA - Metaheuristics Course Project
"""

import numpy as np
import random
import time
import json
from copy import deepcopy
from typing import Dict, List, Tuple, Any


class GeneticAlgorithm:
    """
    Genetic Algorithm for optimizing neural network hyperparameters
    """
    
    def __init__(
        self,
        population_size: int = 20,
        generations: int = 15,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        elitism_count: int = 2,
        tournament_size: int = 3
    ):
        """
        Initialize Genetic Algorithm
        
        Args:
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism_count: Number of best individuals to preserve
            tournament_size: Size of tournament for selection
        """
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        
        # Evolution tracking
        self.population = []
        self.history = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        
        # Hyperparameter search space
        self.search_space = {
            'n_layers': [1, 2, 3, 4],
            'layer_sizes': [16, 32, 64, 128, 256],
            'learning_rate': (0.0001, 0.1),  # continuous range
            'batch_size': [16, 32, 64, 128],
            'dropout': (0.0, 0.5),  # continuous range
            'optimizer': ['adam', 'sgd', 'rmsprop'],
            'activation': ['relu', 'tanh', 'sigmoid']
        }
    
    def create_random_chromosome(self) -> Dict[str, Any]:
        """
        Create a random chromosome (individual)
        
        Returns:
            Dictionary representing a chromosome
        """
        n_layers = random.choice(self.search_space['n_layers'])
        
        chromosome = {
            'n_layers': n_layers,
            'layer_sizes': [
                random.choice(self.search_space['layer_sizes']) 
                for _ in range(n_layers)
            ],
            'learning_rate': random.uniform(*self.search_space['learning_rate']),
            'batch_size': random.choice(self.search_space['batch_size']),
            'dropout': random.uniform(*self.search_space['dropout']),
            'optimizer': random.choice(self.search_space['optimizer']),
            'activation': random.choice(self.search_space['activation'])
        }
        
        return chromosome
    
    def initialize_population(self):
        """
        Create initial population with random individuals
        """
        print(f"Initializing population of {self.population_size} individuals...")
        self.population = [
            self.create_random_chromosome() 
            for _ in range(self.population_size)
        ]
        print(f"✓ Population initialized")
    
    def tournament_selection(
        self, 
        fitness_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Tournament selection: select best from k random individuals
        
        Args:
            fitness_scores: List of fitness values for population
            
        Returns:
            Selected chromosome
        """
        # Select k random indices
        tournament_indices = random.sample(
            range(len(self.population)), 
            self.tournament_size
        )
        
        # Find best in tournament
        best_idx = max(
            tournament_indices, 
            key=lambda idx: fitness_scores[idx]
        )
        
        return deepcopy(self.population[best_idx])
    
    def crossover(
        self, 
        parent1: Dict[str, Any], 
        parent2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Uniform crossover: randomly select genes from each parent
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Child chromosome
        """
        child = {}
        
        for key in parent1.keys():
            # Randomly choose from parent1 or parent2
            if random.random() < 0.5:
                child[key] = deepcopy(parent1[key])
            else:
                child[key] = deepcopy(parent2[key])
        
        # Ensure layer_sizes matches n_layers
        n_layers = child['n_layers']
        if len(child['layer_sizes']) != n_layers:
            # Adjust layer_sizes to match n_layers
            if len(child['layer_sizes']) < n_layers:
                # Add more layers
                while len(child['layer_sizes']) < n_layers:
                    child['layer_sizes'].append(
                        random.choice(self.search_space['layer_sizes'])
                    )
            else:
                # Remove extra layers
                child['layer_sizes'] = child['layer_sizes'][:n_layers]
        
        return child
    
    def mutate(self, chromosome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate chromosome by randomly changing genes
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        mutated = deepcopy(chromosome)
        
        # Mutate each gene with mutation_rate probability
        for key in mutated.keys():
            if random.random() < self.mutation_rate:
                if key == 'n_layers':
                    old_n = mutated['n_layers']
                    mutated['n_layers'] = random.choice(self.search_space['n_layers'])
                    # Adjust layer_sizes
                    new_n = mutated['n_layers']
                    if new_n > old_n:
                        # Add layers
                        mutated['layer_sizes'].extend([
                            random.choice(self.search_space['layer_sizes'])
                            for _ in range(new_n - old_n)
                        ])
                    elif new_n < old_n:
                        # Remove layers
                        mutated['layer_sizes'] = mutated['layer_sizes'][:new_n]
                
                elif key == 'layer_sizes':
                    # Mutate one random layer size
                    if len(mutated['layer_sizes']) > 0:
                        idx = random.randint(0, len(mutated['layer_sizes']) - 1)
                        mutated['layer_sizes'][idx] = random.choice(
                            self.search_space['layer_sizes']
                        )
                
                elif key == 'learning_rate':
                    # Small perturbation or completely random
                    if random.random() < 0.5:
                        # Small change
                        delta = random.uniform(-0.01, 0.01)
                        mutated['learning_rate'] = np.clip(
                            mutated['learning_rate'] + delta,
                            *self.search_space['learning_rate']
                        )
                    else:
                        # Complete reset
                        mutated['learning_rate'] = random.uniform(
                            *self.search_space['learning_rate']
                        )
                
                elif key == 'batch_size':
                    mutated['batch_size'] = random.choice(
                        self.search_space['batch_size']
                    )
                
                elif key == 'dropout':
                    # Small perturbation or completely random
                    if random.random() < 0.5:
                        delta = random.uniform(-0.1, 0.1)
                        mutated['dropout'] = np.clip(
                            mutated['dropout'] + delta,
                            *self.search_space['dropout']
                        )
                    else:
                        mutated['dropout'] = random.uniform(
                            *self.search_space['dropout']
                        )
                
                elif key == 'optimizer':
                    mutated['optimizer'] = random.choice(
                        self.search_space['optimizer']
                    )
                
                elif key == 'activation':
                    mutated['activation'] = random.choice(
                        self.search_space['activation']
                    )
        
        return mutated
    
    def get_elite(
        self, 
        fitness_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Get elite individuals (best performing)
        
        Args:
            fitness_scores: List of fitness values
            
        Returns:
            List of elite chromosomes
        """
        # Get indices of top performers
        elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
        
        # Return copies of elite individuals
        return [deepcopy(self.population[idx]) for idx in elite_indices]
    
    def evolve(
        self, 
        fitness_function,
        callback=None
    ) -> Tuple[Dict[str, Any], float]:
        """
        Main evolution loop
        
        Args:
            fitness_function: Function to evaluate fitness
            callback: Optional callback for progress updates
            
        Returns:
            Tuple of (best_chromosome, best_fitness)
        """
        # Initialize population
        self.initialize_population()
        
        # Evolution loop
        for generation in range(self.generations):
            print(f"\n{'='*60}")
            print(f"Generation {generation + 1}/{self.generations}")
            print(f"{'='*60}")
            
            # Evaluate fitness for all individuals
            fitness_scores = []
            generation_details = []
            
            for idx, chromosome in enumerate(self.population):
                print(f"\nEvaluating Individual {idx + 1}/{self.population_size}...")
                
                # Evaluate fitness
                fitness_data = fitness_function(chromosome)
                fitness = fitness_data['fitness']
                fitness_scores.append(fitness)
                
                # Store details
                individual_info = {
                    'generation': generation + 1,
                    'individual': idx + 1,
                    'chromosome': chromosome,
                    **fitness_data
                }
                generation_details.append(individual_info)
                
                # Update best
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = deepcopy(chromosome)
                
                print(f"  Fitness: {fitness:.4f} | "
                      f"Accuracy: {fitness_data['accuracy']:.4f} | "
                      f"Time: {fitness_data['training_time']:.2f}s")
            
            # Calculate statistics
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            worst_fitness = min(fitness_scores)
            std_fitness = np.std(fitness_scores)
            
            # Save generation history
            generation_summary = {
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'worst_fitness': worst_fitness,
                'std_fitness': std_fitness,
                'best_chromosome': self.population[np.argmax(fitness_scores)],
                'details': generation_details
            }
            self.history.append(generation_summary)
            
            print(f"\n--- Generation {generation + 1} Summary ---")
            print(f"Best Fitness:    {best_fitness:.4f}")
            print(f"Average Fitness: {avg_fitness:.4f}")
            print(f"Worst Fitness:   {worst_fitness:.4f}")
            print(f"Std Dev:         {std_fitness:.4f}")
            
            # Callback for external updates (e.g., UI)
            if callback:
                callback(generation_summary)
            
            # Create next generation
            if generation < self.generations - 1:  # Don't evolve on last generation
                new_population = []
                
                # Elitism: keep best individuals
                elite = self.get_elite(fitness_scores)
                new_population.extend(elite)
                print(f"\n✓ Preserved {len(elite)} elite individuals")
                
                # Generate rest of population
                while len(new_population) < self.population_size:
                    # Selection
                    parent1 = self.tournament_selection(fitness_scores)
                    parent2 = self.tournament_selection(fitness_scores)
                    
                    # Crossover
                    if random.random() < self.crossover_rate:
                        child = self.crossover(parent1, parent2)
                    else:
                        child = deepcopy(parent1)
                    
                    # Mutation
                    if random.random() < self.mutation_rate:
                        child = self.mutate(child)
                    
                    new_population.append(child)
                
                # Update population
                self.population = new_population
                print(f"✓ New generation created")
        
        print(f"\n{'='*60}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*60}")
        print(f"Best Fitness Achieved: {self.best_fitness:.4f}")
        
        return self.best_individual, self.best_fitness
    
    def save_history(self, filepath: str):
        """
        Save evolution history to JSON file
        
        Args:
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
        print(f"\n✓ History saved to {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get evolution statistics
        
        Returns:
            Dictionary with statistics
        """
        if not self.history:
            return {}
        
        best_fitnesses = [gen['best_fitness'] for gen in self.history]
        avg_fitnesses = [gen['avg_fitness'] for gen in self.history]
        
        return {
            'total_generations': len(self.history),
            'final_best_fitness': self.best_fitness,
            'initial_best_fitness': best_fitnesses[0],
            'improvement': self.best_fitness - best_fitnesses[0],
            'avg_improvement': avg_fitnesses[-1] - avg_fitnesses[0],
            'convergence_generation': best_fitnesses.index(max(best_fitnesses)) + 1,
            'best_chromosome': self.best_individual
        }


# Utility function to print chromosome
def print_chromosome(chromosome: Dict[str, Any], title: str = "Chromosome"):
    """Pretty print a chromosome"""
    print(f"\n{title}:")
    print("-" * 50)
    for key, value in chromosome.items():
        if key == 'layer_sizes':
            print(f"  {key:15s}: {value}")
        elif key in ['learning_rate', 'dropout']:
            print(f"  {key:15s}: {value:.6f}")
        else:
            print(f"  {key:15s}: {value}")
    print("-" * 50)

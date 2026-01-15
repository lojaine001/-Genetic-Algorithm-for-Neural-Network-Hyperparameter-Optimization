"""
Visualization Module for GA Hyperparameter Optimization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import pandas as pd


class EvolutionVisualizer:
    """
    Create visualizations for genetic algorithm evolution
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style
        """
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 8)
    
    def plot_fitness_evolution(
        self, 
        history: List[Dict[str, Any]], 
        save_path: str = None
    ):
        """
        Plot fitness evolution over generations
        
        Args:
            history: GA evolution history
            save_path: Path to save plot
        """
        generations = [h['generation'] for h in history]
        best_fitness = [h['best_fitness'] for h in history]
        avg_fitness = [h['avg_fitness'] for h in history]
        worst_fitness = [h['worst_fitness'] for h in history]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot lines
        ax.plot(generations, best_fitness, 
                marker='o', linewidth=2, markersize=8,
                label='Best Fitness', color=self.colors[0])
        ax.plot(generations, avg_fitness, 
                marker='s', linewidth=2, markersize=6,
                label='Average Fitness', color=self.colors[1])
        ax.plot(generations, worst_fitness, 
                marker='^', linewidth=2, markersize=6,
                label='Worst Fitness', color=self.colors[2])
        
        # Fill between best and worst
        ax.fill_between(generations, worst_fitness, best_fitness, 
                        alpha=0.2, color=self.colors[0])
        
        # Formatting
        ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Fitness Score', fontsize=12, fontweight='bold')
        ax.set_title('Fitness Evolution Over Generations', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Annotations
        max_fitness_gen = generations[np.argmax(best_fitness)]
        max_fitness = max(best_fitness)
        ax.annotate(f'Best: {max_fitness:.4f}', 
                   xy=(max_fitness_gen, max_fitness),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved fitness evolution plot to {save_path}")
        
        plt.show()
    
    def plot_parameter_evolution(
        self, 
        history: List[Dict[str, Any]], 
        save_path: str = None
    ):
        """
        Plot how hyperparameters evolve over generations
        
        Args:
            history: GA evolution history
            save_path: Path to save plot
        """
        # Extract best chromosome from each generation
        generations = [h['generation'] for h in history]
        
        # Track continuous parameters
        learning_rates = [h['best_chromosome']['learning_rate'] for h in history]
        dropouts = [h['best_chromosome']['dropout'] for h in history]
        n_layers = [h['best_chromosome']['n_layers'] for h in history]
        batch_sizes = [h['best_chromosome']['batch_size'] for h in history]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Learning Rate
        axes[0, 0].plot(generations, learning_rates, 
                       marker='o', linewidth=2, color=self.colors[0])
        axes[0, 0].set_xlabel('Generation', fontweight='bold')
        axes[0, 0].set_ylabel('Learning Rate', fontweight='bold')
        axes[0, 0].set_title('Learning Rate Evolution', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Dropout
        axes[0, 1].plot(generations, dropouts, 
                       marker='s', linewidth=2, color=self.colors[1])
        axes[0, 1].set_xlabel('Generation', fontweight='bold')
        axes[0, 1].set_ylabel('Dropout Rate', fontweight='bold')
        axes[0, 1].set_title('Dropout Rate Evolution', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Number of Layers
        axes[1, 0].plot(generations, n_layers, 
                       marker='^', linewidth=2, color=self.colors[2])
        axes[1, 0].set_xlabel('Generation', fontweight='bold')
        axes[1, 0].set_ylabel('Number of Layers', fontweight='bold')
        axes[1, 0].set_title('Network Depth Evolution', fontweight='bold')
        axes[1, 0].set_yticks([1, 2, 3, 4])
        axes[1, 0].grid(True, alpha=0.3)
        
        # Batch Size
        axes[1, 1].plot(generations, batch_sizes, 
                       marker='d', linewidth=2, color=self.colors[3])
        axes[1, 1].set_xlabel('Generation', fontweight='bold')
        axes[1, 1].set_ylabel('Batch Size', fontweight='bold')
        axes[1, 1].set_title('Batch Size Evolution', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Hyperparameter Evolution Over Generations', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved parameter evolution plot to {save_path}")
        
        plt.show()
    
    def plot_categorical_parameters(
        self, 
        history: List[Dict[str, Any]], 
        save_path: str = None
    ):
        """
        Plot distribution of categorical parameters across generations
        
        Args:
            history: GA evolution history
            save_path: Path to save plot
        """
        # Extract all individuals from all generations
        optimizers = []
        activations = []
        generations_list = []
        
        for gen_data in history:
            gen = gen_data['generation']
            for individual in gen_data['details']:
                optimizers.append(individual['chromosome']['optimizer'])
                activations.append(individual['chromosome']['activation'])
                generations_list.append(gen)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Generation': generations_list,
            'Optimizer': optimizers,
            'Activation': activations
        })
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Optimizer distribution
        optimizer_counts = df.groupby(['Generation', 'Optimizer']).size().unstack(fill_value=0)
        optimizer_counts.plot(kind='bar', stacked=True, ax=axes[0], 
                             color=self.colors[:3])
        axes[0].set_xlabel('Generation', fontweight='bold')
        axes[0].set_ylabel('Count', fontweight='bold')
        axes[0].set_title('Optimizer Distribution Across Generations', fontweight='bold')
        axes[0].legend(title='Optimizer')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Activation distribution
        activation_counts = df.groupby(['Generation', 'Activation']).size().unstack(fill_value=0)
        activation_counts.plot(kind='bar', stacked=True, ax=axes[1], 
                              color=self.colors[3:6])
        axes[1].set_xlabel('Generation', fontweight='bold')
        axes[1].set_ylabel('Count', fontweight='bold')
        axes[1].set_title('Activation Function Distribution Across Generations', 
                         fontweight='bold')
        axes[1].legend(title='Activation')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved categorical parameters plot to {save_path}")
        
        plt.show()
    
    def plot_comparison(
        self, 
        comparison: Dict[str, Dict[str, float]], 
        save_path: str = None
    ):
        """
        Plot comparison between GA and baseline methods
        
        Args:
            comparison: Comparison dictionary
            save_path: Path to save plot
        """
        methods = list(comparison.keys())
        metrics = ['fitness', 'accuracy', 'training_time', 'n_parameters']
        metric_names = ['Fitness', 'Accuracy', 'Training Time (s)', 'Parameters']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            values = [comparison[method][metric] for method in methods]
            
            # Normalize parameters for better visualization
            if metric == 'n_parameters':
                values = [v / 1000 for v in values]  # Convert to thousands
                metric_name = 'Parameters (×1000)'
            
            bars = axes[idx].bar(methods, values, color=self.colors[:len(methods)])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}',
                             ha='center', va='bottom', fontweight='bold')
            
            axes[idx].set_ylabel(metric_name, fontweight='bold')
            axes[idx].set_title(f'{metric_name} Comparison', fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Rotate x labels
            axes[idx].tick_params(axis='x', rotation=15)
        
        plt.suptitle('Optimization Methods Comparison', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved comparison plot to {save_path}")
        
        plt.show()
    
    def plot_diversity(
        self, 
        history: List[Dict[str, Any]], 
        save_path: str = None
    ):
        """
        Plot population diversity over generations
        
        Args:
            history: GA evolution history
            save_path: Path to save plot
        """
        generations = [h['generation'] for h in history]
        std_fitness = [h['std_fitness'] for h in history]
        
        # Calculate diversity metrics
        diversity_scores = []
        
        for gen_data in history:
            # Extract all fitness scores from generation
            fitness_scores = [ind['fitness'] for ind in gen_data['details']]
            
            # Calculate coefficient of variation (normalized std)
            if np.mean(fitness_scores) > 0:
                cv = np.std(fitness_scores) / np.mean(fitness_scores)
            else:
                cv = 0
            
            diversity_scores.append(cv)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Standard deviation
        axes[0].plot(generations, std_fitness, 
                    marker='o', linewidth=2, color=self.colors[0])
        axes[0].set_xlabel('Generation', fontweight='bold')
        axes[0].set_ylabel('Fitness Std Dev', fontweight='bold')
        axes[0].set_title('Fitness Standard Deviation', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Coefficient of variation
        axes[1].plot(generations, diversity_scores, 
                    marker='s', linewidth=2, color=self.colors[1])
        axes[1].set_xlabel('Generation', fontweight='bold')
        axes[1].set_ylabel('Coefficient of Variation', fontweight='bold')
        axes[1].set_title('Population Diversity (CV)', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Population Diversity Over Generations', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved diversity plot to {save_path}")
        
        plt.show()
    
    def create_summary_report(
        self,
        history: List[Dict[str, Any]],
        best_chromosome: Dict[str, Any],
        comparison: Dict[str, Dict[str, float]],
        save_dir: str = 'plots'
    ):
        """
        Create all visualizations and save them
        
        Args:
            history: GA evolution history
            best_chromosome: Best chromosome found
            comparison: Comparison with baselines
            save_dir: Directory to save plots
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("\nGenerating visualizations...")
        
        # 1. Fitness evolution
        self.plot_fitness_evolution(
            history, 
            save_path=f'{save_dir}/fitness_evolution.png'
        )
        
        # 2. Parameter evolution
        self.plot_parameter_evolution(
            history, 
            save_path=f'{save_dir}/parameter_evolution.png'
        )
        
        # 3. Categorical parameters
        self.plot_categorical_parameters(
            history, 
            save_path=f'{save_dir}/categorical_params.png'
        )
        
        # 4. Diversity
        self.plot_diversity(
            history, 
            save_path=f'{save_dir}/diversity.png'
        )
        
        # 5. Comparison
        if comparison:
            self.plot_comparison(
                comparison, 
                save_path=f'{save_dir}/comparison.png'
            )
        
        print(f"\n✓ All visualizations saved to {save_dir}/")
    
    def plot_final_model_architecture(
        self,
        chromosome: Dict[str, Any],
        save_path: str = None
    ):
        """
        Visualize the final best model architecture
        
        Args:
            chromosome: Best chromosome
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Build layer info
        layers = ['Input']
        layers.extend([f"Hidden {i+1}\n({chromosome['layer_sizes'][i]} neurons)" 
                      for i in range(chromosome['n_layers'])])
        layers.append('Output')
        
        n_layers = len(layers)
        
        # Draw layers
        for i, layer in enumerate(layers):
            # Calculate position
            y = n_layers - i - 1
            
            # Draw box
            if i == 0:
                color = self.colors[0]
            elif i == n_layers - 1:
                color = self.colors[2]
            else:
                color = self.colors[1]
            
            rect = plt.Rectangle((0.2, y - 0.3), 0.6, 0.6, 
                                facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            # Add text
            ax.text(0.5, y, layer, ha='center', va='center', 
                   fontsize=10, fontweight='bold')
            
            # Draw arrow to next layer
            if i < n_layers - 1:
                ax.arrow(0.5, y - 0.35, 0, -0.2, 
                        head_width=0.08, head_length=0.08, 
                        fc='gray', ec='gray', linewidth=1.5)
        
        # Add hyperparameters text
        info_text = (
            f"Activation: {chromosome['activation']}\n"
            f"Optimizer: {chromosome['optimizer']}\n"
            f"Learning Rate: {chromosome['learning_rate']:.6f}\n"
            f"Batch Size: {chromosome['batch_size']}\n"
            f"Dropout: {chromosome['dropout']:.3f}"
        )
        
        ax.text(1.1, n_layers/2, info_text, 
               fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlim(0, 1.8)
        ax.set_ylim(-0.5, n_layers - 0.5)
        ax.axis('off')
        ax.set_title('Best Neural Network Architecture', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved architecture plot to {save_path}")
        
        plt.show()

"""
Interactive Streamlit App for GA Hyperparameter Optimization Demo
UPDATED VERSION - Works with memory-safe code
Run with: streamlit run app_fixed.py
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import pandas as pd
import time
import matplotlib.pyplot as plt
import random
from copy import deepcopy
import gc
import os
import warnings
warnings.filterwarnings('ignore')

# Force CPU to avoid memory issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Page configuration
st.set_page_config(
    page_title="GA Hyperparameter Tuning",
    page_icon="🧬",
    layout="wide"
)

# Title
st.title("🧬 Genetic Algorithm for Neural Network Hyperparameter Optimization")
st.markdown("**ISGA - Metaheuristics Course Project**")
st.markdown("---")

# ============================================================
# SIMPLIFIED GA CLASS (MEMORY-SAFE)
# ============================================================

class SimpleGA:
    def __init__(self, population_size=8, generations=5):
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.history = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        
        self.search_space = {
            'n_layers': [1, 2, 3],
            'layer_sizes': [32, 64, 128],
            'learning_rate': (0.001, 0.01),
            'batch_size': [32, 64],
            'dropout': (0.0, 0.3),
            'optimizer': ['adam', 'sgd'],
            'activation': ['relu', 'tanh']
        }
    
    def create_random_chromosome(self):
        n_layers = random.choice(self.search_space['n_layers'])
        return {
            'n_layers': n_layers,
            'layer_sizes': [random.choice(self.search_space['layer_sizes']) 
                           for _ in range(n_layers)],
            'learning_rate': random.uniform(*self.search_space['learning_rate']),
            'batch_size': random.choice(self.search_space['batch_size']),
            'dropout': random.uniform(*self.search_space['dropout']),
            'optimizer': random.choice(self.search_space['optimizer']),
            'activation': random.choice(self.search_space['activation'])
        }
    
    def initialize_population(self):
        self.population = [self.create_random_chromosome() 
                          for _ in range(self.population_size)]
    
    def tournament_selection(self, fitness_scores):
        indices = random.sample(range(len(self.population)), 3)
        best_idx = max(indices, key=lambda idx: fitness_scores[idx])
        return deepcopy(self.population[best_idx])
    
    def crossover(self, parent1, parent2):
        child = {}
        for key in parent1.keys():
            child[key] = deepcopy(parent1[key] if random.random() < 0.5 else parent2[key])
        
        n_layers = child['n_layers']
        if len(child['layer_sizes']) != n_layers:
            if len(child['layer_sizes']) < n_layers:
                while len(child['layer_sizes']) < n_layers:
                    child['layer_sizes'].append(random.choice(self.search_space['layer_sizes']))
            else:
                child['layer_sizes'] = child['layer_sizes'][:n_layers]
        
        return child
    
    def mutate(self, chromosome):
        mutated = deepcopy(chromosome)
        if random.random() < 0.3:
            key = random.choice(list(mutated.keys()))
            if key == 'learning_rate':
                mutated['learning_rate'] = random.uniform(*self.search_space['learning_rate'])
            elif key == 'dropout':
                mutated['dropout'] = random.uniform(*self.search_space['dropout'])
            elif key == 'batch_size':
                mutated['batch_size'] = random.choice(self.search_space['batch_size'])
        return mutated
    
    def evolve(self, fitness_function, callback=None):
        self.initialize_population()
        
        for generation in range(self.generations):
            fitness_scores = []
            
            for idx, chromosome in enumerate(self.population):
                result = fitness_function(chromosome, generation + 1, idx + 1)
                fitness = result['fitness']
                fitness_scores.append(fitness)
                
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = deepcopy(chromosome)
            
            best_fit = max(fitness_scores)
            avg_fit = np.mean(fitness_scores)
            
            gen_summary = {
                'generation': generation + 1,
                'best_fitness': float(best_fit),
                'avg_fitness': float(avg_fit),
                'best_chromosome': self.population[np.argmax(fitness_scores)]
            }
            
            self.history.append(gen_summary)
            
            if callback:
                callback(gen_summary)
            
            if generation < self.generations - 1:
                new_population = []
                
                # Elitism
                elite_indices = np.argsort(fitness_scores)[-2:]
                for idx in elite_indices:
                    new_population.append(deepcopy(self.population[idx]))
                
                while len(new_population) < self.population_size:
                    parent1 = self.tournament_selection(fitness_scores)
                    parent2 = self.tournament_selection(fitness_scores)
                    
                    child = self.crossover(parent1, parent2) if random.random() < 0.8 else deepcopy(parent1)
                    child = self.mutate(child) if random.random() < 0.3 else child
                    
                    new_population.append(child)
                
                self.population = new_population
        
        return self.best_individual, self.best_fitness

# ============================================================
# NEURAL NETWORK TRAINER (MEMORY-SAFE)
# ============================================================

def train_model(config, X_train, y_train, X_val, y_val, generation=0, individual=0):
    """Train model with memory management"""
    
    keras.backend.clear_session()
    gc.collect()
    
    try:
        model = models.Sequential([layers.Flatten(input_shape=(28, 28))])
        
        for i in range(config['n_layers']):
            model.add(layers.Dense(config['layer_sizes'][i], activation=config['activation']))
            if config['dropout'] > 0:
                model.add(layers.Dropout(config['dropout']))
        
        model.add(layers.Dense(10, activation='softmax'))
        
        if config['optimizer'] == 'adam':
            opt = optimizers.Adam(learning_rate=config['learning_rate'])
        else:
            opt = optimizers.SGD(learning_rate=config['learning_rate'], momentum=0.9)
        
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            batch_size=config['batch_size'],
            epochs=5,  # Quick for demo
            validation_data=(X_val, y_val),
            verbose=0
        )
        train_time = time.time() - start_time
        
        val_acc = float(history.history['val_accuracy'][-1])
        fitness = val_acc
        
        result = {
            'fitness': float(fitness),
            'accuracy': float(val_acc),
            'training_time': float(train_time),
            'n_parameters': int(model.count_params())
        }
        
        del model, history
        keras.backend.clear_session()
        gc.collect()
        
        return result
        
    except Exception as e:
        keras.backend.clear_session()
        gc.collect()
        
        return {
            'fitness': 0.0,
            'accuracy': 0.0,
            'training_time': 0.0,
            'n_parameters': 0
        }

# ============================================================
# STREAMLIT APP
# ============================================================

# Sidebar - Configuration
st.sidebar.header("⚙️ GA Configuration")

population_size = st.sidebar.slider(
    "Population Size", 
    min_value=5, 
    max_value=15, 
    value=8, 
    step=1,
    help="Smaller population for faster demo"
)

generations = st.sidebar.slider(
    "Number of Generations", 
    min_value=3, 
    max_value=10, 
    value=5, 
    step=1,
    help="Fewer generations for quick demo"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Note")
st.sidebar.info(
    "This demo uses smaller settings for speed. "
    "Real optimization would use more individuals and generations."
)

# Cache data loading
@st.cache_data
def load_data():
    """Load and preprocess Fashion-MNIST"""
    (X_train_full, y_train_full), _ = keras.datasets.fashion_mnist.load_data()
    
    X_train_full = X_train_full.astype('float32') / 255.0
    
    # VERY SMALL subset for interactive demo
    X_train = X_train_full[:2000]
    y_train = y_train_full[:2000]
    X_val = X_train_full[2000:2400]
    y_val = y_train_full[2000:2400]
    
    y_train = keras.utils.to_categorical(y_train, 10)
    y_val = keras.utils.to_categorical(y_val, 10)
    
    return X_train, y_train, X_val, y_val

# Main content
tab1, tab2, tab3 = st.tabs(["🚀 Run Optimization", "📊 Results", "ℹ️ About"])

with tab1:
    st.header("Run Genetic Algorithm")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuration")
        st.info(f"**Population**: {population_size} individuals")
        st.info(f"**Generations**: {generations}")
        st.info(f"**Total Evaluations**: {population_size * generations}")
    
    with col2:
        st.subheader("Search Space")
        st.markdown("""
        **Optimizing:**
        - Layers (1-3)
        - Layer sizes (32, 64, 128)
        - Learning rate (0.001-0.01)
        - Batch size (32, 64)
        - Dropout (0.0-0.3)
        - Optimizer (Adam, SGD)
        - Activation (ReLU, Tanh)
        """)
    
    st.markdown("---")
    
    if st.button("🚀 Start Optimization", type="primary", use_container_width=True):
        # Load data
        with st.spinner("Loading Fashion-MNIST dataset..."):
            X_train, y_train, X_val, y_val = load_data()
        
        st.success(f"✓ Dataset: {len(X_train)} train, {len(X_val)} val samples")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()
        
        generation_data = {
            'generation': [],
            'best_fitness': [],
            'avg_fitness': []
        }
        
        # Callback for live updates
        def progress_callback(gen_summary):
            gen_num = gen_summary['generation']
            progress = gen_num / generations
            progress_bar.progress(progress)
            status_text.text(f"Generation {gen_num}/{generations} - "
                           f"Best: {gen_summary['best_fitness']:.4f}, "
                           f"Avg: {gen_summary['avg_fitness']:.4f}")
            
            generation_data['generation'].append(gen_num)
            generation_data['best_fitness'].append(gen_summary['best_fitness'])
            generation_data['avg_fitness'].append(gen_summary['avg_fitness'])
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(generation_data['generation'], generation_data['best_fitness'], 
                   'o-', label='Best Fitness', linewidth=2, markersize=8)
            ax.plot(generation_data['generation'], generation_data['avg_fitness'], 
                   's-', label='Avg Fitness', linewidth=2, markersize=6)
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness (Accuracy)')
            ax.set_title('Live Evolution Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
            chart_placeholder.pyplot(fig)
            plt.close()
        
        # Initialize GA
        ga = SimpleGA(population_size=population_size, generations=generations)
        
        # Fitness function
        def fitness_func(chromosome, gen, ind):
            return train_model(chromosome, X_train, y_train, X_val, y_val, gen, ind)
        
        # Run GA
        start_time = time.time()
        best_chromosome, best_fitness = ga.evolve(fitness_func, callback=progress_callback)
        elapsed_time = time.time() - start_time
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Complete! Time: {elapsed_time:.1f}s, Best Fitness: {best_fitness:.4f}")
        
        st.success("🎉 Optimization completed successfully!")
        
        # Store in session
        st.session_state['ga_complete'] = True
        st.session_state['best_chromosome'] = best_chromosome
        st.session_state['best_fitness'] = best_fitness
        st.session_state['ga_history'] = ga.history
        st.session_state['elapsed_time'] = elapsed_time
        
        # Display results
        st.markdown("---")
        st.subheader("🏆 Best Solution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Best Hyperparameters:**")
            for key, value in best_chromosome.items():
                st.text(f"{key}: {value}")
        
        with col2:
            st.metric("Best Fitness", f"{best_fitness:.4f}")
            st.metric("Time Elapsed", f"{elapsed_time:.1f}s")
            st.metric("Evaluations", f"{population_size * generations}")

with tab2:
    st.header("Results & Analysis")
    
    if 'ga_complete' in st.session_state and st.session_state['ga_complete']:
        history = st.session_state['ga_history']
        best_chromosome = st.session_state['best_chromosome']
        best_fitness = st.session_state['best_fitness']
        
        st.subheader("📈 Evolution Statistics")
        
        generations_list = [h['generation'] for h in history]
        best_fitnesses = [h['best_fitness'] for h in history]
        avg_fitnesses = [h['avg_fitness'] for h in history]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Initial Best", f"{best_fitnesses[0]:.4f}")
        with col2:
            st.metric("Final Best", f"{best_fitnesses[-1]:.4f}")
        with col3:
            improvement = best_fitnesses[-1] - best_fitnesses[0]
            st.metric("Improvement", f"{improvement:.4f}")
        
        # Visualization
        st.markdown("---")
        st.subheader("📊 Fitness Evolution")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(generations_list, best_fitnesses, 'o-', label='Best', linewidth=2, markersize=8)
        ax.plot(generations_list, avg_fitnesses, 's-', label='Average', linewidth=2, markersize=6)
        ax.fill_between(generations_list, avg_fitnesses, best_fitnesses, alpha=0.2)
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel('Fitness (Accuracy)', fontsize=12)
        ax.set_title('Fitness Evolution Over Generations', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Download
        st.markdown("---")
        st.subheader("💾 Download Results")
        
        import json
        results_json = json.dumps({
            'best_chromosome': best_chromosome,
            'best_fitness': float(best_fitness),
            'history': history
        }, indent=2)
        
        st.download_button(
            label="Download Results (JSON)",
            data=results_json,
            file_name="ga_results.json",
            mime="application/json"
        )
    else:
        st.info("👈 Run the optimization first!")

with tab3:
    st.header("About This Project")
    
    st.markdown("""
    ### 🎯 Overview
    
    This interactive demo shows how **Genetic Algorithms** can automatically 
    optimize neural network hyperparameters.
    
    ### 🧬 How It Works
    
    1. **Initialize** random hyperparameter configurations
    2. **Evaluate** by training neural networks
    3. **Select** best performers as parents
    4. **Crossover** to create offspring
    5. **Mutate** for exploration
    6. **Repeat** for multiple generations
    
    ### 📊 Dataset
    
    **Fashion-MNIST**: Clothing images (28×28 pixels, 10 categories)
    
    ### 👤 Author
    
    **Lojaine** - ISGA Marrakech  
    AI & Big Data Engineering - Final Year  
    Metaheuristics Course Project 2024-2025
    
    ### 💡 Note
    
    This demo uses reduced settings for speed. Production use would employ:
    - Larger dataset (60,000 samples)
    - More individuals (20-30)
    - More generations (15-20)
    - More training epochs (10-20)
    """)

# Footer
st.markdown("---")
st.markdown(
    "<center>🧬 GA Hyperparameter Optimization | ISGA 2024-2025</center>",
    unsafe_allow_html=True
)
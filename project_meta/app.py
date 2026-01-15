"""
Interactive Streamlit App for GA Hyperparameter Optimization Demo
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import time
import matplotlib.pyplot as plt

from genetic_algorithm import GeneticAlgorithm, print_chromosome
from neural_network import NeuralNetworkBuilder
from visualization import EvolutionVisualizer


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

# Sidebar - Configuration
st.sidebar.header("⚙️ GA Configuration")

population_size = st.sidebar.slider(
    "Population Size", 
    min_value=10, 
    max_value=50, 
    value=20, 
    step=5,
    help="Number of individuals in each generation"
)

generations = st.sidebar.slider(
    "Number of Generations", 
    min_value=5, 
    max_value=30, 
    value=10, 
    step=5,
    help="Number of evolution iterations"
)

crossover_rate = st.sidebar.slider(
    "Crossover Rate", 
    min_value=0.5, 
    max_value=1.0, 
    value=0.8, 
    step=0.1,
    help="Probability of crossover between parents"
)

mutation_rate = st.sidebar.slider(
    "Mutation Rate", 
    min_value=0.1, 
    max_value=0.5, 
    value=0.2, 
    step=0.05,
    help="Probability of mutation"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app demonstrates using Genetic Algorithms to automatically "
    "find optimal hyperparameters for a neural network trained on Fashion-MNIST."
)


# Cache data loading
@st.cache_data
def load_data():
    """Load and preprocess Fashion-MNIST"""
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Use smaller subset for faster demo
    X_train_full = X_train_full[:10000]
    y_train_full = y_train_full[:10000]
    
    split_idx = int(0.8 * len(X_train_full))
    X_train = X_train_full[:split_idx]
    y_train = y_train_full[:split_idx]
    X_val = X_train_full[split_idx:]
    y_val = y_train_full[split_idx:]
    
    y_train = keras.utils.to_categorical(y_train, 10)
    y_val = keras.utils.to_categorical(y_val, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# Main content
tab1, tab2, tab3 = st.tabs(["📊 Run Optimization", "📈 Results", "ℹ️ About"])

with tab1:
    st.header("Run Genetic Algorithm")
    
    # Show current configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Configuration")
        config_df = pd.DataFrame({
            'Parameter': ['Population Size', 'Generations', 'Crossover Rate', 'Mutation Rate'],
            'Value': [population_size, generations, f"{crossover_rate:.1f}", f"{mutation_rate:.2f}"]
        })
        st.table(config_df)
    
    with col2:
        st.subheader("Search Space")
        st.markdown("""
        **Hyperparameters to optimize:**
        - Number of layers (1-4)
        - Neurons per layer (16, 32, 64, 128, 256)
        - Learning rate (0.0001 - 0.1)
        - Batch size (16, 32, 64, 128)
        - Dropout rate (0.0 - 0.5)
        - Optimizer (Adam, SGD, RMSprop)
        - Activation (ReLU, Tanh, Sigmoid)
        """)
    
    st.markdown("---")
    
    # Run button
    if st.button("🚀 Start Optimization", type="primary", use_container_width=True):
        # Load data
        with st.spinner("Loading Fashion-MNIST dataset..."):
            X_train, y_train, X_val, y_val, X_test, y_test = load_data()
        
        st.success(f"✓ Dataset loaded: {len(X_train)} training samples, {len(X_val)} validation samples")
        
        # Initialize components
        nn_builder = NeuralNetworkBuilder(
            input_shape=(28, 28),
            num_classes=10
        )
        
        def fitness_function(chromosome):
            return nn_builder.evaluate_fitness(
                chromosome, X_train, y_train, X_val, y_val,
                epochs=3, verbose=0  # Quick training for demo
            )
        
        # Initialize GA
        ga = GeneticAlgorithm(
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism_count=2,
            tournament_size=3
        )
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Placeholder for live updates
        chart_placeholder = st.empty()
        
        generation_data = {
            'generation': [],
            'best_fitness': [],
            'avg_fitness': []
        }
        
        # Callback for progress updates
        def progress_callback(gen_summary):
            gen_num = gen_summary['generation']
            progress = gen_num / generations
            progress_bar.progress(progress)
            status_text.text(f"Generation {gen_num}/{generations} - "
                           f"Best Fitness: {gen_summary['best_fitness']:.4f}")
            
            # Update live chart
            generation_data['generation'].append(gen_num)
            generation_data['best_fitness'].append(gen_summary['best_fitness'])
            generation_data['avg_fitness'].append(gen_summary['avg_fitness'])
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(generation_data['generation'], generation_data['best_fitness'], 
                   marker='o', label='Best Fitness', linewidth=2)
            ax.plot(generation_data['generation'], generation_data['avg_fitness'], 
                   marker='s', label='Avg Fitness', linewidth=2)
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.set_title('Evolution Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
            chart_placeholder.pyplot(fig)
            plt.close()
        
        # Run GA
        with st.spinner("Running Genetic Algorithm..."):
            start_time = time.time()
            best_chromosome, best_fitness = ga.evolve(fitness_function, callback=progress_callback)
            elapsed_time = time.time() - start_time
        
        # Complete
        progress_bar.progress(1.0)
        status_text.text(f"✅ Optimization Complete! Time: {elapsed_time:.1f}s")
        
        st.success("Genetic Algorithm completed successfully!")
        
        # Store results in session state
        st.session_state['ga_complete'] = True
        st.session_state['best_chromosome'] = best_chromosome
        st.session_state['best_fitness'] = best_fitness
        st.session_state['ga_history'] = ga.history
        st.session_state['elapsed_time'] = elapsed_time
        
        # Display best solution
        st.markdown("---")
        st.subheader("🏆 Best Solution Found")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Best Hyperparameters:**")
            best_df = pd.DataFrame({
                'Hyperparameter': list(best_chromosome.keys()),
                'Value': [str(v) for v in best_chromosome.values()]
            })
            st.table(best_df)
        
        with col2:
            st.metric("Best Fitness", f"{best_fitness:.4f}")
            st.metric("Time Elapsed", f"{elapsed_time:.1f}s")
            st.metric("Total Evaluations", f"{population_size * generations}")

with tab2:
    st.header("Results & Analysis")
    
    if 'ga_complete' in st.session_state and st.session_state['ga_complete']:
        # Get data from session state
        history = st.session_state['ga_history']
        best_chromosome = st.session_state['best_chromosome']
        best_fitness = st.session_state['best_fitness']
        
        # Statistics
        st.subheader("📊 Evolution Statistics")
        
        generations_list = [h['generation'] for h in history]
        best_fitnesses = [h['best_fitness'] for h in history]
        avg_fitnesses = [h['avg_fitness'] for h in history]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Initial Best", f"{best_fitnesses[0]:.4f}")
        with col2:
            st.metric("Final Best", f"{best_fitnesses[-1]:.4f}")
        with col3:
            improvement = best_fitnesses[-1] - best_fitnesses[0]
            st.metric("Improvement", f"{improvement:.4f}")
        with col4:
            convergence_gen = best_fitnesses.index(max(best_fitnesses)) + 1
            st.metric("Converged at Gen", convergence_gen)
        
        # Visualizations
        st.markdown("---")
        st.subheader("📈 Visualizations")
        
        visualizer = EvolutionVisualizer()
        
        # Fitness evolution
        st.markdown("**Fitness Evolution Over Generations**")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(generations_list, best_fitnesses, marker='o', label='Best', linewidth=2)
        ax1.plot(generations_list, avg_fitnesses, marker='s', label='Average', linewidth=2)
        ax1.fill_between(generations_list, avg_fitnesses, best_fitnesses, alpha=0.2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        
        # Parameter evolution
        st.markdown("**Hyperparameter Evolution**")
        learning_rates = [h['best_chromosome']['learning_rate'] for h in history]
        dropouts = [h['best_chromosome']['dropout'] for h in history]
        
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax2.plot(generations_list, learning_rates, marker='o', color='blue', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Evolution')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(generations_list, dropouts, marker='s', color='green', linewidth=2)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Dropout Rate')
        ax3.set_title('Dropout Evolution')
        ax3.grid(True, alpha=0.3)
        
        st.pyplot(fig2)
        
        # Download results
        st.markdown("---")
        st.subheader("💾 Download Results")
        
        import json
        
        results_json = json.dumps({
            'best_chromosome': best_chromosome,
            'best_fitness': float(best_fitness),
            'evolution_history': history
        }, indent=2)
        
        st.download_button(
            label="Download Results (JSON)",
            data=results_json,
            file_name="ga_results.json",
            mime="application/json"
        )
        
    else:
        st.info("👈 Run the optimization first to see results!")

with tab3:
    st.header("About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This project demonstrates the use of **Genetic Algorithms** (a metaheuristic optimization 
    technique) to automatically find optimal hyperparameters for neural networks.
    
    ### 🧬 How It Works
    
    1. **Initialization**: Create a population of random hyperparameter configurations
    2. **Evaluation**: Train neural networks with each configuration and measure performance
    3. **Selection**: Select best-performing configurations as parents
    4. **Crossover**: Combine parent configurations to create offspring
    5. **Mutation**: Randomly modify some configurations to explore new possibilities
    6. **Iteration**: Repeat for multiple generations
    
    ### 📊 Dataset
    
    **Fashion-MNIST**: 70,000 grayscale images of clothing items in 10 categories
    - Training: 60,000 images
    - Testing: 10,000 images
    - Image size: 28×28 pixels
    
    ### 🎓 Educational Value
    
    This project combines:
    - **Metaheuristic Algorithms** (Genetic Algorithm)
    - **Machine Learning** (Neural Networks)
    - **Optimization Theory** (Hyperparameter Tuning)
    
    ### 👤 Author
    
    **Lojaine**  
    ISGA - Final Year AI and Big Data Engineering Student  
    Metaheuristics Course Project
    
    ### 📚 References
    
    - Genetic Algorithms in Search, Optimization and Machine Learning (Goldberg)
    - Fashion-MNIST Dataset (Zalando Research)
    - TensorFlow/Keras Documentation
    """)
    
    st.markdown("---")
    st.markdown("*Built with Python, TensorFlow, and Streamlit*")


# Footer
st.markdown("---")
st.markdown(
    "<center>🧬 Genetic Algorithm Hyperparameter Optimization | "
    "ISGA Metaheuristics Project 2024-2025</center>",
    unsafe_allow_html=True
)

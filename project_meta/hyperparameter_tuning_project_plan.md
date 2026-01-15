# Machine Learning Hyperparameter Tuning with Genetic Algorithm
## Project Plan for ISGA Metaheuristics Exam

---

## 1. PROJECT OVERVIEW

### Title
**Optimisation des Hyperparamètres d'un Réseau de Neurones par Algorithme Génétique**

### Objective
Develop a Genetic Algorithm to automatically find optimal hyperparameters for a neural network, demonstrating how metaheuristics can solve complex ML optimization problems.

### Why This Project?
- Combines AI/ML expertise with metaheuristic algorithms
- Solves a real practical problem in ML (hyperparameter tuning is time-consuming)
- Shows clear improvement over random/grid search
- Excellent visualization opportunities
- Directly applicable to your previous ML projects

---

## 2. PROBLEM DEFINITION

### Hyperparameters to Optimize
For a Neural Network classifier:
1. **Number of hidden layers** (1-4)
2. **Neurons per layer** (16, 32, 64, 128, 256)
3. **Learning rate** (0.0001 - 0.1)
4. **Batch size** (16, 32, 64, 128)
5. **Dropout rate** (0.0 - 0.5)
6. **Optimizer** (Adam, SGD, RMSprop)
7. **Activation function** (ReLU, Tanh, Sigmoid)

### Fitness Function
```
Fitness = α × Accuracy + β × (1 - Training_Time/Max_Time) + γ × (1 - Model_Complexity/Max_Complexity)

Where:
- α = 0.7 (weight for accuracy - most important)
- β = 0.2 (weight for training speed)
- γ = 0.1 (weight for model simplicity)
```

### Dataset Options (Choose ONE)
1. **MNIST** - Digit classification (simple, fast to train)
2. **Fashion-MNIST** - Clothing classification (moderate complexity)
3. **Iris/Wine Quality** - Tabular data (very fast, good for testing)
4. **CIFAR-10** - Image classification (more challenging)
5. **Your own**: Banking fraud detection from your previous work

**Recommendation**: Start with Fashion-MNIST (good balance of complexity and training time)

---

## 3. GENETIC ALGORITHM DESIGN

### Chromosome Encoding
```python
chromosome = {
    'n_layers': int (1-4),
    'layer_sizes': list[int] (e.g., [128, 64, 32]),
    'learning_rate': float (0.0001-0.1),
    'batch_size': int (16, 32, 64, 128),
    'dropout': float (0.0-0.5),
    'optimizer': str ('adam', 'sgd', 'rmsprop'),
    'activation': str ('relu', 'tanh', 'sigmoid')
}
```

### GA Parameters
- **Population size**: 20 individuals
- **Generations**: 15-20 generations
- **Selection**: Tournament selection (k=3)
- **Crossover rate**: 0.8 (80%)
- **Mutation rate**: 0.2 (20%)
- **Elitism**: Keep top 2 individuals

### Genetic Operators

#### 1. Selection (Tournament)
```python
def tournament_selection(population, fitness_scores, k=3):
    # Select k random individuals
    # Return the best one
```

#### 2. Crossover (Uniform)
```python
def crossover(parent1, parent2):
    # For each gene, randomly choose from parent1 or parent2
    # Example: 
    # - n_layers from parent1
    # - learning_rate from parent2
    # - etc.
```

#### 3. Mutation
```python
def mutation(chromosome, mutation_rate=0.2):
    # For each gene, with probability mutation_rate:
    #   - Randomly change to a valid value
    # Examples:
    #   - n_layers: randomly change to 1, 2, 3, or 4
    #   - learning_rate: add/subtract small random value
    #   - optimizer: randomly select different optimizer
```

---

## 4. IMPLEMENTATION STRUCTURE

### File Organization
```
hyperparameter_tuning/
│
├── main.py                      # Main execution script
├── genetic_algorithm.py         # GA implementation
├── neural_network.py            # NN model builder
├── fitness_evaluator.py         # Fitness function
├── visualization.py             # All visualizations
├── utils.py                     # Helper functions
│
├── data/
│   └── dataset.npz             # Your chosen dataset
│
├── results/
│   ├── best_individuals.json   # Best configs per generation
│   ├── evolution_history.csv   # Complete GA history
│   └── final_best_model.h5     # Best trained model
│
└── plots/
    ├── fitness_evolution.png
    ├── parameter_evolution.png
    ├── comparison_chart.png
    └── confusion_matrix.png
```

---

## 5. DETAILED IMPLEMENTATION STEPS

### Step 1: Data Preparation (Day 1)
```python
# Load and preprocess dataset
- Download Fashion-MNIST
- Normalize images (0-1)
- Split: 80% train, 20% validation
- Create data loaders
```

### Step 2: Neural Network Builder (Day 1-2)
```python
def build_model(chromosome):
    """
    Build a Keras model based on chromosome
    """
    model = Sequential()
    
    # Input layer
    model.add(Flatten(input_shape=(28, 28)))
    
    # Hidden layers based on chromosome
    for i in range(chromosome['n_layers']):
        model.add(Dense(chromosome['layer_sizes'][i], 
                       activation=chromosome['activation']))
        model.add(Dropout(chromosome['dropout']))
    
    # Output layer
    model.add(Dense(10, activation='softmax'))
    
    # Compile
    model.compile(
        optimizer=get_optimizer(chromosome),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

### Step 3: Fitness Evaluator (Day 2)
```python
def evaluate_fitness(chromosome, X_train, y_train, X_val, y_val):
    """
    Train model and return fitness score
    """
    # Build model
    model = build_model(chromosome)
    
    # Train with time tracking
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        batch_size=chromosome['batch_size'],
        epochs=5,  # Quick training for GA
        validation_data=(X_val, y_val),
        verbose=0
    )
    training_time = time.time() - start_time
    
    # Get metrics
    accuracy = history.history['val_accuracy'][-1]
    model_complexity = count_parameters(model)
    
    # Calculate fitness
    fitness = (
        0.7 * accuracy + 
        0.2 * (1 - training_time/MAX_TIME) + 
        0.1 * (1 - model_complexity/MAX_COMPLEXITY)
    )
    
    return fitness, accuracy, training_time, model_complexity
```

### Step 4: Genetic Algorithm Core (Day 3-4)
```python
class GeneticAlgorithm:
    def __init__(self, population_size=20, generations=15):
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.history = []
    
    def initialize_population(self):
        """Create random initial population"""
        for _ in range(self.population_size):
            chromosome = {
                'n_layers': random.randint(1, 4),
                'layer_sizes': [random.choice([16,32,64,128,256]) 
                               for _ in range(random.randint(1, 4))],
                'learning_rate': random.uniform(0.0001, 0.1),
                'batch_size': random.choice([16, 32, 64, 128]),
                'dropout': random.uniform(0.0, 0.5),
                'optimizer': random.choice(['adam', 'sgd', 'rmsprop']),
                'activation': random.choice(['relu', 'tanh', 'sigmoid'])
            }
            self.population.append(chromosome)
    
    def evolve(self, X_train, y_train, X_val, y_val):
        """Main GA loop"""
        self.initialize_population()
        
        for generation in range(self.generations):
            print(f"\n=== Generation {generation + 1}/{self.generations} ===")
            
            # Evaluate fitness
            fitness_scores = []
            for idx, chromosome in enumerate(self.population):
                fitness, acc, time, complexity = evaluate_fitness(
                    chromosome, X_train, y_train, X_val, y_val
                )
                fitness_scores.append(fitness)
                print(f"Individual {idx+1}: Fitness={fitness:.4f}, Acc={acc:.4f}")
            
            # Save history
            self.history.append({
                'generation': generation,
                'best_fitness': max(fitness_scores),
                'avg_fitness': np.mean(fitness_scores),
                'best_chromosome': self.population[np.argmax(fitness_scores)]
            })
            
            # Selection + Crossover + Mutation
            new_population = self.elitism(fitness_scores, n=2)
            
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(fitness_scores)
                parent2 = self.tournament_selection(fitness_scores)
                
                if random.random() < 0.8:  # Crossover rate
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                if random.random() < 0.2:  # Mutation rate
                    child = self.mutate(child)
                
                new_population.append(child)
            
            self.population = new_population
        
        return self.get_best_solution(fitness_scores)
```

### Step 5: Visualization (Day 4-5)
Create comprehensive visualizations:

1. **Fitness Evolution Over Generations**
   - Line plot: Best fitness, Average fitness, Worst fitness
   
2. **Parameter Evolution Heatmap**
   - Show how each parameter changes across generations
   
3. **Population Diversity**
   - Show how diverse the population remains
   
4. **Comparison with Baseline**
   - Compare GA results vs Random Search vs Grid Search
   
5. **Final Model Performance**
   - Confusion matrix
   - ROC curve (if applicable)
   - Training/validation curves

### Step 6: User Interface (Day 5)
```python
# Simple Streamlit interface for demo
import streamlit as st

st.title("Neural Network Hyperparameter Optimization with GA")

# Sidebar for GA parameters
st.sidebar.header("GA Configuration")
population_size = st.sidebar.slider("Population Size", 10, 50, 20)
generations = st.sidebar.slider("Generations", 5, 30, 15)

# Run button
if st.button("Run Optimization"):
    # Run GA
    # Show real-time progress
    # Display results
```

---

## 6. EXPECTED RESULTS

### Metrics to Report
1. **Best hyperparameters found**
2. **Final model accuracy** (should be >90% on Fashion-MNIST)
3. **Convergence speed** (how many generations to reach near-optimal)
4. **Time comparison** (GA vs Grid Search vs Random Search)
5. **Model complexity** (number of parameters)

### Baseline Comparisons
Compare your GA against:
- **Random Search**: 20 random configurations
- **Grid Search**: Systematic search (if feasible)
- **Default Config**: Basic NN with standard hyperparameters

**Expected**: GA should find better solutions faster than random/grid search

---

## 7. PRESENTATION STRUCTURE (15-20 minutes)

### Slide Outline
1. **Introduction** (2 min)
   - Problem: Why hyperparameter tuning matters
   - Challenge: Huge search space (show calculation)

2. **Methodology** (3 min)
   - Genetic Algorithm principles
   - Chromosome encoding
   - Genetic operators (selection, crossover, mutation)

3. **Implementation** (3 min)
   - Dataset choice and preprocessing
   - Fitness function design
   - GA parameters

4. **Live Demo** (4 min)
   - Show the interface
   - Run 2-3 generations live (if fast enough)
   - OR show pre-recorded video

5. **Results** (5 min)
   - Fitness evolution graphs
   - Best hyperparameters found
   - Model performance metrics
   - Comparison with baselines

6. **Conclusion** (2 min)
   - GA effectiveness for hyperparameter optimization
   - Future improvements (parallel evaluation, adaptive mutation)

7. **Q&A** (2-3 min)

---

## 8. ADVANCED FEATURES (Optional - If Time Permits)

### 1. Parallel Fitness Evaluation
```python
from multiprocessing import Pool

def parallel_evaluate(population, data):
    with Pool(processes=4) as pool:
        results = pool.starmap(evaluate_fitness, 
                              [(ind, data) for ind in population])
    return results
```

### 2. Adaptive Mutation Rate
```python
# Increase mutation when population converges
if diversity < threshold:
    mutation_rate *= 1.2
```

### 3. Multi-Objective Optimization
```python
# Pareto front for accuracy vs model size vs speed
fitness = pareto_rank(accuracy, speed, complexity)
```

### 4. Transfer Learning
```python
# Use best individuals from previous runs as starting population
```

---

## 9. TIMELINE (5-7 Days)

**Day 1**: Setup + Data Preparation + NN Builder
**Day 2**: Fitness Function + Initial Testing
**Day 3**: GA Core Implementation
**Day 4**: Complete GA + Basic Visualization
**Day 5**: Advanced Visualization + Interface
**Day 6**: Testing + Comparison with Baselines
**Day 7**: Presentation Preparation + Slides

---

## 10. TROUBLESHOOTING

### Common Issues

**1. GA takes too long**
- Reduce epochs per training (5 epochs max)
- Use smaller validation set (10% instead of 20%)
- Reduce population size to 15
- Use GPU if available

**2. GA not converging**
- Check fitness function (is it rewarding the right thing?)
- Increase mutation rate (0.3 instead of 0.2)
- Verify crossover is producing valid offspring
- Check chromosome bounds

**3. All individuals become similar (premature convergence)**
- Increase mutation rate
- Use diversity-preserving selection
- Add crowding distance to fitness

**4. Poor model accuracy**
- Verify data preprocessing
- Check if epochs are sufficient
- Validate fitness function calculation
- Ensure optimizer is properly configured

---

## 11. GRADING CRITERIA ALIGNMENT

This project covers:
✅ **Algorithm Implementation** - Complete GA with all operators
✅ **Problem Solving** - Real ML optimization problem
✅ **Simulation** - Neural network training in each generation
✅ **Visualization** - Multiple comprehensive plots
✅ **Interface** - Interactive Streamlit app
✅ **Results Analysis** - Comparison with baselines
✅ **Presentation** - Clear explanation of methodology and results

---

## 12. SUCCESS CRITERIA

Your project is successful if:
- ✅ GA finds hyperparameters achieving >85% accuracy
- ✅ Fitness improves over generations (clear upward trend)
- ✅ Results are better than random search
- ✅ All visualizations clearly show GA behavior
- ✅ Code runs without errors
- ✅ Presentation is clear and well-structured

---

## NEXT STEPS

1. Confirm dataset choice (Fashion-MNIST recommended)
2. Set up development environment
3. Start with Step 1 (Data Preparation)
4. Test each component before integrating
5. Create visualizations early (good for debugging)

**Ready to start coding?** Let me know if you want me to generate:
- The complete Python implementation
- Specific code for any component
- Visualization templates
- Presentation slides outline

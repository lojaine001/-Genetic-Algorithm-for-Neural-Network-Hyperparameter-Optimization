# 🧬 Genetic Algorithm for Neural Network Hyperparameter Optimization

**ISGA Metaheuristics Course Project**  
Author: Lojaine  
Academic Year: 2024-2025

---

## 📋 Project Overview

This project implements a **Genetic Algorithm (GA)** to automatically optimize hyperparameters for neural networks. The GA searches through a vast hyperparameter space to find configurations that maximize model performance on the Fashion-MNIST dataset.

### Key Features
- ✅ Complete GA implementation (selection, crossover, mutation, elitism)
- ✅ Automated neural network hyperparameter tuning
- ✅ Real-time visualization of evolution progress
- ✅ Comparison with baseline methods (Random Search, Default Config)
- ✅ Interactive Streamlit web interface
- ✅ Comprehensive visualizations and analysis

---

## 🎯 Problem Statement

**Challenge**: Finding optimal hyperparameters for neural networks is time-consuming and often relies on manual trial-and-error or expensive grid search.

**Solution**: Use Genetic Algorithms to intelligently explore the hyperparameter space and converge on near-optimal configurations.

**Hyperparameters Optimized**:
- Number of hidden layers (1-4)
- Neurons per layer (16, 32, 64, 128, 256)
- Learning rate (0.0001 - 0.1)
- Batch size (16, 32, 64, 128)
- Dropout rate (0.0 - 0.5)
- Optimizer (Adam, SGD, RMSprop)
- Activation function (ReLU, Tanh, Sigmoid)

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the project**
```bash
cd hyperparameter_tuning_ga
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Method 1: Run Complete Optimization (Command Line)

```bash
python main.py
```

This will:
1. Load Fashion-MNIST dataset
2. Run Genetic Algorithm optimization
3. Compare with baseline methods
4. Generate all visualizations
5. Save results to `results_TIMESTAMP/` directory

**Expected Output**:
- Training logs showing evolution progress
- Best hyperparameters found
- Comparison with Random Search and Default config
- Multiple visualization plots saved to disk

### Method 2: Interactive Web Interface

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Features**:
- Configure GA parameters via sliders
- Run optimization with live progress updates
- Interactive visualizations
- Download results as JSON

---

## 📁 Project Structure

```
hyperparameter_tuning_ga/
│
├── main.py                      # Main execution script
├── genetic_algorithm.py         # GA core implementation
├── neural_network.py            # NN builder & fitness evaluator
├── visualization.py             # Visualization functions
├── app.py                       # Streamlit web interface
│
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── hyperparameter_tuning_project_plan.md  # Detailed project plan
│
└── results_TIMESTAMP/           # Generated after running
    ├── ga_history.json          # Evolution history
    ├── comparison.json          # Baseline comparison
    └── plots/                   # All visualizations
        ├── fitness_evolution.png
        ├── parameter_evolution.png
        ├── categorical_params.png
        ├── diversity.png
        ├── comparison.png
        └── best_architecture.png
```

---

## 🧬 Genetic Algorithm Details

### Chromosome Encoding
Each individual (chromosome) represents a complete hyperparameter configuration:
```python
{
    'n_layers': 2,
    'layer_sizes': [128, 64],
    'learning_rate': 0.001,
    'batch_size': 32,
    'dropout': 0.2,
    'optimizer': 'adam',
    'activation': 'relu'
}
```

### Fitness Function
```
Fitness = 0.7 × Accuracy + 0.2 × Speed + 0.1 × Simplicity
```

Where:
- **Accuracy**: Validation accuracy (most important)
- **Speed**: Normalized training time (faster is better)
- **Simplicity**: Normalized model complexity (fewer parameters is better)

### Genetic Operators

**1. Selection**: Tournament selection (k=3)
- Select 3 random individuals
- Choose the one with best fitness

**2. Crossover**: Uniform crossover (rate=0.8)
- For each gene, randomly choose from parent1 or parent2
- Creates diverse offspring

**3. Mutation**: Adaptive mutation (rate=0.2)
- Randomly modify genes with mutation probability
- Small perturbations for continuous values
- Random selection for categorical values

**4. Elitism**: Preserve top 2 individuals
- Ensures best solutions are not lost

---

## 📊 Expected Results

### Performance Metrics
On Fashion-MNIST with 15 generations:
- **Best Accuracy**: >85% (typically 87-90%)
- **Convergence**: Usually by generation 8-12
- **Better than Random Search**: ✅ Yes
- **Better than Default Config**: ✅ Yes

### Sample Best Configuration
```python
{
    'n_layers': 3,
    'layer_sizes': [128, 64, 32],
    'learning_rate': 0.00234,
    'batch_size': 64,
    'dropout': 0.245,
    'optimizer': 'adam',
    'activation': 'relu'
}
```

---

## 📈 Visualizations Generated

1. **Fitness Evolution**
   - Shows best, average, worst fitness over generations
   - Demonstrates convergence behavior

2. **Parameter Evolution**
   - Tracks how each hyperparameter changes
   - Learning rate, dropout, batch size, layers

3. **Categorical Distribution**
   - Shows optimizer and activation function distribution
   - Reveals GA preferences

4. **Population Diversity**
   - Standard deviation and coefficient of variation
   - Indicates exploration vs exploitation

5. **Method Comparison**
   - GA vs Random Search vs Default Config
   - Accuracy, fitness, time, parameters

6. **Best Architecture**
   - Visual representation of final neural network
   - Layer sizes and connections

---

## ⚙️ Configuration

### Modifying GA Parameters

Edit in `main.py`:
```python
ga = GeneticAlgorithm(
    population_size=20,     # Increase for better exploration
    generations=15,         # More generations = better convergence
    crossover_rate=0.8,     # Higher = more offspring from parents
    mutation_rate=0.2,      # Higher = more exploration
    elitism_count=2,        # Preserve best individuals
    tournament_size=3       # Selection pressure
)
```

### Modifying Hyperparameter Search Space

Edit in `genetic_algorithm.py`:
```python
self.search_space = {
    'n_layers': [1, 2, 3, 4],
    'layer_sizes': [16, 32, 64, 128, 256],
    'learning_rate': (0.0001, 0.1),
    'batch_size': [16, 32, 64, 128],
    'dropout': (0.0, 0.5),
    'optimizer': ['adam', 'sgd', 'rmsprop'],
    'activation': ['relu', 'tanh', 'sigmoid']
}
```

### Using Different Datasets

To use MNIST, CIFAR-10, or custom datasets:

1. Modify `load_fashion_mnist()` in `main.py`
2. Update `input_shape` and `num_classes` in `NeuralNetworkBuilder`

Example for MNIST:
```python
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
nn_builder = NeuralNetworkBuilder(input_shape=(28, 28), num_classes=10)
```

---

## 🎓 Educational Value

This project demonstrates:

1. **Metaheuristic Algorithms**
   - Genetic Algorithm principles
   - Population-based search
   - Balance between exploration and exploitation

2. **Machine Learning**
   - Neural network architecture
   - Hyperparameter importance
   - Training and validation

3. **Optimization Theory**
   - Multi-objective optimization
   - Fitness function design
   - Search space exploration

4. **Software Engineering**
   - Modular code design
   - Visualization
   - Interactive interfaces

---

## 🔧 Troubleshooting

### Issue: GA is too slow
**Solution**: 
- Reduce `epochs` in fitness evaluation (currently 5)
- Use smaller subset of training data
- Reduce population size or generations
- Use GPU if available: `tf.config.list_physical_devices('GPU')`

### Issue: Poor convergence
**Solution**:
- Increase mutation rate (try 0.3)
- Increase population size (try 30)
- More generations (try 20-25)
- Check fitness function weights

### Issue: Out of memory
**Solution**:
- Reduce batch size in search space
- Use smaller neural networks
- Enable memory growth: `tf.config.experimental.set_memory_growth(gpu, True)`

---

## 📚 References

1. Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*
2. Fashion-MNIST Dataset: https://github.com/zalandoresearch/fashion-mnist
3. TensorFlow Documentation: https://www.tensorflow.org/
4. Metaheuristic Optimization Course Materials - ISGA

---

## 👤 Author

**Lojaine**  
AI and Big Data Engineering Student  
ISGA Marrakech - Class of 2025

**Contact**:
- GitHub: [Your GitHub]
- LinkedIn: [Your LinkedIn]
- Email: [Your Email]

---

## 📝 License

This project is created for educational purposes as part of the ISGA Metaheuristics course.

---

## 🙏 Acknowledgments

- Professor Oulad Sayad Younes for the Metaheuristics course
- ISGA Marrakech for providing the academic framework
- TensorFlow and Keras teams for excellent ML libraries
- Fashion-MNIST creators for the dataset

---

## ⭐ Project Highlights

✅ **Complete Implementation**: All GA operators implemented from scratch  
✅ **Real Application**: Solves actual ML problem (hyperparameter tuning)  
✅ **Comprehensive Evaluation**: Compares with multiple baselines  
✅ **Rich Visualizations**: 6+ different visualization types  
✅ **Interactive Demo**: Web interface for easy demonstration  
✅ **Well-Documented**: Extensive comments and documentation  
✅ **Reproducible**: Seeds set for consistent results

---

**Status**: ✅ Complete and ready for presentation

**Last Updated**: December 2024

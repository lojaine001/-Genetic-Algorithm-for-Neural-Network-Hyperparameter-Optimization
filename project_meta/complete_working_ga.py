"""
OPTIMISATION D'HYPERPARAMÈTRES PAR ALGORITHME GÉNÉTIQUE
Version complète avec commentaires détaillés pour présentation

Ce programme utilise un algorithme génétique pour trouver automatiquement
les meilleurs hyperparamètres pour un réseau de neurones sur Fashion-MNIST.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import time
import random
from copy import deepcopy
import gc
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION INITIALE
# ============================================================

# Force l'utilisation du CPU pour éviter les problèmes de mémoire GPU
# Ceci est essentiel car nous allons entraîner beaucoup de modèles successivement
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*70)
print("🧬 ALGORITHME GÉNÉTIQUE POUR OPTIMISATION D'HYPERPARAMÈTRES")
print("="*70)

# Fixe les seeds pour la reproductibilité des résultats
# Cela permet d'obtenir les mêmes résultats à chaque exécution
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Crée un dossier pour sauvegarder tous les résultats
# Le timestamp permet d'avoir un dossier unique pour chaque exécution
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f'results_{timestamp}'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(f'{results_dir}/plots', exist_ok=True)
print(f"📁 Les résultats seront sauvegardés dans: {results_dir}/\n")

# ============================================================
# CLASSE ALGORITHME GÉNÉTIQUE
# ============================================================

class GeneticAlgorithm:
    """
    Implémentation d'un algorithme génétique pour l'optimisation d'hyperparamètres.
    
    L'algorithme génétique fonctionne en 5 étapes principales:
    1. Initialisation: Créer une population aléatoire
    2. Évaluation: Calculer le fitness de chaque individu
    3. Sélection: Choisir les meilleurs parents
    4. Croisement: Combiner les parents pour créer des enfants
    5. Mutation: Modifier légèrement certains enfants
    """
    
    def __init__(self, population_size=10, generations=8):
        """
        Initialise l'algorithme génétique.
        
        Args:
            population_size: Nombre d'individus dans chaque génération (10 individus)
            generations: Nombre de générations à exécuter (8 générations)
        
        Avec ces paramètres, nous allons entraîner 10 × 8 = 80 réseaux de neurones
        """
        self.population_size = population_size
        self.generations = generations
        self.population = []  # Liste qui contiendra tous les individus actuels
        self.history = []     # Liste qui stocke l'historique de chaque génération
        self.best_individual = None  # Meilleure solution trouvée
        self.best_fitness = -float('inf')  # Meilleur fitness trouvé
        
        # ESPACE DE RECHERCHE
        # Définit tous les hyperparamètres possibles que l'AG peut explorer
        # C'est l'équivalent du "génotype" en biologie
        self.search_space = {
            'n_layers': [1, 2, 3],  # Nombre de couches cachées (1 à 3)
            'layer_sizes': [32, 64, 128, 256],  # Taille de chaque couche (nombre de neurones)
            'learning_rate': (0.0001, 0.01),  # Taux d'apprentissage (valeur continue)
            'batch_size': [32, 64],  # Taille du batch pour l'entraînement
            'dropout': (0.0, 0.4),  # Taux de dropout pour la régularisation (valeur continue)
            'optimizer': ['adam', 'sgd', 'rmsprop'],  # Algorithme d'optimisation
            'activation': ['relu', 'tanh', 'sigmoid']  # Fonction d'activation
        }
    
    def create_random_chromosome(self):
        """
        Crée un chromosome aléatoire (un individu).
        
        Un chromosome représente une configuration complète d'hyperparamètres.
        C'est comme un "ADN" qui encode toutes les caractéristiques du réseau.
        
        Returns:
            dict: Un dictionnaire contenant tous les hyperparamètres
        """
        # Choisit d'abord le nombre de couches
        n_layers = random.choice(self.search_space['n_layers'])
        
        # Crée un chromosome avec tous les hyperparamètres
        return {
            'n_layers': n_layers,
            # Génère une liste de tailles de couches (une taille par couche)
            'layer_sizes': [random.choice(self.search_space['layer_sizes']) 
                           for _ in range(n_layers)],
            # Learning rate: valeur continue entre min et max
            'learning_rate': random.uniform(*self.search_space['learning_rate']),
            'batch_size': random.choice(self.search_space['batch_size']),
            # Dropout: valeur continue entre 0 et 0.4
            'dropout': random.uniform(*self.search_space['dropout']),
            'optimizer': random.choice(self.search_space['optimizer']),
            'activation': random.choice(self.search_space['activation'])
        }
    
    def initialize_population(self):
        """
        ÉTAPE 1: Initialisation de la population.
        
        Crée la population initiale avec des individus aléatoires.
        C'est la génération 0, le point de départ de l'évolution.
        """
        self.population = [self.create_random_chromosome() 
                          for _ in range(self.population_size)]
    
    def tournament_selection(self, fitness_scores):
        """
        ÉTAPE 3: Sélection par tournoi.
        
        Sélectionne un parent en faisant un "tournoi" entre 3 individus aléatoires.
        Le meilleur des 3 gagne et devient parent.
        
        Cette méthode donne plus de chances aux bons individus d'être sélectionnés,
        mais permet aussi à des moins bons d'avoir une chance (diversité).
        
        Args:
            fitness_scores: Liste des fitness de tous les individus
            
        Returns:
            dict: Copie du chromosome sélectionné
        """
        tournament_size = 3  # Taille du tournoi: on compare 3 individus
        # Choisit 3 individus au hasard
        indices = random.sample(range(len(self.population)), tournament_size)
        # Trouve celui qui a le meilleur fitness
        best_idx = max(indices, key=lambda idx: fitness_scores[idx])
        # Retourne une copie pour ne pas modifier l'original
        return deepcopy(self.population[best_idx])
    
    def crossover(self, parent1, parent2):
        """
        ÉTAPE 4: Croisement (crossover).
        
        Combine deux parents pour créer un enfant.
        Pour chaque hyperparamètre, on choisit aléatoirement s'il vient du parent1 ou parent2.
        C'est l'équivalent de la reproduction sexuée en biologie.
        
        Args:
            parent1: Premier parent (chromosome)
            parent2: Deuxième parent (chromosome)
            
        Returns:
            dict: Enfant créé (nouveau chromosome)
        """
        child = {}
        
        # Pour chaque hyperparamètre, choisit aléatoirement le parent
        for key in parent1.keys():
            # 50% de chance de prendre la valeur du parent1, 50% du parent2
            child[key] = deepcopy(parent1[key] if random.random() < 0.5 else parent2[key])
        
        # CAS SPÉCIAL: Ajuste layer_sizes pour correspondre à n_layers
        # Si on a hérité 2 couches mais 3 tailles, il faut corriger
        n_layers = child['n_layers']
        if len(child['layer_sizes']) != n_layers:
            if len(child['layer_sizes']) < n_layers:
                # Ajoute des couches manquantes
                while len(child['layer_sizes']) < n_layers:
                    child['layer_sizes'].append(random.choice(self.search_space['layer_sizes']))
            else:
                # Retire les couches en trop
                child['layer_sizes'] = child['layer_sizes'][:n_layers]
        
        return child
    
    def mutate(self, chromosome):
        """
        ÉTAPE 5: Mutation.
        
        Modifie aléatoirement certains gènes du chromosome.
        Cela permet d'explorer de nouvelles solutions et éviter de rester coincé
        dans un optimum local.
        
        Args:
            chromosome: Chromosome à muter
            
        Returns:
            dict: Chromosome muté
        """
        mutated = deepcopy(chromosome)
        mutation_rate = 0.3  # 30% de chance de mutation pour chaque gène
        
        # Pour chaque hyperparamètre
        for key in mutated.keys():
            # 30% de chance de muter ce gène
            if random.random() < mutation_rate:
                # Différentes stratégies selon le type d'hyperparamètre
                if key == 'n_layers':
                    # Mutation du nombre de couches
                    old_n = mutated['n_layers']
                    mutated['n_layers'] = random.choice(self.search_space['n_layers'])
                    new_n = mutated['n_layers']
                    # Ajuste layer_sizes en conséquence
                    if new_n > old_n:
                        mutated['layer_sizes'].extend([
                            random.choice(self.search_space['layer_sizes'])
                            for _ in range(new_n - old_n)
                        ])
                    elif new_n < old_n:
                        mutated['layer_sizes'] = mutated['layer_sizes'][:new_n]
                
                elif key == 'layer_sizes' and len(mutated['layer_sizes']) > 0:
                    # Mutation d'une taille de couche aléatoire
                    idx = random.randint(0, len(mutated['layer_sizes']) - 1)
                    mutated['layer_sizes'][idx] = random.choice(self.search_space['layer_sizes'])
                
                elif key == 'learning_rate':
                    # Nouvelle valeur aléatoire pour le learning rate
                    mutated['learning_rate'] = random.uniform(*self.search_space['learning_rate'])
                
                elif key == 'batch_size':
                    mutated['batch_size'] = random.choice(self.search_space['batch_size'])
                
                elif key == 'dropout':
                    # Nouvelle valeur aléatoire pour le dropout
                    mutated['dropout'] = random.uniform(*self.search_space['dropout'])
                
                elif key == 'optimizer':
                    mutated['optimizer'] = random.choice(self.search_space['optimizer'])
                
                elif key == 'activation':
                    mutated['activation'] = random.choice(self.search_space['activation'])
        
        return mutated
    
    def evolve(self, fitness_function):
        """
        BOUCLE PRINCIPALE: Fait évoluer la population sur plusieurs générations.
        
        C'est ici que tout se passe! L'algorithme:
        1. Initialise la population
        2. Pour chaque génération:
           - Évalue tous les individus
           - Sélectionne les meilleurs
           - Crée une nouvelle génération par croisement et mutation
        
        Args:
            fitness_function: Fonction qui évalue un chromosome et retourne son fitness
            
        Returns:
            tuple: (meilleur_chromosome, meilleur_fitness)
        """
        # ÉTAPE 1: Crée la population initiale
        self.initialize_population()
        
        # BOUCLE SUR LES GÉNÉRATIONS
        for generation in range(self.generations):
            print(f"\n{'='*70}")
            print(f"GÉNÉRATION {generation + 1}/{self.generations}")
            print(f"{'='*70}")
            
            fitness_scores = []  # Stocke le fitness de chaque individu
            gen_details = []     # Stocke les détails complets de cette génération
            
            # ÉTAPE 2: ÉVALUATION - Calcule le fitness de chaque individu
            for idx, chromosome in enumerate(self.population):
                print(f"\n🧬 Individu {idx + 1}/{self.population_size}")
                
                # Appelle la fonction fitness (qui va entraîner un réseau de neurones)
                result = fitness_function(chromosome)
                fitness = result['fitness']
                fitness_scores.append(fitness)
                
                # Sauvegarde les détails pour l'historique
                gen_details.append({
                    'individual': idx + 1,
                    'chromosome': chromosome,
                    **result
                })
                
                # Met à jour le meilleur individu si nécessaire
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = deepcopy(chromosome)
            
            # STATISTIQUES DE LA GÉNÉRATION
            best_fit = max(fitness_scores)    # Meilleur fitness de cette génération
            avg_fit = np.mean(fitness_scores)  # Fitness moyen
            worst_fit = min(fitness_scores)   # Pire fitness
            
            print(f"\n📊 Résumé Génération {generation + 1}:")
            print(f"   Meilleur Fitness: {best_fit:.4f}")
            print(f"   Fitness Moyen:    {avg_fit:.4f}")
            print(f"   Pire Fitness:     {worst_fit:.4f}")
            
            # Sauvegarde dans l'historique
            self.history.append({
                'generation': generation + 1,
                'best_fitness': float(best_fit),
                'avg_fitness': float(avg_fit),
                'worst_fitness': float(worst_fit),
                'std_fitness': float(np.std(fitness_scores)),
                'best_chromosome': self.population[np.argmax(fitness_scores)],
                'details': gen_details
            })
            
            # CRÉATION DE LA NOUVELLE GÉNÉRATION (sauf pour la dernière)
            if generation < self.generations - 1:
                new_population = []
                
                # ÉLITISME: Garde les 2 meilleurs individus intacts
                # Cela garantit que les bonnes solutions ne sont jamais perdues
                elite_indices = np.argsort(fitness_scores)[-2:]
                for idx in elite_indices:
                    new_population.append(deepcopy(self.population[idx]))
                
                # Crée le reste de la nouvelle population
                while len(new_population) < self.population_size:
                    # SÉLECTION: Choisit deux parents
                    parent1 = self.tournament_selection(fitness_scores)
                    parent2 = self.tournament_selection(fitness_scores)
                    
                    # CROISEMENT: 80% de chance de faire un croisement
                    if random.random() < 0.8:
                        child = self.crossover(parent1, parent2)
                    else:
                        child = deepcopy(parent1)  # Sinon, copie simple du parent
                    
                    # MUTATION: 30% de chance de muter l'enfant
                    if random.random() < 0.3:
                        child = self.mutate(child)
                    
                    new_population.append(child)
                
                # Remplace l'ancienne population par la nouvelle
                self.population = new_population
        
        # Retourne la meilleure solution trouvée sur toutes les générations
        return self.best_individual, self.best_fitness

# ============================================================
# ENTRAÎNEMENT DES RÉSEAUX DE NEURONES - AVEC GESTION MÉMOIRE
# ============================================================

def train_model(config, X_train, y_train, X_val, y_val, epochs=10):
    """
    Entraîne un réseau de neurones avec une configuration donnée.
    
    Cette fonction est appelée par l'AG pour évaluer chaque individu.
    Elle:
    1. Construit un réseau selon les hyperparamètres
    2. L'entraîne sur les données
    3. Retourne le fitness (= accuracy de validation)
    
    Args:
        config: Configuration des hyperparamètres (chromosome)
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        epochs: Nombre d'epochs d'entraînement
        
    Returns:
        dict: Résultats (fitness, accuracy, temps, nombre de paramètres)
    """
    
    # IMPORTANT: Nettoie la mémoire avant de commencer
    # Sans ça, TensorFlow accumule les modèles en mémoire et ça plante!
    keras.backend.clear_session()
    gc.collect()
    
    try:
        # CONSTRUCTION DU MODÈLE selon le chromosome
        
        # Couche d'entrée: aplatit les images 28×28 en vecteur de 784
        model = models.Sequential([layers.Flatten(input_shape=(28, 28))])
        
        # Ajoute les couches cachées selon n_layers et layer_sizes
        for i in range(config['n_layers']):
            # Couche dense avec activation
            model.add(layers.Dense(
                config['layer_sizes'][i], 
                activation=config['activation']
            ))
            # Dropout pour régularisation (si > 0)
            if config['dropout'] > 0:
                model.add(layers.Dropout(config['dropout']))
        
        # Couche de sortie: 10 neurones (10 classes) avec softmax
        model.add(layers.Dense(10, activation='softmax'))
        
        # CHOIX DE L'OPTIMIZER selon le chromosome
        if config['optimizer'] == 'adam':
            opt = optimizers.Adam(learning_rate=config['learning_rate'])
        elif config['optimizer'] == 'sgd':
            opt = optimizers.SGD(learning_rate=config['learning_rate'], momentum=0.9)
        else:
            opt = optimizers.RMSprop(learning_rate=config['learning_rate'])
        
        # Compilation du modèle
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',  # Pour classification multi-classes
            metrics=['accuracy']
        )
        
        # ENTRAÎNEMENT DU MODÈLE
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            batch_size=config['batch_size'],
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=0  # Pas d'affichage pour ne pas polluer la sortie
        )
        train_time = time.time() - start_time
        
        # EXTRACTION DES MÉTRIQUES
        val_acc = float(history.history['val_accuracy'][-1])   # Accuracy validation
        train_acc = float(history.history['accuracy'][-1])     # Accuracy entraînement
        n_params = int(model.count_params())                   # Nombre de paramètres
        
        # CALCUL DU FITNESS
        # Ici, on utilise simplement l'accuracy de validation
        # On pourrait aussi pénaliser les modèles trop complexes ou trop lents
        fitness = val_acc
        
        # Affiche un résumé
        print(f"   Config: {config['n_layers']}×{config['layer_sizes']}, "
              f"{config['optimizer']}, lr={config['learning_rate']:.5f}")
        print(f"   Résultat: Train={train_acc:.4f}, Val={val_acc:.4f}, "
              f"Temps={train_time:.1f}s")
        
        # Prépare le résultat
        result = {
            'fitness': float(fitness),
            'accuracy': float(val_acc),
            'train_accuracy': float(train_acc),
            'training_time': float(train_time),
            'n_parameters': int(n_params)
        }
        
        # NETTOYAGE CRITIQUE: Supprime le modèle et libère la mémoire
        del model, history
        keras.backend.clear_session()
        gc.collect()
        
        return result
        
    except Exception as e:
        # En cas d'erreur, retourne un fitness de 0
        print(f"   ❌ Entraînement échoué: {str(e)}")
        keras.backend.clear_session()
        gc.collect()
        
        return {
            'fitness': 0.0,
            'accuracy': 0.0,
            'train_accuracy': 0.0,
            'training_time': 0.0,
            'n_parameters': 0
        }

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

print("\n📦 Chargement de Fashion-MNIST...")

# Fashion-MNIST: 70,000 images de vêtements (28×28 pixels, 10 classes)
# Classes: T-shirt, Pantalon, Pull, Robe, Manteau, Sandale, Chemise, Basket, Sac, Bottine
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Utilise un sous-ensemble pour accélérer le calcul
# En production, on utiliserait le dataset complet (60,000)
X_train = X_train_full[:5000].astype('float32') / 255.0  # Normalise entre 0 et 1
y_train = y_train_full[:5000]
X_val = X_train_full[5000:6000].astype('float32') / 255.0
y_val = y_train_full[5000:6000]

# Convertit les labels en one-hot encoding
# Ex: 3 devient [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = keras.utils.to_categorical(y_train, 10)
y_val = keras.utils.to_categorical(y_val, 10)

print(f"✅ Dataset: {len(X_train)} entraînement, {len(X_val)} validation")

# ============================================================
# EXÉCUTION DE L'ALGORITHME GÉNÉTIQUE
# ============================================================

print("\n🧬 Démarrage de l'Algorithme Génétique...")
print("Paramètres: 10 individus × 8 générations = 80 évaluations")
print("Temps estimé: ~30-40 minutes\n")

# Initialise l'AG avec 10 individus et 8 générations
ga = GeneticAlgorithm(population_size=10, generations=8)

# Définit la fonction fitness: simplement appeler train_model
def fitness_func(chromosome):
    return train_model(chromosome, X_train, y_train, X_val, y_val, epochs=10)

# LANCE L'ÉVOLUTION!
# C'est ici que tout se passe: 80 réseaux vont être entraînés
best_chromosome, best_fitness = ga.evolve(fitness_func)

# Sauvegarde l'historique complet en JSON
with open(f'{results_dir}/ga_history.json', 'w') as f:
    json.dump(ga.history, f, indent=2, default=str)

# ============================================================
# ENTRAÎNEMENT DU MODÈLE FINAL
# ============================================================

print("\n" + "="*70)
print("🏆 MEILLEURE SOLUTION TROUVÉE")
print("="*70)
print(f"\nMeilleur Fitness: {best_fitness:.4f}\n")
print("Meilleurs Hyperparamètres:")
for key, value in best_chromosome.items():
    print(f"  {key:15s}: {value}")

# Entraîne le meilleur modèle avec plus d'epochs pour des résultats finaux
print("\n🎯 Entraînement du modèle final avec 20 epochs...")
final_result = train_model(best_chromosome, X_train, y_train, X_val, y_val, epochs=20)

print(f"\n✅ Performance du Modèle Final:")
print(f"   Accuracy Validation: {final_result['accuracy']:.4f}")
print(f"   Temps d'Entraînement: {final_result['training_time']:.2f}s")
print(f"   Nombre de Paramètres: {final_result['n_parameters']:,}")

# ============================================================
# COMPARAISONS AVEC LES BASELINES
# ============================================================

print("\n" + "="*70)
print("📊 COMPARAISONS AVEC LES MÉTHODES DE BASE")
print("="*70)

# BASELINE 1: Recherche Aléatoire
# Teste 5 configurations aléatoires (sans AG)
print("\n🎲 Recherche Aléatoire (5 essais)...")
random_results = []
for i in range(5):
    print(f"\n  Essai {i+1}/5")
    config = ga.create_random_chromosome()  # Configuration aléatoire
    result = train_model(config, X_train, y_train, X_val, y_val, epochs=10)
    random_results.append(result)

# Garde le meilleur résultat aléatoire
best_random = max(random_results, key=lambda x: x['accuracy'])

# BASELINE 2: Configuration par Défaut
# Teste une configuration "standard" souvent utilisée
print("\n📋 Configuration par Défaut...")
default_config = {
    'n_layers': 2,
    'layer_sizes': [128, 64],
    'learning_rate': 0.001,
    'batch_size': 32,
    'dropout': 0.2,
    'optimizer': 'adam',
    'activation': 'relu'
}
default_result = train_model(default_config, X_train, y_train, X_val, y_val, epochs=10)

# COMPILATION DES RÉSULTATS
comparison = {
    'Genetic Algorithm': {
        'fitness': float(best_fitness),
        'accuracy': float(final_result['accuracy']),
        'training_time': float(final_result['training_time']),
        'n_parameters': int(final_result['n_parameters'])
    },
    'Random Search': {
        'fitness': float(best_random['fitness']),
        'accuracy': float(best_random['accuracy']),
        'training_time': float(best_random['training_time']),
        'n_parameters': int(best_random['n_parameters'])
    },
    'Default Config': {
        'fitness': float(default_result['fitness']),
        'accuracy': float(default_result['accuracy']),
        'training_time': float(default_result['training_time']),
        'n_parameters': int(default_result['n_parameters'])
    }
}

# Sauvegarde en JSON
with open(f'{results_dir}/comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

# AFFICHAGE DU TABLEAU COMPARATIF
print("\n" + "="*70)
print("COMPARAISON FINALE DES MÉTHODES")
print("="*70)
print(f"{'Méthode':<20} {'Accuracy':>12} {'Paramètres':>15}")
print("-"*70)
for method, metrics in comparison.items():
    print(f"{method:<20} {metrics['accuracy']:>12.4f} {metrics['n_parameters']:>15,}")

print("="*70)

# VERDICT
if comparison['Genetic Algorithm']['accuracy'] > comparison['Random Search']['accuracy']:
    improvement = (comparison['Genetic Algorithm']['accuracy'] - 
                  comparison['Random Search']['accuracy']) * 100
    print(f"\n✅ SUCCÈS: L'AG a trouvé une solution {improvement:.1f}% meilleure!")
else:
    print("\n⚠️  L'AG était compétitif avec la recherche aléatoire")

# ============================================================
# CRÉATION DES VISUALISATIONS
# ============================================================

print("\n📊 Création des visualisations...")

# GRAPHIQUE 1: Évolution du Fitness
# Montre comment le fitness s'améliore sur les générations
generations = [h['generation'] for h in ga.history]
best_fitness_list = [h['best_fitness'] for h in ga.history]
avg_fitness_list = [h['avg_fitness'] for h in ga.history]

plt.figure(figsize=(10, 6))
plt.plot(generations, best_fitness_list, 'o-', 
         label='Meilleur Fitness', linewidth=2, markersize=8, color='#2E86DE')
plt.plot(generations, avg_fitness_list, 's-', 
         label='Fitness Moyen', linewidth=2, markersize=6, color='#EE5A6F')
plt.xlabel('Génération', fontsize=12)
plt.ylabel('Fitness (Accuracy de Validation)', fontsize=12)
plt.title('Évolution du Fitness sur les Générations', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{results_dir}/plots/fitness_evolution.png', dpi=300)
plt.close()

# GRAPHIQUE 2: Comparaison des Méthodes
# Barres pour comparer GA vs Random Search vs Default
methods = list(comparison.keys())
accuracies = [comparison[m]['accuracy'] for m in methods]

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, accuracies, color=['#FF6B6B', '#4ECDC4', '#95E1D3'])
plt.ylabel('Accuracy', fontsize=12)
plt.title('Comparaison des Méthodes', fontsize=14, fontweight='bold')
plt.ylim([0, 1.0])
# Ajoute les valeurs au-dessus des barres
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{results_dir}/plots/comparison.png', dpi=300)
plt.close()

print(f"✅ Visualisations sauvegardées dans {results_dir}/plots/")

# ============================================================
# CONCLUSION
# ============================================================

print("\n" + "="*70)
print("✨ PROJET TERMINÉ AVEC SUCCÈS!")
print("="*70)
print(f"\n📁 Tous les résultats sont dans: {results_dir}/")
print(f"   - ga_history.json       : Historique complet de l'évolution")
print(f"   - comparison.json       : Comparaison avec les baselines")
print(f"   - plots/fitness_evolution.png : Graphique d'évolution")
print(f"   - plots/comparison.png  : Graphique de comparaison")
print("\n" + "="*70)
print("🎓 Projet Métaheuristiques - ISGA Marrakech 2024-2025")
print("="*70)
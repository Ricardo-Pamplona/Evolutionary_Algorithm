import numpy as np
from numpy.random import randint, rand

# Função objetivo para calcular a distância total da rota do mochileiro
def distancia_mochileiro(tour, distance_matrix):
    return sum(distance_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1)) + distance_matrix[tour[-1], tour[0]]

# Seleção por torneio
def selection(population, scores, k=3):
    selection_ix = randint(len(population))
    for ix in randint(0, len(population), k - 1):
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return population[selection_ix]

# Crossover entre dois pais para criar dois filhos
def crossover(parent1, parent2, c_rate):
    if rand() < c_rate:
        start, end = sorted(randint(0, len(parent1), 2))
        child1 = [-1] * len(parent1)
        child1[start:end + 1] = parent1[start:end + 1]

        fill_positions = [i for i in range(len(parent1)) if child1[i] == -1]
        parent2_positions = [city for city in parent2 if city not in child1]
        for pos in fill_positions:
            child1[pos] = parent2_positions.pop(0)

        child2 = [-1] * len(parent2)
        child2[start:end + 1] = parent2[start:end + 1]

        fill_positions = [i for i in range(len(parent2)) if child2[i] == -1]
        parent1_positions = [city for city in parent1 if city not in child2]
        for pos in fill_positions:
            child2[pos] = parent1_positions.pop(0)

        return [child1, child2]
    return [parent1.copy(), parent2.copy()]

# Operador de mutação
def mutation(tour, mutation_rate):
    for i in range(len(tour)):
        if rand() < mutation_rate:
            j = randint(0, len(tour))
            tour[i], tour[j] = tour[j], tour[i]  # Swap two cities

# Algoritmo Genético
def genetic_algorithm(objective, distance_matrix, n_iterations, n_population, c_rate, m_rate):
    n_cities = distance_matrix.shape[0]
    population = [np.random.permutation(n_cities).tolist() for _ in range(n_population)]
    best, best_eval = population[0], objective(population[0], distance_matrix)

    for gen in range(n_iterations):
        scores = [objective(c, distance_matrix) for c in population]
        for i in range(n_population):
            if scores[i] < best_eval:
                best, best_eval = population[i], scores[i]
                print(f"> Geração {gen}: nova melhor rota {best} com distância {best_eval:.3f}")

        selected = [selection(population, scores) for _ in range(n_population)]
        children = []

        for i in range(0, n_population, 2):
            p1, p2 = selected[i], selected[i + 1]
            for c in crossover(p1, p2, c_rate):
                mutation(c, m_rate)
                children.append(c)

        population = children
    return [best, best_eval]

# Exemplo de matriz de distâncias para 5 cidades
# Exemplo de matriz de distâncias para 10 cidades
distances = np.array([
    [0, 29, 20, 21, 16, 31, 100, 12, 4, 31],
    [29, 0, 15, 29, 28, 40, 72, 21, 29, 41],
    [20, 15, 0, 15, 14, 25, 81, 18, 24, 30],
    [21, 29, 15, 0, 4, 12, 92, 24, 21, 32],
    [16, 28, 14, 4, 0, 16, 94, 25, 17, 38],
    [31, 40, 25, 12, 16, 0, 78, 30, 23, 20],
    [100, 72, 81, 92, 94, 78, 0, 38, 31, 35],
    [12, 21, 18, 24, 25, 30, 38, 0, 6, 23],
    [4, 29, 24, 21, 17, 23, 31, 6, 0, 20],
    [31, 41, 30, 32, 38, 20, 35, 23, 20, 0]
])


# Definindo os parâmetros do algoritmo
n_iterations = 100
n_population = 100   
c_rate = 0.2 
m_rate = 0.8 / 5
d_scores = []

for i in range(20):
    best, score = genetic_algorithm(distancia_mochileiro, distances, n_iterations, n_population, c_rate, m_rate)
    d_scores.append(score)
    print(f"Execução {i + 1}: Melhor Distância {score}, Melhor Rota: {best}") 

d = np.array(d_scores)

if len(d) == 0:
    print("No scores recorded.")
else:
    print('Resultados das Execuções:')
    print(f'Melhor Distância Encontrada: {d.min()}')
    print(f'Pior Distância Encontrada: {d.max()}')
    print(f'Distância Média: {d.mean():.3f}')
    print(f'Desvio Padrão: {d.std():.3f}')

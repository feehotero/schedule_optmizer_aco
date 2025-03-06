import numpy as np
from numpy import random
import time
import matplotlib.pyplot as plt

subject_classes = [
    "Matemática", "História", "Ciências", "Literatura",
    "Educação Física", "Geografia", "Artes", "Física",
    "Química", "Inglês", "Biologia", "Filosofia"
]

subjects_dict = {
    1: "Matemática", 2: "História", 3: "Ciências", 4: "Literatura",
    5: "Educação Física", 6: "Geografia", 7: "Artes", 8: "Física",
    9: "Química", 10: "Inglês", 11: "Biologia", 12: "Filosofia"
}

workload_dict = {
    1: 2, 2: 2, 3: 3, 4: 2, 5: 2, 6: 3, 7: 2, 8: 3, 9: 3, 10: 3, 11: 3, 12: 2
}

score_mapping = {
    0: {6, 10}, 
    1: {8, 10}, 
    2: {1}, 
    3: {1}, 
    4: {3}
}

classes_dict = {
        0:1, 1:1, 2:2, 3:2, 4:3, 5:3, 6:3, 7:4, 8:4, 9:5, 
        10:5, 11:6, 12:6, 13:6, 14:7, 15:7, 16:8, 17:8, 18:8, 19:9, 
        20:9, 21:9, 22:10, 23:10, 24:10, 25:11, 26:11, 27:11, 28:12, 29:12
}

classes: int = 6
days: int = 5
total_classes: int = 30
ants: int = 30
alpha: float = 0.5
beta: float = 0.5
pheromone_evaporation: float = 0.5
pheromone_adjustment: float = 1.0

tours = np.empty((total_classes, total_classes))
disciplines = np.arange(total_classes)
inicio = np.copy(disciplines)
melhor_agente = -1
custos = np.zeros(total_classes)
qtde_feromonio = np.zeros(total_classes)

def create_graph():
    
    expanded_classes = [
        1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 
        5, 6, 6, 6, 7, 7, 8, 8, 8, 9, 
        9, 9, 10, 10, 10, 11, 11, 11, 12, 12
    ]
    
    num_classes = len(expanded_classes) 

    dist = [[0] * num_classes for _ in range(num_classes)]

    for i in range(num_classes):
        for j in range(i + 1, num_classes):  
            if expanded_classes[i] == expanded_classes[j]:
                dist[i][j] = 1  
            else:
                dist[i][j] = 10  
            
            dist[j][i] = dist[i][j]

    return dist, expanded_classes

pheromones = [[1.0] * total_classes for _ in range(total_classes)]

def next_class(_atual, _tour):
    soma = 0
    probabilidades = []
    cidades_disponiveis = []

    for c in range(total_classes):
        if c not in _tour: 
            cidades_disponiveis.append(c)
            soma += ((1 / dist[_atual][c]) ** beta) * (pheromones[_atual][c] ** alpha)

    for c in cidades_disponiveis:
        prob = (((1 / dist[_atual][c]) ** beta) * (pheromones[_atual][c] ** alpha)) / soma
        probabilidades.append(prob)

    return np.random.choice(cidades_disponiveis, p=probabilidades)

def calculo_custo():
    global custos
    global melhor_agente

    custos.fill(0) 

    for f in range(total_classes):
        for a in range(total_classes - 1): 
            custos[f] += dist[tours[f][a].astype(int)][tours[f][a + 1].astype(int)]
        
        blocos_por_dia = total_classes // 6  
        for dia_idx in range(blocos_por_dia):
            start_idx = dia_idx * 6
            end_idx = start_idx + 6
            valores = tours[f][start_idx:end_idx]

            classes_do_dia = set(
                classes_dict.get(int(val), None) for val in valores if int(val) in classes_dict
            )
            classes_do_dia.discard(None)

            scores_esperados = score_mapping.get(dia_idx, set())
            
            if classes_do_dia & scores_esperados:
                custos[f] += 300  

    print("Custos:", custos)

    melhor_agente = -1
    menor_custo = float('inf')  
    for a in range(total_classes):
        qtde_feromonio[a] = pheromone_adjustment / custos[a] 

        if custos[a] < menor_custo:  
            menor_custo = custos[a]
            melhor_agente = a

def atualiza_feromonio():
    global pheromones

    for c1 in range(total_classes):
        for c2 in range(total_classes):
            pheromones[c1][c2] *= (1 - pheromone_evaporation)

    for f in range(total_classes):
        for t in range(total_classes - 1):
            c1 = int(tours[f][t])
            c2 = int(tours[f][t + 1])
            pheromones[c1][c2] += pheromone_adjustment / custos[f]
            pheromones[c2][c1] += pheromone_adjustment / custos[f]  

    
dist, expanded_classes = create_graph()

def print_melhor_tour_geral():
    global melhor_tour_geral, melhor_custo_geral

    tour_melhor = melhor_tour_geral.astype(int)
    
    materias_tour = [subjects_dict[expanded_classes[i]] for i in tour_melhor] 
    
    dias_da_semana = ["segunda", "terça", "quarta", "quinta", "sexta"]

    print("")
    print("Cronograma do melhor agente geral:")

    for i, dia in enumerate(dias_da_semana):
        inicio_dia = i * 6
        fim_dia = inicio_dia + 6
        materias_do_dia = materias_tour[inicio_dia:fim_dia]
    
        print(f"{dia.capitalize()}: {', '.join(materias_do_dia)}")

    print("Custo do caminho:", melhor_custo_geral)

max_iter_sem_melhoria = 20  
iteracoes_sem_melhoria = 0  

custo_por_iteracao = []

melhor_custo_geral = float('inf')
melhor_tour_geral = None

start_time_aco = time.time()

for i in range(200):  
    np.random.shuffle(inicio)

    for f in range(total_classes):
        print("Formiga", f, "iniciando tour na materia", inicio[f])

        t = 0
        tours[f][t] = inicio[f] 
        while t < total_classes - 1:
            t += 1
            tours[f][t] = next_class(tours[f][t - 1].astype(int), tours[f][:t])

        print("Tour da formiga", f, ":", tours[f])

    calculo_custo()

    menor_custo = custos[melhor_agente]  

    if menor_custo < melhor_custo_geral:
        melhor_custo_geral = menor_custo
        melhor_tour_geral = tours[melhor_agente].copy()
        iteracoes_sem_melhoria = 0  
    else:
        iteracoes_sem_melhoria += 1  

    custo_por_iteracao.append(melhor_custo_geral) 

    atualiza_feromonio()

    if iteracoes_sem_melhoria >= max_iter_sem_melhoria:
        print("")
        print(f"Estagnação detectada após {i+1} iterações. Parando a execução.")
        break


end_time_aco = time.time()
tempo_aco = end_time_aco - start_time_aco


print_melhor_tour_geral()

plt.figure(figsize=(10, 6))
plt.plot(custo_por_iteracao, marker='o', linestyle='-', color='b', label="Custo mínimo")
plt.title("Convergência do Algoritmo")
plt.xlabel("Iterações")
plt.ylabel("Custo mínimo")
plt.legend()
plt.grid()
plt.show()

algoritmos = ["ACO"]
tempos_totais = [tempo_aco] 
desvios_padroes = [0] 

plt.figure(figsize=(8, 5))
bars = plt.bar(
    algoritmos, tempos_totais, capsize=8, 
    color=['limegreen'], edgecolor='black', alpha=0.85
)

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, 
        yval + 0.1, f"{yval:.2f}s", ha='center', fontsize=12, color='black'
    )

plt.title("Tempo Total de Execução - ACO", fontsize=16, fontweight='bold')
plt.ylabel("Tempo Total (s)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()

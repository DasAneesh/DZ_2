# Подзадача 1

import io
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from google.colab import files

# Загрузка файла пользователем
uploaded = files.upload()

# Проверка загрузки
if not uploaded:
    raise FileNotFoundError("""
    ОШИБКА: Файл не загружен!
    Действия:
    1. Нажмите на кнопку 'Choose File'
    2. Выберите файл alg_phys-cite.txt
    3. Дождитесь завершения загрузки
    """)

file_name = list(uploaded.keys())[0]

# Создание графа: {цитируемая статья: список тех, кто её цитирует}
graph = defaultdict(list)

with io.BytesIO(uploaded[file_name]) as f:
    for line in f:
        decoded_line = line.decode('utf-8').strip()
        if not decoded_line:
            continue
        papers = decoded_line.split()
        citing_paper = papers[0]
        for cited_paper in papers[1:]:
            graph[cited_paper].append(citing_paper)

# Собираем множество всех статей
all_papers = set(graph.keys()) | set(paper for citing in graph.values() for paper in citing)
in_degrees = {paper: 0 for paper in all_papers}

# Подсчет входящих степеней
for cited, citers in graph.items():
    in_degrees[cited] = len(citers)

# Построение распределения: [входящая степень] -> [кол-во статей с такой степенью]
degree_distribution = defaultdict(int)
for deg in in_degrees.values():
    degree_distribution[deg] += 1

# Нормализация
total = sum(degree_distribution.values())
norm_distribution = {k: v / total for k, v in degree_distribution.items()}

# Лог-лог график
x = list(norm_distribution.keys())
y = list(norm_distribution.values())

plt.figure(figsize=(8, 6))
plt.loglog(x, y, 'bo', markersize=4)
plt.xlabel('log(входящая степень)', fontsize=12)
plt.ylabel('log(доля статей)', fontsize=12)
plt.title('Лог-лог график распределения входящих степеней', fontsize=14)
plt.grid(True)
plt.show()

print("Основание логарифма: 10 (log-log график с масштабом по основанию 10).")


# 2

import random
from collections import defaultdict
import matplotlib.pyplot as plt

num_nodes = len(in_degrees)                 # = 27770
num_edges_real = sum(len(citers) for citers in graph.values())  # ≈ 352807
N = 10  # число случайных графов для усреднения

# Быстрая генерация: случайные пары (i, j), i ≠ j
degree_counts = defaultdict(float)

for _ in range(N):
    in_deg = defaultdict(int)

    for _ in range(num_edges_real):
        i = random.randint(0, num_nodes - 1)
        j = random.randint(0, num_nodes - 1)
        while i == j:
            j = random.randint(0, num_nodes - 1)
        in_deg[j] += 1

    for deg in in_deg.values():
        degree_counts[deg] += 1

# Усреднение и нормализация
for k in degree_counts:
    degree_counts[k] /= N

total = sum(degree_counts.values())
norm_dist_er = {k: v / total for k, v in degree_counts.items()}

# Построение графика
x_er = list(norm_dist_er.keys())
y_er = list(norm_dist_er.values())

plt.figure(figsize=(8, 6))
plt.loglog(x, y, 'bo', label='Реальный граф')
plt.loglog(x_er, y_er, 'ro', label='ER-граф (среднее по {} раз)'.format(N))
plt.xlabel('log(входящая степень)')
plt.ylabel('log(доля вершин)')
plt.title('Сравнение: реальный граф vs случайный ER-граф')
plt.legend()
plt.grid(True)
plt.show()



# 3

import math

# Из условия задачи
n = 27770             # Количество статей (узлов)
target_edges = 352807 # Количество рёбер в реальном графе

# Решим уравнение: m * (n - m) = target_edges
# Преобразуем к квадратному: m^2 - n*m + target_edges = 0
a = 1
b = -n
c = target_edges

# Дискриминант
discriminant = b**2 - 4 * a * c

# Два корня, но нам подходит тот, что поменьше
m1 = int((n - math.sqrt(discriminant)) / 2)
m2 = int((n + math.sqrt(discriminant)) / 2)

print("n = 27770")
print(f"m = {m1}")
print(f"Проверка: m * (n - m) = {m1 * (n - m1)} рёбер (приближенно к 352807)")





# 4

import random
from collections import defaultdict
import matplotlib.pyplot as plt

n = 27770  # число вершин, как в реальном графе
m = 13     # количество рёбер, исходящих из каждой новой вершины

class DPATrial:
    def __init__(self, num_nodes):
        self._num_nodes = num_nodes
        self._node_list = [node for node in range(num_nodes) for _ in range(num_nodes)]

    def run_trial(self, num_nodes_to_attach):
        new_neighbors = set()
        while len(new_neighbors) < num_nodes_to_attach:
            chosen = random.choice(self._node_list)
            new_neighbors.add(chosen)

        self._node_list.extend(new_neighbors)
        self._node_list.extend([self._num_nodes])
        self._num_nodes += 1
        return new_neighbors

# Шаг 1: построение полного графа на m вершинах
dpa_graph = defaultdict(list)
for i in range(m):
    for j in range(m):
        if i != j:
            dpa_graph[i].append(j)

# Шаг 2: генерация графа DPA с использованием DPATrial
dpa_trial = DPATrial(m)

for new_node in range(m, n):
    neighbors = dpa_trial.run_trial(m)
    dpa_graph[new_node] = list(neighbors)

# Шаг 3: Расчёт входящих степеней
in_degrees = defaultdict(int)
for src in dpa_graph:
    for dst in dpa_graph[src]:
        in_degrees[dst] += 1

for node in range(n):
    in_degrees[node] += 0

# Шаг 4: Распределение входящих степеней
degree_dist = defaultdict(int)
for deg in in_degrees.values():
    degree_dist[deg] += 1

# Нормализация
total = sum(degree_dist.values())
norm_dist_dpa = {k: v / total for k, v in degree_dist.items()}

# Шаг 5: Лог-лог график распределения
x_dpa = list(norm_dist_dpa.keys())
y_dpa = list(norm_dist_dpa.values())

plt.figure(figsize=(8, 6))
plt.loglog(x_dpa, y_dpa, 'go', label='Граф DPA (n=27770, m=13)')
plt.xlabel('log(входящая степень)')
plt.ylabel('log(доля вершин)')
plt.title('Распределение входящих степеней в графе DPA (лог-лог график)')
plt.grid(True)
plt.legend()
plt.show()



#3.2 


import random
from collections import defaultdict

class DPATrial:
    def __init__(self, num_nodes):
        self._num_nodes = num_nodes
        self._node_list = [node for node in range(num_nodes) for _ in range(num_nodes)]

    def run_trial(self, num_nodes_to_attach):
        new_neighbors = set()
        while len(new_neighbors) < num_nodes_to_attach:
            chosen = random.choice(self._node_list)
            new_neighbors.add(chosen)
        self._node_list.extend(new_neighbors)
        self._node_list.extend([self._num_nodes])
        self._num_nodes += 1
        return new_neighbors

# Параметры теста
n = 6  # всего вершин
m = 2  # исходящих ребра у каждой новой

# Построение полного графа на m вершинах
graph = defaultdict(list)
for i in range(m):
    for j in range(m):
        if i != j:
            graph[i].append(j)

# Вывод начального графа
print("Шаг 0: полный граф")
for k in graph:
    print(f"{k} -> {graph[k]}")

# Генерация оставшихся вершин
trial = DPATrial(m)
for new_node in range(m, n):
    neighbors = trial.run_trial(m)
    graph[new_node] = list(neighbors)
    print(f"Шаг {new_node - m + 1}: вершина {new_node} -> {graph[new_node]}")

# Финальный граф
print("\nИтоговый граф:")
for k in sorted(graph):
    print(f"{k} -> {graph[k]}")

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.io import loadmat
import time


#np.random.seed(0)
#ciudades = 80
#coordenadas = np.random.rand(ciudades, 2) * 100

###############################################
## Carga el archivo .mat (reemplaza 'archivo.mat' con el nombre real del archivo)
data = loadmat('2023_fase_5.mat')

## Accede a las variables de coordenadas (reemplaza 'x' y 'y' con los nombres reales de las variables en el archivo)
coordenadas_x = data['x']
coordenadas_y = data['y']

## Crea una matriz de coordenadas combinando x e y
coordenadas = np.column_stack((coordenadas_x, coordenadas_y))
coordenadas = coordenadas.astype(np.float64)


###############################################

num_ciudades = coordenadas.shape[0]

print(num_ciudades)

temperatura_inicial = 1000
temperatura_minima = 0.0001 #0.0000001 0.1
factor_enfriamiento = 0.99999 #0.999999  0.95

#elapsed_time = 0

def distancia(ciudad_a, ciudad_b):
    x_a, y_a = coordenadas[ciudad_a]
    x_b, y_b = coordenadas[ciudad_b]
    return np.hypot(x_a - x_b, y_a - y_b)



def generar_solucion_inicial(num_ciudades):
    ruta_inicial = list(range(num_ciudades))
    random.shuffle(ruta_inicial)
    return ruta_inicial

def costo(ruta):
  distancia_total = 0
  for i in range(num_ciudades-1):
    ciudad_a = ruta[i]
    ciudad_b = ruta[i+1]
    distancia_entre_ciudades = distancia(ciudad_a, ciudad_b)
    distancia_total += distancia_entre_ciudades

  ciudad_a = ruta[-1]
  ciudad_b = ruta[0]
  distancia_entre_ciudades = distancia(ciudad_a, ciudad_b)
  distancia_total += distancia_entre_ciudades

  return distancia_total

def solucion_veci(ruta_actual):
    ciudad_idx_1, ciudad_idx_2 = random.sample(range(num_ciudades), 2)

    ruta_vecina = ruta_actual.copy()
    ruta_vecina[ciudad_idx_1], ruta_vecina[ciudad_idx_2] = ruta_vecina[ciudad_idx_2], ruta_vecina[ciudad_idx_1]

    return ruta_vecina


def probabilidad_aceptacion(delta_costo, temperatura, iteracion):
    return random.random() < math.exp(-delta_costo / (temperatura * math.log(iteracion+1)))
    
#, ruta_inicial    
#dummy_arg
def recocido_simulado():
    start_time = time.time()
    temperatura = temperatura_inicial
    ruta_N = generar_solucion_inicial(num_ciudades)
    #ruta_N = ruta_inicial
    costo_N = costo(ruta_N)
    ruta_N2 = solucion_veci(ruta_N)
    iteracion = 1


    while temperatura > temperatura_minima:
        costo_N2 = costo(ruta_N2)
        delta_costo = costo_N2 - costo_N

        if delta_costo < 0 or probabilidad_aceptacion(delta_costo, temperatura, iteracion):
            ruta_N = ruta_N2
            costo_N = costo_N2
            ruta_N3 = solucion_veci(ruta_N2)
            ruta_N2 = ruta_N3
        else:
            ruta_N2 = solucion_veci(ruta_N)

        temperatura *= factor_enfriamiento
        iteracion += 1
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tiempo transcurrido: {elapsed_time} segundos")
    
    return ruta_N, costo_N



def imprimir_ruta_y_costo(ruta, costo):
    print("Ruta:")
    print(" -> ".join(map(str, ruta)))
    print("Distancia:", costo)
    

    





def dibujar_ruta(ruta, coordenadas, costo_total):
    x_coords = coordenadas[:, 0]
    y_coords = coordenadas[:, 1]

    plt.figure(figsize=(10, 6))
    plt.scatter(x_coords[ruta], y_coords[ruta], color='blue', marker='o', linestyle='-', label='Ruta')
    plt.scatter(x_coords[ruta[0]], y_coords[ruta[0]], color='red', marker='o', label='Inicio')
    
    for i in range(len(ruta) - 1):
        plt.plot([x_coords[ruta[i]], x_coords[ruta[i + 1]]], [y_coords[ruta[i]], y_coords[ruta[i + 1]]], linestyle='-', color='b')
    
    plt.plot([x_coords[ruta[-1]], x_coords[ruta[0]]], [y_coords[ruta[-1]], y_coords[ruta[0]]], linestyle='-', color='b')
    
    plt.title("Ruta óptima encontrada")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")

    info_text = f"Distancia total: {costo_total:.2f}"
    plt.annotate(info_text, xy=(0.002, 0.99), xycoords='axes fraction', fontsize=8, verticalalignment='top')

    plt.legend()
    plt.grid()
    plt.show()

    

def correr_varias_veces(num_veces):
    distancias = []
    start_time2 = time.time()

    for _ in range(num_veces):
        ruta_optima, costo_optimo = recocido_simulado()
        distancia_optima = costo_optimo
        distancias.append(distancia_optima)

    promedio_distancia = np.mean(distancias)
    desviacion_estandar = np.std(distancias)
    

    print(f"Promedio de distancia: {promedio_distancia:.2f}")
    print(f"Desviación estándar de distancia: {desviacion_estandar:.2f}")
    end_time2 = time.time()
    elapsed_time2 = end_time2 - start_time2
    print(f"Tiempo transcurrido: {elapsed_time2} segundos")

## Número de veces que quieres correr el algoritmo
num_veces = 10
#correr_varias_veces(num_veces)

## Algoritmoo paralelizado
from multiprocessing import Pool

def recocido_simulado_paralelo(num_procesos):
    # Crea un pool de procesos
    pool = Pool(processes=num_procesos)
    
    # Ejecuta el algoritmo de recocido simulado en cada proceso
    resultados = pool.map(recocido_simulado, range(num_procesos))
    
    # Encuentra la mejor solución entre todas las instancias
    mejor_ruta = None
    mejor_costo = float('inf')
    for ruta, costo in resultados:
        if costo < mejor_costo:
            mejor_ruta = ruta
            mejor_costo = costo
    
    return mejor_ruta, mejor_costo

## Función no terminada
def recocido_simulado_iterativo(ruta_inicial, num_iteraciones):
    ruta_optima = ruta_inicial
    for i in range(num_iteraciones):
        ruta_optima, costo_optimo = recocido_simulado(ruta_optima)
    return ruta_optima, costo_optimo

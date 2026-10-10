# Federated Learning con Flower

Ejercicio computacional de aprendizaje federado/colaborativo con [Flower](https://flower.ai/).
El proyecto entrena una regresión lineal distribuida entre varios clientes sin
compartir sus datos locales con el servidor.

El modelo aprende la relación:
    
```text
y = 3x + 2 + ruido
```

Cada cliente genera su propia partición sintética, entrena localmente mediante
descenso de gradiente y devuelve únicamente los parámetros del modelo. El
servidor combina las actualizaciones mediante **FedAvg** y reporta el error
cuadrático medio (**MSE**) global ponderado por la cantidad de datos de cada
cliente.

## Estructura

```text
.
├── client.py       # Cliente Flower y entrenamiento/evaluación local
├── server.py       # Servidor Flower, FedAvg y agregación del MSE
├── pyproject.toml  # Metadatos y dependencias del proyecto
└── README.md
```

## Requisitos

- Python 3.10 o superior
- `pip`
- Flower `1.26` o superior
- NumPy

## Instalación

Se recomienda usar un entorno virtual:

```bash
python -m venv .venv
```

Instala el proyecto y sus dependencias:

```bash
pip install -e .
```

## Ejecución

Desde la raíz del repositorio, inicia la aplicación de Flower:

```bash
flwr run .
```

La configuración del proyecto define cinco rondas de entrenamiento:

```toml
[tool.flwr.app.config]
num-server-rounds = 5
```

El servidor espera al menos tres clientes disponibles y utiliza todos los
clientes participantes tanto para entrenamiento como para evaluación. Durante
la ejecución se muestran mensajes similares a:

```text
[Cliente 0] Entrenamiento local -> w=..., b=...
[Cliente 0] Evaluación del modelo global -> MSE=...
[Servidor] MSE global promedio: ...
```

Flower asigna el `partition-id` de cada cliente a través de su configuración
de nodo. Ese identificador se usa como semilla para generar datos reproducibles
distintos para cada cliente.

## Cómo funciona

1. El servidor inicializa los parámetros del modelo en `w = 0` y `b = 0`.
2. Cada cliente genera 200 muestras locales y separa 160 para entrenamiento y
   40 para evaluación.
3. En cada ronda, el cliente recibe el modelo global y ejecuta 50 iteraciones
   de descenso de gradiente con una tasa de aprendizaje de `0.01`.
4. El cliente devuelve los parámetros actualizados y el número de muestras de
   entrenamiento utilizadas.
5. `FedAvg` agrega los parámetros de los clientes.
6. El servidor calcula el MSE global como un promedio ponderado de las métricas
   de evaluación.

## Configuración

La configuración principal está en [`pyproject.toml`](pyproject.toml):

- `num-server-rounds`: número de rondas federadas.
- `min-fit-clients`: clientes mínimos para entrenar en una ronda.
- `min-evaluate-clients`: clientes mínimos para evaluar.
- `min-available-clients`: clientes que deben conectarse antes de comenzar.

La generación de datos y el entrenamiento local están en
[`client.py`](client.py). La estrategia federada y la agregación de métricas
están en [`server.py`](server.py).
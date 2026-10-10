import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context


def cargar_datos(partition_id: int):
    """Datos locales privados. Cada cliente ve un rango distinto de x."""
    rng = np.random.default_rng(partition_id)
    X = rng.uniform(-3 + partition_id, 3 + partition_id, size=200)
    y = 3 * X + 2 + rng.normal(0, 0.5, size=200)   # y = 3x + 2 + ruido
    return X[:160], y[:160], X[160:], y[160:]


class ClienteRegresion(NumPyClient):
    def __init__(self, partition_id: int):
        self.cid = partition_id
        self.X_train, self.y_train, self.X_test, self.y_test = cargar_datos(partition_id)

    def fit(self, parameters, config):
        # 1. Recibir el modelo global
        w, b = parameters[0].copy(), parameters[1].copy()

        # 2. Entrenar localmente (descenso de gradiente)
        lr = 0.01
        for _ in range(50):
            error = w * self.X_train + b - self.y_train
            w -= lr * np.mean(2 * error * self.X_train)
            b -= lr * np.mean(2 * error)

        print(f"[Cliente {self.cid}] Entrenamiento local -> w={w[0]:.3f}, b={b[0]:.3f}")

        # 3. Devolver solo los pesos y la cantidad de datos usados
        return [w, b], len(self.X_train), {}

    def evaluate(self, parameters, config):
        w, b = parameters
        mse = float(np.mean((w * self.X_test + b - self.y_test) ** 2))
        print(f"[Cliente {self.cid}] Evaluación del modelo global -> MSE={mse:.4f}")
        return mse, len(self.X_test), {"mse": mse}


def client_fn(context: Context):
    # El ID lo define cada SuperNode con --node-config "partition-id=N"
    partition_id = context.node_config["partition-id"]
    return ClienteRegresion(partition_id).to_client()


app = ClientApp(client_fn=client_fn)
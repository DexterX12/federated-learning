import numpy as np
from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

W_REAL, B_REAL = 3.0, 2.0   # parámetros verdaderos: y = 3x + 2

def promedio_ponderado(metricas: list[tuple[int, Metrics]]) -> Metrics:
    """Promedia el MSE de los clientes ponderando por su cantidad de datos."""
    total = sum(n for n, _ in metricas)
    mse = sum(n * m["mse"] for n, m in metricas) / total
    print(f"[Servidor] MSE global promedio: {mse:.4f}")
    return {"mse": mse}


class FedAvgConParametros(FedAvg):
    """FedAvg que además imprime los parámetros globales w y b."""

    def __init__(self, num_rounds: int, **kwargs):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds
        self.w = None
        self.b = None

    def aggregate_fit(self, server_round, results, failures):
        parametros, metricas = super().aggregate_fit(server_round, results, failures)
        if parametros is not None:
            w, b = parameters_to_ndarrays(parametros)
            self.w, self.b = float(w[0]), float(b[0])
            print(f"[Servidor] Ronda {server_round} -> w={self.w:.4f}, b={self.b:.4f}")
        return parametros, metricas

    def aggregate_evaluate(self, server_round, results, failures):
        resultado = super().aggregate_evaluate(server_round, results, failures)
        if server_round == self.num_rounds and self.w is not None:
            print("\n===== PARÁMETROS FINALES =====")
            print(f"w final = {self.w:.4f}  (real: {W_REAL})  error = {abs(self.w - W_REAL):.4f}")
            print(f"b final = {self.b:.4f}  (real: {B_REAL})  error = {abs(self.b - B_REAL):.4f}")
            print("==============================\n")
        return resultado
    

def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = context.run_config["num-server-rounds"]

    # Modelo inicial: w = 0, b = 0
    initial_parameters = ndarrays_to_parameters([np.zeros(1), np.zeros(1)])

    strategy = FedAvgConParametros(
        num_rounds=num_rounds,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,   # espera a que se conecten 3 SuperNodes
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=promedio_ponderado,
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
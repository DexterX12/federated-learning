import numpy as np
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


def promedio_ponderado(metricas: list[tuple[int, Metrics]]) -> Metrics:
    """Promedia el MSE de los clientes ponderando por su cantidad de datos."""
    total = sum(n for n, _ in metricas)
    mse = sum(n * m["mse"] for n, m in metricas) / total
    print(f"[Servidor] MSE global promedio: {mse:.4f}")
    return {"mse": mse}


def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = context.run_config["num-server-rounds"]

    # Modelo inicial: w = 0, b = 0
    initial_parameters = ndarrays_to_parameters([np.zeros(1), np.zeros(1)])

    strategy = FedAvg(
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
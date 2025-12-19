from prometheus_client import start_http_server, Gauge, Counter, Histogram
import psutil
import time
import random

# SYSTEM METRICS
cpu_usage = Gauge(
    "cpu_usage_percent",
    "CPU usage percentage"
)

ram_usage = Gauge(
    "ram_usage_percent",
    "RAM usage percentage"
)

START_TIME = time.time()
uptime_seconds = Gauge(
    "uptime_seconds",
    "Uptime of exporter in seconds"
)

# INFERENCE METRICS
inference_requests_total = Counter(
    "inference_requests_total",
    "Total number of inference requests"
)

inference_latency_seconds = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2)
)

model_prediction_value = Gauge(
    "model_prediction_value",
    "Latest model prediction output (simulated)"
)

# METRIC COLLECTOR
def collect_metrics():
    while True:
        # CPU & RAM
        cpu_usage.set(psutil.cpu_percent())
        ram_usage.set(psutil.virtual_memory().percent)

        # Uptime exporter
        uptime_seconds.set(time.time() - START_TIME)

        # Simulasi inference
        start_time = time.time()
        time.sleep(0.02)  # simulasi waktu prediksi (20ms)

        inference_requests_total.inc()
        inference_latency_seconds.observe(time.time() - start_time)

        # Simulasi output prediksi model (0 / 1)
        model_prediction_value.set(random.choice([0, 1]))

        time.sleep(5)

# MAIN
if __name__ == "__main__":
    print("Starting Prometheus exporter on port 8000...")
    start_http_server(8000)
    collect_metrics()

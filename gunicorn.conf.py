# Gunicorn configuration file
import multiprocessing

# Server socket
bind = "0.0.0.0:10000"

# Worker processes
workers = 1  # Keep low for ML models to avoid memory issues
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Increased timeout for model inference
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "fake_news_detector"

# Server mechanics
preload_app = True  # Load app before forking workers (saves memory for ML models)

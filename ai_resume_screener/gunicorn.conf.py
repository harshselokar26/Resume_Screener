# Gunicorn configuration for AI Resume Screener
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker processes
workers = int(os.getenv('WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 30
keepalive = 5
max_requests = 1000
max_requests_jitter = 100

# Restart workers after this many requests, with up to jitter additional requests
preload_app = True

# Logging
loglevel = os.getenv('LOG_LEVEL', 'info').lower()
accesslog = "logs/access.log"
errorlog = "logs/error.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "ai_resume_screener"

# Server mechanics
daemon = False
pidfile = "tmp/gunicorn.pid"
user = None
group = None
tmp_upload_dir = "tmp"

# Worker timeout
graceful_timeout = 30

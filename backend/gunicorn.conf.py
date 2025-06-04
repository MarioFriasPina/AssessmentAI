""" Gunicorn configuration file for the backend server."""
import multiprocessing

bind = "0.0.0.0:443"
workers = multiprocessing.cpu_count() * 2 + 1

wsgi_app = "main:app"

accesslog = "/var/log/gunicorn_access.log"

errorlog = "/var/log/gunicorn_error.log"

loglevel = "info"

access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
)

keyfile='/etc/ssl/igdrasil/key.pem'
certfile='/etc/ssl/igdrasil/cert.crt'

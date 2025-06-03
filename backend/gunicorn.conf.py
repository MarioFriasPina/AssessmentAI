""" Gunicorn configuration file for the backend server."""
import multiprocessing

bind = "127.0.0.1:443"
workers = multiprocessing.cpu_count() * 2 + 1

wsgi_app = "server:app"

accesslog = "/var/log/gunicorn_access.log"

errorlog = "/var/log/gunicorn_error.log"

loglevel = "info"

access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
)

ssl_context = ('/etc/ssl/igdrasil/key.pem', '/etc/ssl/igdrasil/cert.crt')

daemon = True

# Add the frontend ip here
forwarded_allow_ips = ['*']
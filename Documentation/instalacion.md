# Instalacion del sistema

El sistema esta compuesto de 4 secciones:
- BD
- IA
- Back-end
- Front-end

# Sistema de 4 Instancias

## Requerimientos

### IA
- Servidor Linux basado en Debian
- Capacidad de subir un cloud init
- GPU
- Conexión exclusiva con el backend
- Acceso por SSH
- Python 3.12
- Python3-pip
- CUDA o similar instalado
- Python3-venv

### Backend
- Servidor Linux basado en Debian
- Capacidad de subir un cloud init
- Conexión exclusiva con la IA y la DB
- Acceso por SSH

### DB
- Servidor Linux basado en Debian
- Capacidad de subir un cloud init
- Conexión exclusiva con el backend
- Acceso por SSH

### Front-end
- Servidor Linux basado en Debian
- Capacidad de subir un cloud init
- Conexión exclusiva con el backend
- Acceso por SSH


## Comunicación

- La IA se comunica exclusivamente con el backend.
- El backend se comunica exclusivamente con la IA y la DB.
- La DB se comunica exclusivamente con el backend.
- El front-end se comunica exclusivamente con el backend.

## Acceso

Todos los servidores deben tener acceso por SSH.

# Instalacion

Para instalar el sistema, se debe de descargar el `cloud-init-<area>.yaml` de cada servicio (A excepción del de GPU), y generar por cada uno una instancia en la Nube. Asegurar que las instancias sigan reglas de acceso según definidas en la arquitectura.

Una vez generadas las instancias, se deben de configurar las variables de entorno para cada archivo:

## Front-end 

[Cloud-init](../frontend/cloud-init-front.yaml)

Entrar a `/etc/nginx/sites-available/default` y editar:

```
upstream api_backend {
    server 0.0.0.0:443;
    # agregar aquí servidores, remplazar 0.0.0.0
}
```

Para que estén las IPs que los backends tienen.

## Back-end

[Cloud-init](../backend/cloud-init-back.yaml)


Entrar a `/home/clouduser/ig-drasil-back/backend/main.py` y editar:

```
DATABASE_URL = "postgresql+asyncpg://clouduser:Mnb%40sdpoi87@172.24.0.22:5432/igdrasil"  
BACKEND_URL = "https://172.24.0.9"
```

Para `BACKEND_URL` agregar la IP del servidor del IA.

Para `DATABASE_URL` agregar el conn string de la base de datos completa.

Para aplicar cambios, correr: `sudo systemctl restart ig-drasil-back`

## DB

[Cloud-init](../DB/cloud-init-db.yaml)

No se requiere configuración adicional, sin embargo se recomienda editar el `cloud-init-db.yaml` para tener otras contraseñas.

## IA

Descargar el repositorio con `git clone https://github.com/MarioFriasPina/AssessmentAI.git`. Una vez descargado acceder desde una consola a la carpeta de IA.

Correr:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Al correr el servicio, se genera un servicio de administración. Seguir instrucciones en https://www.ssltrust.com/help/setup-guides/lightspeed-ssl-install-guide


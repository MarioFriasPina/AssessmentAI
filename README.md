# IA vs. Jugador Racing

IA vs. Jugador Racing es una plataforma interactiva de carreras donde jugadores humanos compiten en tiempo real contra una inteligencia artificial entrenada mediante aprendizaje por refuerzo. El proyecto fue desarrollado como un reto universitario en tan solo una semana, y representa una integración completa de múltiples disciplinas: desarrollo web, inteligencia artificial y despliegue en la nube.

Esta iniciativa no solo demuestra habilidades técnicas en backend, frontend y modelado de IA, sino también la capacidad de diseñar una arquitectura escalable, segura y funcional en un entorno distribuido. Desde la inferencia en GPU hasta la transmisión en tiempo real con WebSockets, el sistema simula un entorno de competencia justo entre humanos y una IA que observa y actúa bajo las mismas condiciones visuales.

## Autores

- Alejandro Fernández del Valle Herrera
- Alan Anthony Hernández Pérez
- Mario Ignacio Frías Piña
- Oswaldo Ilhuicatzi Mendizábal
---

## Tabla de contenido

- [Descripción general](#descripción-general)
- [Arquitectura de la solución](#arquitectura-de-la-solución)
  - [Redes y Nube](#redes-y-nube)
  - [Frontend](#frontend)
  - [Backend](#backend)
  - [Modelo de Inteligencia Artificial](#modelo-de-inteligencia-artificial)
- [Cómo ejecutar el proyecto](#cómo-ejecutar-el-proyecto)
- [Referencias y recursos](#referencias-y-recursos)

---

## Descripción general

Se desarrolló una plataforma de carreras en la que el usuario compite contra una IA y otros jugadores humanos. El enfoque principal fue crear una infraestructura robusta, segura y escalable, distribuyendo la carga en diferentes capas (datos, procesamiento y balanceador/frontend), además de implementar una IA entrenada por refuerzo que compite en igualdad de condiciones contra los jugadores.

---

## Arquitectura de la solución

### Redes y Nube

La infraestructura del proyecto se desplegó en la nube, organizando los recursos en tres capas principales:

- **Capa de datos**: Servidor PostgreSQL para almacenar usuarios, runs y puntajes, estandarizado y accesible solo por la capa de procesamiento.
- **Capa de procesamiento (Backend)**: Encargada de la lógica de negocio, autenticación/autorización, interacción con la IA y la base de datos. Es replicable, soportando múltiples instancias para escalar horizontalmente.
- **Capa de servicio y balanceador (Frontend)**: Servidor Nginx para archivos estáticos, balanceo de carga y terminación SSL, asegurando el acceso seguro al frontend y la API.

Adicionalmente, se implementó una máquina de cómputo específica con GPU para correr el modelo de IA de forma eficiente.

#### Ejemplo de despliegue cloud-init (Frontend y Backend)

- **Frontend**: [(cloud-init-front.yaml)](https://github.com/MarioFriasPina/AssessmentAI/blob/main/frontend/cloud-init-front.yaml)
- **Backend**: [(cloud-init-back.yaml)](https://github.com/MarioFriasPina/AssessmentAI/blob/main/backend/cloud-init-back.yaml)

---

### Frontend

El frontend se encuentra **exclusivamente** en la carpeta [`frontend2/`](https://github.com/MarioFriasPina/AssessmentAI/tree/main/frontend2) y está basado en tecnologías modernas como HTML, SCSS, JavaScript y Bootstrap 5.

- **Pantalla principal**: Permite visualizar en tiempo real la carrera del usuario y de la IA, gracias a WebRTC/WebSockets.
- **Sistema de autenticación**: Login y registro de usuarios.
- **Visualización**: Interfaz atractiva y responsiva utilizando SCSS, Bootstrap y utilidades de Tailwind.
- **Automatización y build**: Uso de Gulp para compilar SCSS, minificar recursos, optimizar imágenes y servir la aplicación.
- **Interacción con el backend**: El frontend inicia sesiones, recibe streams de video y envía controles de usuario al backend por WebSockets.

**Estructura relevante:**
- `frontend2/src/`: Código fuente principal (HTML, JS, SCSS, imágenes).
- `frontend2/gulpfile.js`: Automatización de tareas, build, minificación y live reload.
- `frontend2/dist/`: Carpeta de salida lista para producción, utilizada por Nginx o cualquier servidor estático.

---

### Backend

El backend está desarrollado en **Python** usando FastAPI, con soporte para WebSockets, autenticación JWT y acceso asíncrono a base de datos PostgreSQL.

- **Autenticación y seguridad**: Utiliza JWT y bcrypt para gestión segura de usuarios.
- **API REST y WebSockets**: Expone endpoints para iniciar sesión, registro, iniciar carreras y enviar/recibir acciones en tiempo real.
- **Procesamiento de datos**: Almacena y recupera información relevante sobre partidas y usuarios.
- **Integración con IA**: Conecta con el servidor de IA para obtener las acciones de la IA en la carrera.

**Archivos relevantes:**
- [`backend/main.py`](https://github.com/MarioFriasPina/AssessmentAI/blob/main/backend/main.py)
- [`backend/requirements.txt`](https://github.com/MarioFriasPina/AssessmentAI/blob/main/backend/requirements.txt)
- [`backend/gunicorn.conf.py`](https://github.com/MarioFriasPina/AssessmentAI/blob/main/backend/gunicorn.conf.py)

---

### Modelo de Inteligencia Artificial

La IA implementa una red neuronal convolucional entrenada mediante aprendizaje por refuerzo (Proximal Policy Optimization - PPO). Utiliza observaciones visuales iguales a las del usuario para tomar decisiones en tiempo real.

- **Ambiente**: Gymnasium/Box2D (`CarRacing-v3`), procesamiento de imágenes en escala de grises y apilado de frames.
- **Entrenamiento**: El modelo se entrenó con PPO usando código en Jupyter Notebooks y guardando checkpoints.
- **Servicio de inferencia**: El modelo se despliega usando LitServe en un servidor Python dedicado, recibiendo peticiones y retornando acciones en tiempo real.

**Archivos relevantes:**
- [`AI/SBCarRacing.ipynb`](https://github.com/MarioFriasPina/AssessmentAI/blob/main/AI/SBCarRacing.ipynb)
- [`AI/server.py`](https://github.com/MarioFriasPina/AssessmentAI/blob/main/AI/server.py)

---

## Cómo ejecutar el proyecto

### Requisitos generales

- **Python 3.10+**
- **Node.js (para build frontend2)**
- **PostgreSQL**
- **Acceso a una GPU para el servidor de IA (opcional, recomendado)**

### Pasos básicos

1. **Clona este repositorio** en las máquinas correspondientes (frontend, backend, IA).
2. **Instala las dependencias** en cada máquina usando los scripts `cloud-init-*.yaml` como referencia.
3. **Configura las variables de entorno** necesarias (bases de datos, claves JWT, etc.).
4. **Compila el frontend**:
    ```bash
    cd frontend2
    npm install
    gulp
    ```
    El resultado saldrá en `frontend2/dist`.
5. **Ejecuta los servicios**:
    - Backend: `gunicorn -c backend/gunicorn.conf.py`
    - Frontend: Servidor Nginx apuntando a la carpeta `frontend2/dist`
    - IA: `lightning deploy AI/server.py`
6. **Accede al frontend** mediante el navegador y prueba el login y la carrera.

---

## Referencias y recursos

- [Repositorio en GitHub](https://github.com/MarioFriasPina/AssessmentAI)
- [Gymnasium CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/)
- [Stable Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [FastAPI](https://fastapi.tiangolo.com/)
- [LitServe](https://github.com/Lightning-AI/litserve)

---

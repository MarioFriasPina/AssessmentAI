# IA vs. Jugador Racing

Este proyecto tiene como propósito mostrar nuestras habilidades adquiridas durante nuestrainstancia universitaria.

Elaborado por:

- Alejandro Fernandez
- Alan Aleado
- Mario Frias
- Oswaldo Ilhicatzi

Para la elaboración de wste proyecto, se tuvo solamente una semana. Se requirió generar una infraestructura de nube, y crear 3 imstancias diferentes; Front, Back y Una estancia de IA. La implementación igualmente requiere de estandares de seguridad, balanceamiento de nodos, que se vea bien visualmente, y que muestre una IA.

Para cumplir con los requisitos, se creó un juego donde tengas que competir contra una IA y otros jugadores. 

# El juego

El juego que elaboramos es uno para hacer carreras, donde tienes un carro de carreras el cual puedes controlae con acelerar, frenar y girar hacia ambos lados. 

Para medir el desempeño, se mide que tan rapido esta llendo el carro, y lo dentro que esta de la pista. A su vez la distancia a la que se encuentra de la meta. 

Una vez iniciado el juego, el jugador (ya sea humano o no, tiene que dar vueltas a la pista, hasta cumplir con una vuelta. una vez cumplida, se genera una puntuación y se le asigna esa misma al jugador. 

# Inteligencia Artificial (IA)

La IA solo puede ver lo mismo que lo que ve el jugador, y debe de recorrer la pista con los mismos movimientos con los que cuentan los jugadores. Para completar esto, se generó una red neuronal convolucional y se entrenó utilizando entenamiento por refuerzo, utilizando la puntuación como metrica de desempeño. 

# Nube 

La nube utilizada consta de 3 o más maquinas virtuales, separado en 3 secciones:

- capa de datos
- capa de procesamiento
- capa de servido de datos + balanceo

Adicionalmente, se tiene un servidor de computo donde se corre el GPU. 

## Capa de datos

La capa de datos se usa para almacenar información del usuario, corridas, etc. Esta cuenta de un servidor en postgresql que almacena datos de forma estructurada.

Los datos aparte de estar estructurados, se cuenta con una estandarización a nivel 3. 

## Capa de procesamiento 

La capa de procesamiento, o 'backend' es utilizada para autenticar, autorizar y conocer que hacen los usuarios. En esta capa se tiene acceso al servidor de computo y a la capa de datos. Es donde se mantienen los estados de los usuarios.

Adicional al manejo de información, en esta capa es se procesan los datos principalmente, por esta razón, la capa es replicable, significando que pueden existir n cantidad de nodos, los cuales pueden expandirse dependiendo de la necesidad de computo y el uso en dado momento. 

## Capa de servicio de datos estaticos y balanceo de nodos


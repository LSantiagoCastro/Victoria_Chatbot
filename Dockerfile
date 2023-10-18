FROM python:3.9-slim

# Copia el código fuente al contenedor
COPY . /app

# Actualiza pip e instala las dependencias
RUN pip install --upgrade pip && \
    pip install -r /app/requirementspip.txt

# Establece el directorio de trabajo
WORKDIR /app

# Ejecuta el proyecto
CMD ["python", "ETB_bot.py"]

# Utilisation de Python 3.13 (ou 3.12 si 3.13 n'existe pas encore)
FROM python:3.13

# Définition du répertoire de travail
WORKDIR /app

# Copie des fichiers
COPY . /app

# Installation des dépendances
RUN pip install -r requirements.txt

# Exposition du port
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

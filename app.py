from fastapi import FastAPI, HTTPException
import uvicorn
import numpy as np
import pandas as pd
import pickle
from pydantic import BaseModel
import os
app = FastAPI()


# ✅ Définition de la classe avec 10 variables (ajout de "feature_missing")
class InputVar(BaseModel):
    age: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
    feature_missing: float  # Ajout de la 10ᵉ variable


# ✅ Chargement du modèle et du scaler
try:
    regmodel = pickle.load(open('regmodel.pkl', 'rb'))
    scaler = pickle.load(open('scaling.pkl', 'rb'))
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle ou du scaler: {e}")


@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de prédiction du diabète !"}


@app.post('/predict', response_model=dict)
def predict(input_data: InputVar):
    try:
        # ✅ Conversion en DataFrame
        df = pd.DataFrame([input_data.dict()])

        # ✅ Débogage : Vérification du nombre de colonnes
        print("Nombre de colonnes reçues :", df.shape[1])
        print("Colonnes reçues :", df.columns.tolist())

        # ✅ Vérification du nombre de features avant transformation
        if df.shape[1] != 10:
            raise ValueError(f"Le modèle attend 10 features, mais {df.shape[1]} ont été reçues.")

        # ✅ Transformation des données avec le scaler
        scaled_data = scaler.transform(df)

        # ✅ Prédiction
        prediction = regmodel.predict(scaled_data)

        return {"prediction": prediction[0], "message": "Prédiction de la progression du diabète"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")




if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Récupère la variable PORT ou utilise 5000 par défaut
    uvicorn.run(app, host="0.0.0.0", port=port)

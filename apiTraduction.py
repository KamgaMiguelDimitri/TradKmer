

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import MarianMTModel, MarianTokenizer
from pydantic import BaseModel



# Charger le modèle et le tokenizer
model_name = "Helsinki-NLP/opus-mt-fr-ln"  # Modèle de traduction français-lingala
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

app = FastAPI()

# Définir les origines autorisées
origins = [
    "http://127.0.0.1:5500",  # Ajoute l'URL de ton site ici
]

# Ajouter le middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   #Autoriser ces origines
    allow_credentials=True,
    allow_methods=["*"],  # Autoriser toutes les méthodes HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Autoriser tous les en-têtes
)

# Définir le modèle de requête
class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")
async def translate(objecttext:TranslationRequest):

    # Tokeniser le texte d'entrée
    inputs = tokenizer.encode(objecttext.text, return_tensors="pt", padding=True)
    
    # Générer la traduction
    outputs = model.generate(inputs, max_length=40)
    
    # Décoder la sortie
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"translated_text": translated_text}


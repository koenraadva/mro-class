import numpy as np
from joblib import load
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from utils import text_preprocessing

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# It would've been better to use environment variables...
MRO_LANG = "norwegian,french,russian"

MRO_LANG = MRO_LANG.split(',')

models = []
for mro_lang in MRO_LANG:
    model_path =  './outputs/mro-model_'+mro_lang[:2]+'/gridsearch_model.joblib'
    print("Loading model from path:", model_path)
    models.append(load(model_path))

@app.post('/predict_class', summary='Predict MRO class of a single product')
async def predict_class(product: str, lang: Optional[str] = 'norwegian'):
    # First preprocess the product text
    minWordSize = 2
    product = text_preprocessing(product, lang, minWordSize)
    # print(product)

    # Pick the correct model
    # Some proper error handling would be nice here if language is not supported
    model = MRO_LANG.index(lang)

    # Then predict the MRO class
    data = np.array([product])
    pred_class = models[model].predict(data)[0]

    return {"class": pred_class}

#if __name__ == "__main__":
#    # Run the app with uvicorn and autoreload
#    import uvicorn
#    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
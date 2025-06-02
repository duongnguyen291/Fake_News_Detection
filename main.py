from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import torch
from transformers import (
    RobertaTokenizer, 
    XLNetTokenizer, 
    RobertaForSequenceClassification, 
    XLNetForSequenceClassification,
    DebertaTokenizer,
    DebertaForSequenceClassification
)
import preprocessor as p
import re
from pathlib import Path
import os

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global variables for models and tokenizers
models = {}
tokenizers = {}

# Available models and their paths
MODEL_PATHS = {
    # Combined models
    "roberta_combined": "model/Combined/Roberta_Combined",
    "xlnet_combined": "model/Combined/XLNet_Combined.ckpt",
    "deberta_combined": "model/Combined/Deberta_Combined",
    
    # Politics (PolitiFact) models
    "roberta_politics": "model/Roberta/roberta_politifact",
    "xlnet_politics": "model/XLNet/XLNet_PolitiFact.ckpt",
    "deberta_politics": "model/Deberta/deberta_politifact_model",
    
    # Entertainment (GossipCop) models
    "roberta_entertainment": "model/Roberta/roberta_gossip",
    "xlnet_entertainment": "model/XLNet/XLNet_Gossip.ckpt",
    "deberta_entertainment": "model/Deberta/deberta_gossipcop_model",
    
    # Covid models
    "roberta_covid": "model/Roberta/roberta_covid",
    "xlnet_covid": "model/XLNet/XLNet_Covid.ckpt",
    "deberta_covid": "model/Deberta/deberta_covid_model",
    
    # Liar models
    "roberta_liar": "model/Roberta/roberta_liar",
    "xlnet_liar": "model/XLNet/XLNet_Liar.ckpt",
    "deberta_liar": "model/Deberta/deberta_liar_model"
}

class ModelHandler:
    def __init__(self, model_type: str, tokenizer):
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = 256
        
    def preprocess(self, text: str):
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def predict(self, model, text: str):
        # Preprocess text
        inputs = self.preprocess(text)
        
        # Move model to correct device
        model = model.to(self.device)
        model.eval()
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities)
            
            # Handle different label mappings for RoBERTa and XLNet
            if self.model_type == "roberta":
                # RoBERTa: 0 = fake, 1 = real
                prediction_label = "real" if prediction.item() == 1 else "fake"
            else:  # xlnet
                # XLNet: 1 = fake, 0 = real
                prediction_label = "fake" if prediction.item() == 1 else "real"
            
        return {
            "prediction": prediction_label,
            "confidence": confidence.item()
        }

def load_model(model_type: str, category: str):
    """Load model based on type and category"""
    model_key = f"{model_type}_{category}"
    
    # Return cached model if available
    if model_key in models:
        return models[model_key]
    
    # Get model path
    model_path = MODEL_PATHS.get(model_key)
    if not model_path:
        raise ValueError(f"No model found for {model_key}")
    
    # Load appropriate model class based on type
    if model_type == "roberta":
        # For RoBERTa, we can load directly from the directory
        model = RobertaForSequenceClassification.from_pretrained(model_path)
    elif model_type == "xlnet":
        # For XLNet, we need to load from the base model first, then load state dict
        model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased')
        # Load the state dict
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        # If state_dict is wrapped in DataParallel, remove the 'module.' prefix
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    elif model_type == "deberta":
        # For DeBERTa, we can load directly from the directory
        model = DebertaForSequenceClassification.from_pretrained(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Cache model
    models[model_key] = model
    return model

def preprocess_text(text):
    """Clean and preprocess input text"""
    # Clean using tweet-preprocessor
    text = p.clean(text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.on_event("startup")
async def startup_event():
    """Load models and tokenizers on startup"""
    # Initialize tokenizers
    tokenizers['roberta'] = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizers['xlnet'] = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=False)
    tokenizers['deberta'] = DebertaTokenizer.from_pretrained('microsoft/deberta-v3-base')

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "categories": ["politics", "entertainment", "covid", "all"],
            "model_types": ["roberta", "xlnet", "deberta", "ensemble"]
        }
    )

@app.post("/predict")
async def predict(
    text: str = Form(...),
    category: str = Form(...),
    model_type: str = Form(...)
):
    """Make prediction based on input text, category and model type"""
    try:
        # Preprocess text
        cleaned_text = preprocess_text(text)
        
        if model_type == "ensemble":
            # For ensemble, use both models and average results
            results = []
            for m_type in ["roberta", "xlnet", "deberta"]:
                # Get tokenizer and create handler
                tokenizer = tokenizers[m_type]
                handler = ModelHandler(m_type, tokenizer)
                
                # Load and run model
                model = load_model(m_type, category)
                result = handler.predict(model, cleaned_text)
                results.append(result)
            
            # Average confidences for same predictions, take max confidence if different
            if results[0]["prediction"] == results[1]["prediction"] == results[2]["prediction"]:
                final_pred = results[0]["prediction"]
                final_conf = (results[0]["confidence"] + results[1]["confidence"] + results[2]["confidence"]) / 3
            else:
                # Take prediction with higher confidence
                final_pred = results[0]["prediction"] if results[0]["confidence"] > results[1]["confidence"] and results[0]["confidence"] > results[2]["confidence"] else results[1]["prediction"] if results[1]["confidence"] > results[2]["confidence"] else results[2]["prediction"]
                final_conf = max(results[0]["confidence"], results[1]["confidence"], results[2]["confidence"])
                
            result = {
                "prediction": final_pred,
                "confidence": final_conf
            }
        else:
            # Single model prediction
            tokenizer = tokenizers[model_type]
            handler = ModelHandler(model_type, tokenizer)
            model = load_model(model_type, category)
            result = handler.predict(model, cleaned_text)
        
        return {
            "text": cleaned_text,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "model_used": f"{model_type}_{category}"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
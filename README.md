# Fake News Detection

This project is the **Capstone Project** for the **Machine Learning & Data Mining (IT3191E)** course at **Hanoi University of Science and Technology (HUST)**, semester 2024.2.

---

## Objective

We developed a fake news detection system leveraging transformer-based language models fine-tuned on multi-domain datasets. The project investigates three training strategies:

- **Domain-Specific Fine-Tuning (DSFT)**  
- **Pooled-Domain Fine-Tuning (PDFT)**  
- **Domain-Matched Ensemble (DME)**
  
## Models & Datasets

We evaluated the following pretrained models:

- **RoBERTa** (roberta-base)  
- **XLNet** (xlnet-base-cased)  
- **DeBERTa** (deberta-v3-base)

Training and evaluation were conducted on:

- **COVID-19 Fake News Dataset**
- **FakeNewsNet** (GossipCop & PolitiFact)
- **LIAR**

## Key Takeaways

- Domain-specific models excel within their respective contexts.
- Pooled training provides broader generalization.
- The ensemble strategy improves overall robustness and precision.

---

## Installation Guide

### Prerequisites

- Python 3.8+
- pip (Python package installer)
- Docker (optional)

### Method 1: Local Installation

1. Clone the repository:
```bash
git clone "https://github.com/duongnguyen291/Fake_News_Detection"
cd Fake_News_Detection
```

2. Create and activate a virtual environment **(recommended)**:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python -m uvicorn main:app --reload
```

The application will be available at `http://localhost:8000`

### Method 2: Using Docker

1. Build the Docker image:
```bash
docker build -t fake-news-detection .
```

2. Run the container:
```bash
# Windows PowerShell
docker run -d -p 8000:8000 -v ${PWD}/model:/app/model fake-news-detection

# Linux/Mac
docker run -d -p 8000:8000 -v $(pwd)/model:/app/model fake-news-detection
```

The application will be available at `http://localhost:8000`

### Project Structure

```
Fake_News_Detection/
├── docs/                       # Project documentation
│   ├── notebooks/              # Jupyter notebooks
│   ├── Group1-Report-ML.pdf    # Final report
│   └── Group1-Slide-ML.pdf     # Presentation slides
├── static/                     # Static files (CSS, JS)
├── templates/                  # HTML templates for FastAPI
├── main.py                     # FastAPI entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview and guide
├── .dockerignore               # Docker ignore config
└── .gitignore                  # Git ignore config
```

### Web Interface Usage

1. Open your web browser and navigate to `http://localhost:8000`
2. Select a category (Politics, Entertainment, COVID-19, All)
3. Choose a model (RoBERTa, XLNet, or Ensemble)
4. Enter the text you want to analyze
5. Click "Analyze" and wait for the results

### API Usage

The application provides a REST API endpoint for predictions:

**POST** `/predict`

Parameters:
- `text` (string, required): The text to analyze
- `category` (string, required): Category of news (politics, entertainment, covid, all)
- `model_type` (string, required): Model to use (roberta, xlnet, ensemble)

Example Response:
```json
{
    "text": "preprocessed text",
    "prediction": "fake/real",
    "confidence": 0.95,
    "model_used": "model_category"
}
```

## Technologies & Frameworks

- Python 3.8+
- PyTorch / Transformers (by HuggingFace)
- FastAPI
- Docker
- Scikit-learn
- Pandas / NumPy
- Jupyter Notebook

---

## Information

**Instructor:** Ph.D. Nguyen Duc Anh  
**Group:** 1  

| No. | Name               | Student ID | Role         |
|-----|--------------------|------------|--------------|
| 1   | Ho Bao Thu         | 20226003   | Team Leader  |
| 2   | Tran Kim Cuong     | 20226017   | Member       |
| 3   | Nguyen Dinh Duong  | 20225966   | Member       |
| 4   | Nguyen My Duyen    | 20225967   | Member       |
| 5   | Ha Viet Khanh      | 20225979   | Member       |
| 6   | Dang Van Nhan      | 20225990   | Member       |



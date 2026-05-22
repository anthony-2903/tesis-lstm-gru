# pyrefly: ignore [missing-import]
from fastapi import FastAPI, UploadFile, File, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import random
from datetime import datetime, timedelta

app = FastAPI(title="Tesis Data API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State for simulation
global_state = {
    "is_trained": False,
    "total_records_processed": 0
}

@app.get("/")
def read_root():
    return {"status": "Backend Python funcionando"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    # Use modular cleaner for initial preprocessing
    from .b_limpieza.cleaner import clean_uploaded_file
    from .c_transformacion.normalizer import min_max_scale
    # Perform cleaning
    cleaned = clean_uploaded_file(file)
    df_cleaned = cleaned["df_cleaned"]
    # Apply normalization to numeric columns
    df_normalized = min_max_scale(df_cleaned)
    # Update global state with cleaned rows count
    global_state["total_records_processed"] += cleaned["cleaned_rows"]
    # Recompute data types after normalization (they remain same)
    data_types = cleaned["data_types"]
    # Sample data after normalization
    sample_data = df_normalized.head(5).to_dict(orient='records')
    # Return response using normalized data
    return {
        "message": "Archivo procesado exitosamente",
        "filename": file.filename,
        "original_rows": cleaned["original_rows"],
        "cleaned_rows": cleaned["cleaned_rows"],
        "rows_removed": cleaned["original_rows"] - cleaned["cleaned_rows"],
        "columns": cleaned["columns"],
        "data_types": data_types,
        "sample_data": sample_data
    }
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado. Sube un archivo .csv o .xlsx")
    
    try:
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
            
        original_rows = len(df)
        columns = df.columns.tolist()
        
        df_cleaned = df.dropna()
        cleaned_rows = len(df_cleaned)
        
        # Update global state
        global_state["total_records_processed"] += cleaned_rows
        
        data_types = {}
        for col in df_cleaned.columns:
            dtype = str(df_cleaned[col].dtype)
            if 'object' in dtype:
                data_types[col] = 'string'
            elif 'float' in dtype or 'int' in dtype:
                data_types[col] = 'number'
            elif 'datetime' in dtype:
                data_types[col] = 'date'
            else:
                data_types[col] = 'other'
                
        if len(columns) > 0:
            df_cleaned = df_cleaned.sort_values(by=columns[0])
            
        sample_data = df_cleaned.head(5).to_dict(orient="records")
        
        return {
            "message": "Archivo procesado exitosamente",
            "filename": file.filename,
            "original_rows": original_rows,
            "cleaned_rows": cleaned_rows,
            "rows_removed": original_rows - cleaned_rows,
            "columns": columns,
            "data_types": data_types,
            "sample_data": sample_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {str(e)}")

@app.post("/api/train")
async def train_models():
    # Simulate training modifying the state
    global_state["is_trained"] = True
    return {
        "status": "success",
        "message": "Entrenamiento iniciado",
        "models": ["LSTM", "GRU", "Transformer", "TCN"]
    }

def get_fluctuation(base, is_pct=True):
    if not global_state["is_trained"]:
        return 0 if not is_pct else 0.0 # If not trained, return 0 or minimal
    
    # Return base + small random fluctuation
    if is_pct:
        # +/- 0.02
        val = base + random.uniform(-0.02, 0.02)
        return min(max(val, 0.0), 1.0) # clamp between 0 and 1
    else:
        # +/- 5%
        return int(base * random.uniform(0.95, 1.05))

@app.get("/api/dashboard")
def get_dashboard_data():
    is_trained = global_state["is_trained"]
    records = global_state["total_records_processed"] if global_state["total_records_processed"] > 0 else 38294
    
    # Generate dynamic data
    kpiData = {
        "totalRecords": records,
        "totalAnomalies": {
            "phishtank": get_fluctuation(1247, False) if is_trained else 0,
            "energia": get_fluctuation(389, False) if is_trained else 0,
            "finanzas": get_fluctuation(842, False) if is_trained else 0
        },
        "bestF1": {
            "phishtank": {"model": "Transformer", "score": get_fluctuation(0.978)},
            "energia": {"model": "TCN", "score": get_fluctuation(0.945)},
            "finanzas": {"model": "Transformer", "score": get_fluctuation(0.965)}
        },
        "avgInferenceTime": round(random.uniform(10.0, 15.0), 1) if is_trained else 0
    }

    phishtankMetrics = {
        "lstm": {"precision": get_fluctuation(0.957), "recall": get_fluctuation(0.971), "f1": get_fluctuation(0.964)},
        "gru": {"precision": get_fluctuation(0.941), "recall": get_fluctuation(0.928), "f1": get_fluctuation(0.934)},
        "transformer": {"precision": get_fluctuation(0.975), "recall": get_fluctuation(0.982), "f1": get_fluctuation(0.978)},
        "tcn": {"precision": get_fluctuation(0.948), "recall": get_fluctuation(0.955), "f1": get_fluctuation(0.951)}
    }

    phishtankBarData = [
        {"name": "Anomalía (Real)", "count": get_fluctuation(1247, False)},
        {"name": "Detectado LSTM", "count": get_fluctuation(1198, False)},
        {"name": "Detectado GRU", "count": get_fluctuation(1156, False)},
        {"name": "Detectado Transf.", "count": get_fluctuation(1224, False)},
        {"name": "Detectado TCN", "count": get_fluctuation(1182, False)},
    ]

    phishtankConfusionMatrix = {
        "lstm": {"tp": get_fluctuation(1198, False), "fp": get_fluctuation(52, False), "fn": get_fluctuation(49, False), "tn": get_fluctuation(11423, False)},
        "gru": {"tp": get_fluctuation(1156, False), "fp": get_fluctuation(87, False), "fn": get_fluctuation(91, False), "tn": get_fluctuation(11389, False)},
        "transformer": {"tp": get_fluctuation(1224, False), "fp": get_fluctuation(31, False), "fn": get_fluctuation(23, False), "tn": get_fluctuation(11444, False)},
        "tcn": {"tp": get_fluctuation(1182, False), "fp": get_fluctuation(70, False), "fn": get_fluctuation(65, False), "tn": get_fluctuation(11405, False)},
    }
    
    # We omit generating ALL the domains fully here to keep it concise, but we'll return the ones needed by index.tsx and analysis.tsx
    # Return structure matching frontend expectations
    return {
        "isTrained": is_trained,
        "kpiData": kpiData,
        "phishtankMetrics": phishtankMetrics,
        "phishtankBarData": phishtankBarData,
        "phishtankConfusionMatrix": phishtankConfusionMatrix,
        "confusionMatrix": phishtankConfusionMatrix, # Alias for analysis page default
        
        "phishtankUrls": [
            {"id": 1, "url": "http://secure-paypal-login.com/verify", "real": "phishing", "lstm": "phishing", "gru": "phishing", "transformer": "phishing", "tcn": "phishing", "anomaly": True},
            {"id": 2, "url": "https://www.google.com/search?q=test", "real": "legit", "lstm": "legit", "gru": "legit", "transformer": "legit", "tcn": "legit", "anomaly": False},
            {"id": 3, "url": "http://amaz0n-security.net/update", "real": "phishing", "lstm": "phishing", "gru": "legit", "transformer": "phishing", "tcn": "phishing", "anomaly": True},
            {"id": 4, "url": "https://github.com/tensorflow/keras", "real": "legit", "lstm": "legit", "gru": "legit", "transformer": "legit", "tcn": "legit", "anomaly": False},
            {"id": 5, "url": "http://bank0famerica-login.xyz/auth", "real": "phishing", "lstm": "phishing", "gru": "phishing", "transformer": "phishing", "tcn": "phishing", "anomaly": True},
        ],
        
        "phishtankTimeline": [{"date": "2024-03-01", "anomalies": get_fluctuation(10, False), "lstm": get_fluctuation(9, False), "gru": get_fluctuation(8, False), "transformer": get_fluctuation(10, False), "tcn": get_fluctuation(9, False)} for _ in range(30)],

        "energyMetrics": {
            k: {"precision": get_fluctuation(0.9), "recall": get_fluctuation(0.9), "f1": get_fluctuation(0.9), "rmse": get_fluctuation(15.0, False)} 
            for k in ["lstm", "gru", "transformer", "tcn"]
        },
        "energyBarData": [
            {"name": "Anomalía (Real)", "count": get_fluctuation(389, False)},
            {"name": "Detectado LSTM", "count": get_fluctuation(342, False)},
            {"name": "Detectado GRU", "count": get_fluctuation(355, False)},
            {"name": "Detectado Transf.", "count": get_fluctuation(351, False)},
            {"name": "Detectado TCN", "count": get_fluctuation(365, False)},
        ],
        "energyConfusionMatrix": phishtankConfusionMatrix, # reuse structure
        
        "energySamples": [
            {"id": 1, "date": "2024-03-15 14:00", "value": get_fluctuation(742.5), "real": "anomalía", "lstm": "anomalía", "gru": "anomalía", "transformer": "anomalía", "tcn": "anomalía", "anomaly": True},
            {"id": 2, "date": "2024-03-15 15:00", "value": get_fluctuation(450.2), "real": "normal", "lstm": "normal", "gru": "normal", "transformer": "normal", "tcn": "normal", "anomaly": False},
            {"id": 3, "date": "2024-03-15 16:00", "value": get_fluctuation(812.9), "real": "anomalía", "lstm": "anomalía", "gru": "anomalía", "transformer": "anomalía", "tcn": "anomalía", "anomaly": True},
            {"id": 4, "date": "2024-03-15 17:00", "value": get_fluctuation(462.1), "real": "normal", "lstm": "normal", "gru": "normal", "transformer": "normal", "tcn": "normal", "anomaly": False},
            {"id": 5, "date": "2024-03-15 18:00", "value": get_fluctuation(448.7), "real": "normal", "lstm": "normal", "gru": "normal", "transformer": "normal", "tcn": "normal", "anomaly": False},
        ],
        
        "energyData": [{"date": "2024-03-01", "actual": get_fluctuation(450), "lstm": get_fluctuation(440), "gru": get_fluctuation(460), "transformer": get_fluctuation(445), "tcn": get_fluctuation(455), "anomaly": False} for _ in range(30)],

        "financeMetrics": {
            k: {"precision": get_fluctuation(0.9), "recall": get_fluctuation(0.9), "f1": get_fluctuation(0.9)} 
            for k in ["lstm", "gru", "transformer", "tcn"]
        },
        "financeBarData": [
            {"name": "Fraude (Real)", "count": get_fluctuation(842, False)},
            {"name": "Detectado LSTM", "count": get_fluctuation(742, False)},
            {"name": "Detectado GRU", "count": get_fluctuation(765, False)},
            {"name": "Detectado Transf.", "count": get_fluctuation(810, False)},
            {"name": "Detectado TCN", "count": get_fluctuation(802, False)},
        ],
        "financeConfusionMatrix": phishtankConfusionMatrix, # reuse structure
        
        "financeTransactions": [
            {"id": 1, "txn": "TXN_9482_AD", "amount": get_fluctuation(12450.00), "real": "fraude", "lstm": "fraude", "gru": "fraude", "transformer": "fraude", "tcn": "fraude", "anomaly": True},
            {"id": 2, "txn": "TXN_1029_BS", "amount": get_fluctuation(45.20), "real": "normal", "lstm": "normal", "gru": "normal", "transformer": "normal", "tcn": "normal", "anomaly": False},
            {"id": 3, "txn": "TXN_5521_CQ", "amount": get_fluctuation(8900.50), "real": "fraude", "lstm": "normal", "gru": "normal", "transformer": "fraude", "tcn": "fraude", "anomaly": True},
            {"id": 4, "txn": "TXN_8820_ZZ", "amount": get_fluctuation(120.00), "real": "normal", "lstm": "normal", "gru": "normal", "transformer": "normal", "tcn": "normal", "anomaly": False},
            {"id": 5, "txn": "TXN_3310_PL", "amount": get_fluctuation(15400.00), "real": "fraude", "lstm": "fraude", "gru": "fraude", "transformer": "fraude", "tcn": "fraude", "anomaly": True},
        ],
        
        "financeTimeline": [{"date": "2024-03-01", "score": get_fluctuation(50, False), "lstm": get_fluctuation(50, False), "gru": get_fluctuation(50, False), "transformer": get_fluctuation(50, False), "tcn": get_fluctuation(50, False), "anomaly": False} for _ in range(30)],

        # Radar Data for comparison
        "radarData": [
            {"metric": "F1-Score", "LSTM": get_fluctuation(0.964), "GRU": get_fluctuation(0.934), "Transformer": get_fluctuation(0.978), "TCN": get_fluctuation(0.951)},
            {"metric": "AUC-ROC", "LSTM": get_fluctuation(0.983), "GRU": get_fluctuation(0.969), "Transformer": get_fluctuation(0.991), "TCN": get_fluctuation(0.978)},
            {"metric": "Precision", "LSTM": get_fluctuation(0.957), "GRU": get_fluctuation(0.941), "Transformer": get_fluctuation(0.975), "TCN": get_fluctuation(0.948)},
            {"metric": "Recall", "LSTM": get_fluctuation(0.971), "GRU": get_fluctuation(0.928), "Transformer": get_fluctuation(0.982), "TCN": get_fluctuation(0.955)},
            {"metric": "Velocidad", "LSTM": get_fluctuation(0.62), "GRU": get_fluctuation(0.78), "Transformer": get_fluctuation(0.45), "TCN": get_fluctuation(0.96)},
            {"metric": "Eficiencia Mem.", "LSTM": get_fluctuation(0.55), "GRU": get_fluctuation(0.72), "Transformer": get_fluctuation(0.35), "TCN": get_fluctuation(0.92)},
        ],
        "comparisonBarData": [
            {"metric": "F1 (PT)", "LSTM": get_fluctuation(0.964), "GRU": get_fluctuation(0.934), "Transformer": get_fluctuation(0.978), "TCN": get_fluctuation(0.951)},
            {"metric": "F1 (FN)", "LSTM": get_fluctuation(0.903), "GRU": get_fluctuation(0.917), "Transformer": get_fluctuation(0.965), "TCN": get_fluctuation(0.952)},
            {"metric": "F1 (EN)", "LSTM": get_fluctuation(0.883), "GRU": get_fluctuation(0.918), "Transformer": get_fluctuation(0.911), "TCN": get_fluctuation(0.945)},
        ],
        "scatterData": [
            {"model": "LSTM-PhishTank", "time": get_fluctuation(342, False), "accuracy": get_fluctuation(0.964), "domain": "PhishTank"},
            {"model": "GRU-PhishTank", "time": get_fluctuation(218, False), "accuracy": get_fluctuation(0.934), "domain": "PhishTank"},
            {"model": "Transf-PhishTank", "time": get_fluctuation(512, False), "accuracy": get_fluctuation(0.978), "domain": "PhishTank"},
            {"model": "TCN-PhishTank", "time": get_fluctuation(145, False), "accuracy": get_fluctuation(0.951), "domain": "PhishTank"},
            {"model": "LSTM-Energía", "time": get_fluctuation(489, False), "accuracy": get_fluctuation(0.883), "domain": "Energía"},
            {"model": "GRU-Energía", "time": get_fluctuation(312, False), "accuracy": get_fluctuation(0.918), "domain": "Energía"},
            {"model": "Transf-Energía", "time": get_fluctuation(642, False), "accuracy": get_fluctuation(0.911), "domain": "Energía"},
            {"model": "TCN-Energía", "time": get_fluctuation(198, False), "accuracy": get_fluctuation(0.945), "domain": "Energía"},
            {"model": "LSTM-Finanzas", "time": get_fluctuation(412, False), "accuracy": get_fluctuation(0.903), "domain": "Finanzas"},
            {"model": "GRU-Finanzas", "time": get_fluctuation(275, False), "accuracy": get_fluctuation(0.917), "domain": "Finanzas"},
            {"model": "Transf-Finanzas", "time": get_fluctuation(580, False), "accuracy": get_fluctuation(0.965), "domain": "Finanzas"},
            {"model": "TCN-Finanzas", "time": get_fluctuation(165, False), "accuracy": get_fluctuation(0.952), "domain": "Finanzas"},
        ],
        "comparisonTable": [
            {"metric": "Precision Global", "lstm": get_fluctuation(0.942), "gru": get_fluctuation(0.938), "transformer": get_fluctuation(0.971), "tcn": get_fluctuation(0.945), "winner": "Transformer"},
            {"metric": "F1-Score Promedio", "lstm": get_fluctuation(0.916), "gru": get_fluctuation(0.923), "transformer": get_fluctuation(0.951), "tcn": get_fluctuation(0.949), "winner": "Transformer"},
            {"metric": "Tiempo Entrenamiento (s)", "lstm": get_fluctuation(415.5), "gru": get_fluctuation(265.0), "transformer": get_fluctuation(1240.2), "tcn": get_fluctuation(169.3), "winner": "TCN"},
            {"metric": "Memoria GPU (MB)", "lstm": get_fluctuation(1248, False), "gru": get_fluctuation(876, False), "transformer": get_fluctuation(4250, False), "tcn": get_fluctuation(452, False), "winner": "TCN"},
            {"metric": "Tiempo Inferencia (ms)", "lstm": get_fluctuation(14.2), "gru": get_fluctuation(10.6), "transformer": get_fluctuation(22.4), "tcn": get_fluctuation(4.8), "winner": "TCN"},
        ],
        "anomalyHistory": [
            {"id": 1, "date": "2024-03-15", "domain": "PhishTank", "data": "http://secure-paypal-login.com/verify", "model": "Transformer", "confidence": get_fluctuation(99.1, False), "realLabel": "phishing", "predicted": "phishing"},
            {"id": 2, "date": "2024-03-15", "domain": "Finanzas", "data": "TXN_9482_AD", "model": "TCN", "confidence": get_fluctuation(98.5, False), "realLabel": "fraude", "predicted": "fraude"},
            {"id": 3, "date": "2024-03-14", "domain": "Energía", "data": "2024-03-14 14:30:00", "model": "GRU", "confidence": get_fluctuation(89.3, False), "realLabel": "anomalía", "predicted": "anomalía"},
        ]
    }

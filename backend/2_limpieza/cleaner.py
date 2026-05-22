import pandas as pd
import io
# pyrefly: ignore [missing-import]
from fastapi import UploadFile, HTTPException


def clean_uploaded_file(file: UploadFile) -> dict:
    """Read the uploaded CSV/Excel, drop nulls, sort, and return cleaned DataFrame and metadata.
    This mirrors the previous logic inside `main.py` but is now isolated in its own module.
    Returns a dict with keys:
        - original_rows
        - cleaned_rows
        - columns
        - data_types
        - sample_data (list of dicts)
        - df_cleaned (the pandas DataFrame object)  # used by downstream stages
    """
    if not file.filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado. Sube un archivo .csv o .xlsx")
    contents = file.file.read()
    # cargar con pandas
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    else:
        df = pd.read_excel(io.BytesIO(contents))
    original_rows = len(df)
    columns = df.columns.tolist()
    # eliminación de nulos
    df_cleaned = df.dropna()
    cleaned_rows = len(df_cleaned)
    # clasificación de tipos
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
    # ordenar por primera columna si existe
    if len(columns) > 0:
        df_cleaned = df_cleaned.sort_values(by=columns[0])
    sample_data = df_cleaned.head(5).to_dict(orient='records')
    return {
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "columns": columns,
        "data_types": data_types,
        "sample_data": sample_data,
        "df_cleaned": df_cleaned,
    }

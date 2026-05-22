import pandas as pd

def min_max_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Min-Max scaling to all numeric columns in the DataFrame.
    Returns a new DataFrame with scaled values between 0 and 1.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_scaled = df.copy()
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val - min_val != 0:
            df_scaled[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_scaled[col] = 0.0
    return df_scaled

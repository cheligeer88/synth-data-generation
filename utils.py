import pandas as pd

def get_datetime_columns(df: pd.DataFrame) -> list:
    datetime_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        elif pd.api.types.is_string_dtype(df[col]):
            try:
                pd.to_datetime(df[col], errors='coerce')
                if not pd.to_datetime(df[col], errors='coerce').isna().all():
                    datetime_cols.append(col)
            except:
                pass
    return datetime_cols

def convert_datetime_numeric(df: pd.DataFrame, col: str, to_numeric: bool = True) -> pd.DataFrame:
    if to_numeric:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors='coerce').view('int64')
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].view('int64')
    else:
        df[col] = pd.Timestamp(df[col], unit='ns')
    return df

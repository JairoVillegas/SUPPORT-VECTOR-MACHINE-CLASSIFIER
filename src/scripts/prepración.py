import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def construir_dataset_listo_para_SVM(df):

    df = df.copy()

    
    # 1. Eliminar identificadores
    
    df = df.drop(columns=["ID"], errors="ignore")


    
    # 2. Convertir fechas a variables numéricas
    
    date_cols = [col for col in df.columns if "Date" in col]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    if len(date_cols) > 0:
        reference_date = df[date_cols].min().min()

        for col in date_cols:
            df[col + "_days"] = (df[col] - reference_date).dt.days

        df = df.drop(columns=date_cols)


    
    # 3. Convertir a numéricas las columnas que realmente lo sean
    
    for col in df.select_dtypes(include="object").columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass


    
    # 4. Detectar automáticamente la variable objetivo
    
    target_col = [col for col in df.columns if "Retained" in col][0]

    y = pd.to_numeric(df[target_col], errors="coerce")

    X = df.drop(columns=[target_col])


    
    # 5. Detectar columnas numéricas y categóricas
    
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns


    
    # 6. Reducir cardinalidad en categóricas
    
    max_categories = 20

    for col in cat_cols:
        top = X[col].value_counts().nlargest(max_categories).index
        X[col] = np.where(X[col].isin(top), X[col], "OTHER")


    
    # 7. Pipelines
    
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])


    
    # 8. ColumnTransformer
    
    preprocess = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])


    
    # 9. Transformación final
    
    X_prepared = preprocess.fit_transform(X)

    feature_names = preprocess.get_feature_names_out()

    X_prepared = pd.DataFrame(
        X_prepared,
        columns=feature_names,
        index=X.index
    )


    
    # 10. Salida
    
    return X_prepared, y, preprocess 

# Se retorna preprocess porque contiene las transformaciones ya ajustadas (imputación, escalado y one-hot);
# esto permite aplicar exactamente la misma estructura a datos nuevos con .transform() sin recalcular parámetros,
# evitando data leakage y asegurando que el modelo reciba las mismas columnas en el mismo orden.
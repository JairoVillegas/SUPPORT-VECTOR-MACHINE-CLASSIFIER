def construir_dataset_listo_para_SVM(df):

    import numpy as np
    import pandas as pd

    from sklearn.compose import ColumnTransformer      # para aplicar transformaciones distintas a columnas numéricas y categóricas
    from sklearn.pipeline import Pipeline              # para encadenar pasos de transformación
    from sklearn.preprocessing import OneHotEncoder, StandardScaler  # encoding para categóricas y escalado para numéricas
    from sklearn.impute import SimpleImputer           # para rellenar valores faltantes

    df = df.copy()  # trabajamos sobre una copia para no modificar el dataframe original


    df = df.drop(columns=["ID"], errors="ignore")  # eliminamos ID porque es solo un identificador y no aporta al modelo



    date_cols = [col for col in df.columns if "Date" in col]  # buscamos automáticamente columnas que sean fechas

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")  # convertimos a formato fecha (lo que no se pueda se vuelve NaT)

    if len(date_cols) > 0:
        reference_date = df[date_cols].min().min()  # tomamos la fecha más antigua como referencia

        for col in date_cols:
            df[col + "_days"] = (df[col] - reference_date).dt.days  
            # convertimos cada fecha en "días desde la fecha mínima" → número que el modelo sí puede usar

        df = df.drop(columns=date_cols)  # eliminamos las fechas originales porque el modelo no trabaja con datetime



    for col in df.select_dtypes(include="object").columns:  # recorremos columnas tipo texto
        try:
            df[col] = pd.to_numeric(df[col])  # si en realidad eran números guardados como texto los convertimos
        except:
            pass  # si no se puede convertir, es una categórica real y la dejamos así



    target_col = [col for col in df.columns if "Retained" in col][0]  # detectamos automáticamente la variable objetivo

    y = pd.to_numeric(df[target_col], errors="coerce")  # la convertimos a numérica por seguridad

    X = df.drop(columns=[target_col])  # el resto de columnas son las variables predictoras



    num_cols = X.select_dtypes(include=["int64", "float64"]).columns  # columnas numéricas
    cat_cols = X.select_dtypes(include=["object", "category"]).columns  # columnas categóricas



    max_categories = 20  # límite de categorías que dejamos por columna

    for col in cat_cols:
        top = X[col].value_counts().nlargest(max_categories).index  # nos quedamos con las más frecuentes
        X[col] = np.where(X[col].isin(top), X[col], "OTHER")  
        # las demás se agrupan como OTHER para que el one-hot no genere demasiadas columnas



    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # rellenamos valores faltantes con la mediana (robusto a outliers)
        ("scaler", StandardScaler())                    # escalamos porque SVM es sensible a la escala
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),  
        # los NA en categóricas se vuelven una categoría llamada "Missing"

        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  
        # convertimos las categorías en columnas binarias (formato denso para poder pasarlo a DataFrame)
    ])



    preprocess = ColumnTransformer([
        ("num", num_pipeline, num_cols),  # a las numéricas les aplicamos imputación + escalado
        ("cat", cat_pipeline, cat_cols)   # a las categóricas imputación + one-hot
    ])



    X_prepared = preprocess.fit_transform(X)  
    # aquí se aprenden las transformaciones (medianas, medias, categorías) y se aplican

    feature_names = preprocess.get_feature_names_out()  # obtenemos los nombres de las columnas transformadas

    X_prepared = pd.DataFrame(
        X_prepared,
        columns=feature_names,
        index=X.index
    )  # lo convertimos en DataFrame para poder verlo y usarlo fácilmente



    return X_prepared, y, preprocess  
    # devolvemos:
    # X_prepared → dataset listo para el modelo
    # y → variable objetivo
    # preprocess → las transformaciones ya aprendidas para aplicarlas luego a datos nuevos

# Se retorna preprocess porque guarda TODO lo que se aprendió al transformar los datos:
# medianas para imputar, medias y desviaciones para escalar, categorías del one-hot, etc.
# Luego permite hacer preprocess.transform(datos_nuevos) sin recalcular nada,
# evitando data leakage y asegurando que el modelo reciba exactamente las mismas columnas.
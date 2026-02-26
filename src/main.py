
# ESTA VUELTA HACE:
# 1. PREPARA LOS DATOS
# 2. ENTRENA SVM BASE
# 3. HACE EL FINE TUNING
# ============================================================

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)



# 1. FUNCIÓN → PREPARACIÓN DEL DATASET
# ============================================================

def construir_dataset_listo_para_SVM(df):

    df = df.copy()  # trabajamos sobre una copia para no alterar el original

    df = df.drop(columns=["ID"], errors="ignore")  # eliminamos identificadores


    date_cols = [col for col in df.columns if "Date" in col]  # detectamos fechas automáticamente

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")  # convertimos a formato fecha

    if len(date_cols) > 0:
        reference_date = df[date_cols].min().min()  # tomamos la fecha más antigua

        for col in date_cols:
            df[col + "_days"] = (df[col] - reference_date).dt.days  
            # convertimos cada fecha en número de días

        df = df.drop(columns=date_cols)  # eliminamos columnas originales tipo fecha


    for col in df.select_dtypes(include="object").columns:
        try:
            df[col] = pd.to_numeric(df[col])  # convertimos a número si realmente eran numéricas
        except:
            pass


    target_col = [col for col in df.columns if "Retained" in col][0]  # detectamos la variable objetivo

    y = pd.to_numeric(df[target_col], errors="coerce")

    X = df.drop(columns=[target_col])


    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns


    max_categories = 20

    for col in cat_cols:
        top = X[col].value_counts().nlargest(max_categories).index
        X[col] = np.where(X[col].isin(top), X[col], "OTHER")  # reducimos cardinalidad


    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])


    preprocess = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])


    X_prepared = preprocess.fit_transform(X)

    feature_names = preprocess.get_feature_names_out()

    X_prepared = pd.DataFrame(X_prepared, columns=feature_names, index=X.index)

    return X_prepared, y



# 2. FUNCIÓN → SVM BASELINE
# ============================================================

def aplicar_svm_baseline(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = SVC(kernel="linear", C=1, class_weight="balanced", probability=True)

    model.fit(X_train, y_train)  # entrenamiento

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n===== RESULTADOS SVM BASELINE =====\n")

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1-score :", f1_score(y_test, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_test, y_proba))

    print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))



# 3. FUNCIÓN → FINE TUNING SVM
# ============================================================

def finetuning_SVM(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    svm = SVC(probability=True, class_weight="balanced")

    param_grid = [
        {"kernel": ["linear"], "C": [0.01, 0.1, 1, 10, 100]},
        {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1]}
    ]

    grid = GridSearchCV(svm, param_grid, scoring="roc_auc", cv=5, n_jobs=-1)

    grid.fit(X_train, y_train)

    print("\nMejores hiperparámetros:", grid.best_params_)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n===== RESULTADOS SVM TUNEADO =====\n")

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1-score :", f1_score(y_test, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_test, y_proba))

    print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))



# 4. MAIN → EJECUCIÓN AUTOMÁTICA
# ============================================================

if __name__ == "__main__":

    df = pd.read_csv(r"C:\Users\jairo\OneDrive - Pontificia Universidad Javeriana\Desktop\SUPPORT-VECTOR-MACHINE-CLASSIFIER\Data\03 CSV data -- STC(A)_numerical dates.csv")  # carga el dataset
    # parte 1, esto nos devuelve el df listo y limpio
    X_prepared, y = construir_dataset_listo_para_SVM(df)
    #parte 2, esto nos hace el SVM con sus hiperparametros por defecto
    aplicar_svm_baseline(X_prepared, y)
    #parte 3, esta ultima nos hace el finetuning
    finetuning_SVM(X_prepared, y)
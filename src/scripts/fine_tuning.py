def finetuning_SVM(X, y, test_size=0.2, random_state=42):

    from sklearn.model_selection import train_test_split, GridSearchCV  # para dividir los datos y hacer la búsqueda de hiperparámetros
    from sklearn.svm import SVC                                         # el modelo SVM

    from sklearn.metrics import (
        accuracy_score,        # qué porcentaje clasifico bien
        precision_score,       # de los que dije que eran positivos, cuántos sí lo eran
        recall_score,          # de los positivos reales, cuántos detecté
        f1_score,              # balance entre precision y recall
        roc_auc_score,         # qué tan bien separa las clases en general
        classification_report, # resumen completo por clase
        confusion_matrix,      # para ver los tipos de error
        RocCurveDisplay        # para dibujar la curva ROC
    )

    import matplotlib.pyplot as plt  # para mostrar la curva ROC


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,        # porcentaje de datos que se guardan para test
        stratify=y,                 # mantiene la misma proporción de clases en train y test
        random_state=random_state  # hace que siempre obtengas el mismo split
    )


    svm = SVC(
        probability=True,           # necesario para poder calcular ROC-AUC (porque necesitamos probabilidades)
        class_weight="balanced",    # le da más importancia a la clase minoritaria
        random_state=random_state
    )


    param_grid = [

        {"kernel": ["linear"],
         "C": [0.01, 0.1, 1, 10, 100]},   # probamos distintos niveles de regularización para frontera lineal

        {"kernel": ["rbf"],
         "C": [0.1, 1, 10, 100],          # qué tan flexible queremos el modelo
         "gamma": ["scale", 0.01, 0.1, 1]}  # qué tan compleja será la frontera no lineal
    ]


    grid = GridSearchCV(
        svm,
        param_grid,          # todas las combinaciones que queremos probar
        scoring="roc_auc",   # usamos ROC-AUC como métrica principal para elegir el mejor modelo
        cv=5,                # validación cruzada en 5 partes para que el resultado sea más robusto
        n_jobs=-1,           # usa todos los núcleos del computador
        verbose=1            # muestra el progreso en pantalla
    )

    grid.fit(X_train, y_train)  # entrena todos los modelos posibles y se queda con el mejor


    print("\nMejores hiperparámetros encontrados:")
    print(grid.best_params_)    # muestra la combinación ganadora


    best_model = grid.best_estimator_  # este ya es el modelo entrenado con los mejores parámetros


    y_pred = best_model.predict(X_test)                 # predicciones finales sobre datos que nunca vio
    y_proba = best_model.predict_proba(X_test)[:, 1]    # probabilidades para calcular ROC-AUC


    accuracy  = accuracy_score(y_test, y_pred)   # desempeño general
    precision = precision_score(y_test, y_pred)  # calidad de los positivos predichos
    recall    = recall_score(y_test, y_pred)     # cuántos positivos reales encontró
    f1        = f1_score(y_test, y_pred)         # balance entre precision y recall
    roc_auc   = roc_auc_score(y_test, y_proba)   # capacidad de separar las clases


    print("\nMÉTRICAS MODELO TUNEADO\n")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-score :", f1)
    print("ROC-AUC  :", roc_auc)


    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))  # métricas por clase (útil para ver desbalance)


    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))       # muestra exactamente en qué se equivoca el modelo


    RocCurveDisplay.from_predictions(y_test, y_proba)  # dibuja la curva ROC
    plt.title("ROC Curve - Tuned SVM")
    plt.show()


    return {
        "best_model": best_model,        # el modelo final ya entrenado
        "best_params": grid.best_params_,# los hiperparámetros que ganaron
        "metrics": {                     # todas las métricas para que las uses después
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        }
    }
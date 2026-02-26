def aplicacion_modelo_SVM(X, y, test_size=0.2, random_state=42):


    from sklearn.model_selection import train_test_split   # para separar datos en entrenamiento y prueba
    from sklearn.svm import SVC                            # el modelo Support Vector Machine

    from sklearn.metrics import (
        accuracy_score,        # porcentaje total de aciertos
        precision_score,       # qué tan confiables son los positivos que predigo
        recall_score,          # qué tantos positivos reales logro detectar
        f1_score,              # balance entre precision y recall
        roc_auc_score,         # qué tan bien separa las clases en general
        classification_report, # resumen completo por clase
        confusion_matrix,      # para ver exactamente en qué se equivoca
        RocCurveDisplay        # para dibujar la curva ROC
    )

    import matplotlib.pyplot as plt   # para mostrar la gráfica de la ROC



    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,         # porcentaje de datos que se dejan para test
        stratify=y,                  # mantiene la misma proporción de clases en train y test
        random_state=random_state   # hace que el resultado sea reproducible
    )



    svm_model = SVC(
        kernel="linear",             # usamos una frontera lineal → este es el modelo base
        C=1,                         # nivel de regularización estándar (ni muy flexible ni muy rígido)
        class_weight="balanced",    # compensa si hay desbalance entre clases
        probability=True,           # necesario para poder calcular ROC-AUC
        random_state=random_state
    )



    svm_model.fit(X_train, y_train)  # el modelo aprende los patrones usando solo los datos de entrenamiento



    y_pred = svm_model.predict(X_test)                 # predicciones finales (clase 0 o 1)
    y_proba = svm_model.predict_proba(X_test)[:, 1]    # probabilidades de la clase positiva (para ROC-AUC)



    accuracy  = accuracy_score(y_test, y_pred)   # desempeño general del modelo
    precision = precision_score(y_test, y_pred)  # qué tan bien le apunta cuando dice “positivo”
    recall    = recall_score(y_test, y_pred)     # qué tantos positivos reales logra encontrar
    f1        = f1_score(y_test, y_pred)         # equilibrio entre precision y recall
    roc_auc   = roc_auc_score(y_test, y_proba)   # capacidad de separar las clases sin depender del umbral


    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-score :", f1)
    print("ROC-AUC  :", roc_auc)



    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))   # métricas detalladas para cada clase



    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))        # muestra aciertos y errores (FP, FN, etc.)



    RocCurveDisplay.from_predictions(y_test, y_proba)  # dibuja la curva ROC usando las probabilidades
    plt.show()



    return {
        "modelo_usado": svm_model,   # el modelo ya entrenado (lo puedes reutilizar sin volver a entrenar)
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }
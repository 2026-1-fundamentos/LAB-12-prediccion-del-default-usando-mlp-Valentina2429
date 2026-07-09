# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import gzip
import json
import os
import pickle

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

def cargar_datos():
    train = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    test = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
    return train, test

def limpiar_datos(df):
    df = df.copy()
    df = df.rename(columns={"default payment next month": "default"})
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    df = df[(df["MARRIAGE"] != 0) & (df["EDUCATION"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x <= 3 else 4)
    df = df.dropna()
    return df

def dividir_xy(train, test):
    x_train = train.drop(columns=["default"])
    y_train = train["default"]

    x_test = test.drop(columns=["default"])
    y_test = test["default"]

    return x_train, y_train, x_test, y_test

def crear_pipeline(x_train):
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical_features = [col for col in x_train.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
            ('scaler', StandardScaler(), numerical_features),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('pca', PCA()),
        ('classifier', MLPClassifier(max_iter=15000, random_state=21))
    ])
                

    return pipeline

def entrenar_modelo(pipeline, x_train, y_train):
    params = {
        "pca__n_components": [None],
        "feature_selection__k": [20],
        "classifier__hidden_layer_sizes": [(50, 30, 40, 60)],
        "classifier__alpha": [0.26],
        'classifier__learning_rate_init': [0.001],
        }
    '''
    validacion = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=12345,
    )
    '''
    busqueda = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )

    busqueda.fit(x_train, y_train)

    return busqueda

def guardar_modelo(modelo):
    os.makedirs("files/models", exist_ok=True)

    with gzip.open("files/models/model.pkl.gz", "wb") as archivo:
        pickle.dump(modelo, archivo)

def calcular_metricas(modelo, x, y, nombre_dataset):

    predicciones = modelo.predict(x)

    return {
        "type": "metrics",
        "dataset": nombre_dataset,
        "precision": round(precision_score(y, predicciones, zero_division=0), 3),
        "balanced_accuracy": round(balanced_accuracy_score(y, predicciones), 3),
        "recall": round(recall_score(y, predicciones, zero_division=0), 3),
        "f1_score": round(f1_score(y, predicciones, zero_division=0), 3),
        }

def calcular_matriz_confusion(modelo, x, y, nombre_dataset):
    predicciones = modelo.predict(x)
    matriz = confusion_matrix(y, predicciones, labels=[0, 1])

    return {
        "type": "cm_matrix",
        "dataset": nombre_dataset,
        "true_0": {
            "predicted_0": int(matriz[0, 0]),
            "predicted_1": int(matriz[0, 1]),
        },
        "true_1": {
            "predicted_0": int(matriz[1, 0]),
            "predicted_1": int(matriz[1, 1]),
        },
    }

def guardar_metricas(modelo, x_train, y_train, x_test, y_test):
    os.makedirs("files/output", exist_ok=True)

    registros = [
        calcular_metricas(modelo, x_train, y_train, "train"),
        calcular_metricas(modelo, x_test, y_test, "test"),
        calcular_matriz_confusion(modelo, x_train, y_train, "train"),
        calcular_matriz_confusion(modelo, x_test, y_test, "test"),
    ]

    with open("files/output/metrics.json", "w", encoding="utf-8") as archivo:
        for registro in registros:
            archivo.write(json.dumps(registro) + "\n")

if __name__ == "__main__":
    
    train, test = cargar_datos()

    train = limpiar_datos(train)
    test = limpiar_datos(test)

    x_train, y_train, x_test, y_test = dividir_xy(train, test)

    pipeline = crear_pipeline(x_train)
    modelo = entrenar_modelo(pipeline, x_train, y_train)

    guardar_modelo(modelo)
    guardar_metricas(modelo, x_train, y_train, x_test, y_test)
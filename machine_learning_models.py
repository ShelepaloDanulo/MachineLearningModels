import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_optimized_xgboost_model(X, y):
    # Оголошення діапазонів гіперпараметрів, які потрібно оптимізувати
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01],
        'n_estimators': [100, 200],
        'gamma': [0, 0.1],
        'min_child_weight': [1, 3, 5],
    }
    # Розділення на тренувальний та тестовий набори даних
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
    # Перетворення y_test в мультикласовий формат
    y_test_multiclass = np.argmax(y_test, axis=1)
    # Створення та навчання моделей XGBoost для кожного класу
    num_classes = 3
    models = []

    reg_strength_L2 = 0.1  # Задаємо силу L2-регуляризації
    for class_idx in range(num_classes):
        y_train_class = y_train[:, class_idx]
        # Створення моделі з L2-регуляризацією
        model = xgb.XGBClassifier(objective='binary:logistic', reg_lambda=reg_strength_L2)
        # Використання RandomizedSearchCV для пошуку найкращих гіперпараметрів
        random_search = RandomizedSearchCV(model, param_distributions=param_grid, cv=5, n_iter=30)
        random_search.fit(X_train, y_train_class)
        # Збереження моделі з найкращими гіперпараметрами
        best_model = random_search.best_estimator_
        models.append(best_model)
    # Прогнозування на тестових даних
    y_pred_prob = np.zeros((len(X_test), num_classes))
    for class_idx in range(num_classes):
        model = models[class_idx]
        y_pred_prob[:, class_idx] = model.predict_proba(X_test)[:, 1]
    # Вибір класу з найвищою ймовірністю
    y_pred = np.argmax(y_pred_prob, axis=1)
    # Оцінка точності моделі
    accuracy = accuracy_score(y_test_multiclass, y_pred)
    precision = precision_score(y_test_multiclass, y_pred, average='weighted')
    recall = recall_score(y_test_multiclass, y_pred, average='weighted')
    f1 = f1_score(y_test_multiclass, y_pred, average='weighted')
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    # Перехресна перевірка
    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5)
    print("Cross-Validation scores:", cv_scores)
    print("Average Cross-Validation score:", np.mean(cv_scores))

def train_random_forest_model(X, y):
    # Розділення на тренувальний та тестовий набори даних
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
    # Створення та навчання моделі Random Forest
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    # Прогнозування на тестових даних
    y_pred = model.predict(X_test)
    # Оцінка точності моделі
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    # Перехресна перевірка
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("Cross-Validation scores:", cv_scores)
    print("Average Cross-Validation score:", np.mean(cv_scores))

def train_logistic_regression(X, y):
    # Розділення на тренувальний та тестовий набори даних
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
    # Створення та навчання моделі Логістичної регресії з підходом One-vs-Rest
    num_classes = len(np.unique(y))
    models = []

    reg_strengths = [0.1, 0.1, 0.1]  # Задайте сили регуляризації для кожного класу
    for class_idx in range(num_classes):
        y_train_class = y_train[:, class_idx]
        # Створення моделі Логістичної регресії з відповідною силою регуляризації
        model = LogisticRegression(C=0.1 / reg_strengths[class_idx])
        model.fit(X_train, y_train_class)
        models.append(model)
    # Прогнозування на тестових даних
    y_pred_prob = np.zeros((len(X_test), num_classes))
    for class_idx in range(num_classes):
        model = models[class_idx]
        y_pred_prob[:, class_idx] = model.predict_proba(X_test)[:, 1]
    # Вибір класу з найвищою ймовірністю
    y_pred = np.argmax(y_pred_prob, axis=1)
    # Оцінка точності моделі
    accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    precision = precision_score(np.argmax(y_test, axis=1), y_pred, average='weighted', zero_division=0)
    recall = recall_score(np.argmax(y_test, axis=1), y_pred, average='weighted')
    f1 = f1_score(np.argmax(y_test, axis=1), y_pred, average='weighted')
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

if __name__ == "__main__":
    # Отримання шляху до файла відносно кореня проекту
    data_path = os.path.join(os.getcwd(), 'data', 'wine.data')
    # Завантаження даних з локального файла
    column_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                    'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
                    'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                    'Proline']
    data = pd.read_csv(data_path, names=column_names)
    # Виведення таблиці даних
    df = pd.DataFrame(data, columns=column_names)
    print(df)
    # Розділення ознак та цілової змінної
    X = data.drop('Class', axis=1)
    y = data['Class']
    # Обробка пропущених значень
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    # Масштабування ознак
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    # Конвертація масиву NumPy у DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    y_df = pd.DataFrame(y, columns=['Class'])
    # Кодування категоріальної змінної (Class)
    categorical_feature = ['Class']
    categorical_transformer = OneHotEncoder()
    class_encoder = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_feature)
        ])
    y_encoded = class_encoder.fit_transform(y_df)
    # Кодування числових ознак
    numeric_features = X.columns
    numeric_transformer = StandardScaler()
    numeric_encoder = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Застосування кодування до ознак та цілової змінної
    X_processed = numeric_encoder.fit_transform(X_scaled_df)
    processed_data = pd.DataFrame(X_processed, columns=numeric_features)
    processed_data = pd.concat([processed_data, pd.DataFrame(y_encoded)], axis=1)
    # Виведення оброблених даних
    print(processed_data)
    # Виклик функції зі змінними X_scaled та y_encoded
    print('----------------------optimized_xgboost_model------------------------')
    train_optimized_xgboost_model(X_scaled, y_encoded)
    print('-----------------------random_forest_model---------------------------')
    train_random_forest_model(X_scaled, y_encoded)
    print('-----------------------logistic_regression---------------------------')
    train_logistic_regression(X_scaled, y_encoded)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных
train_data = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')
print("Столбцы в датасете:", train_data.columns.tolist())

# Удаляем только 'Name' (остальные столбцы полезны)
data = train_data.drop(['Name'], axis=1)

# Заполняем пропуски в 'Age' (если есть)
data['Age'].fillna(data['Age'].median(), inplace=True)

# Преобразуем 'Sex' в числовой формат (male=1, female=0)
data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})

# Разделяем данные
X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    objective='binary:logistic',
    eval_metric='logloss',
    early_stopping_rounds=20,
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=10)

# Оценка модели
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Матрица ошибок
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.show()

# Важность признаков
xgb.plot_importance(model)
plt.show()

# Сохранение модели
model.save_model('titanic_xgboost.json')


# Пример нового пассажира
new_passenger = pd.DataFrame({
    'Pclass': [1],                      # Класс билета (1, 2, 3)
    'Sex': [1],                         # 1 = male, 0 = female
    'Age': [26],                        # Возраст
    'Siblings/Spouses Aboard': [1],     # Аналог SibSp
    'Parents/Children Aboard': [2],     # Аналог Parch
    'Fare': [100]                      # Стоимость билета
})

# Загрузка модели и предсказание
model = xgb.XGBClassifier()
model.load_model('titanic_xgboost.json')
pred = model.predict(new_passenger)
print("Выжил" if pred[0] == 1 else "Не выжил")

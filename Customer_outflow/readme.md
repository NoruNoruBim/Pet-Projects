Тестовое задание по оттоку клиентов (классификация).

Условие задания находится в файле "Описание задания.ipynb"

В папке code:

1) extract_data.py - получение данных из исходных файлов (train_action и train_pay)
2) cleaning_preprocessing.py - очистка и предобработка получившегося датасета
3) feature_selection.py - отбор признаков методом SequentialFeatureSelector
4) model_selection.py - отбор моделей с помощью GridSearchCV
5) predict_final.py - построение финального предсказания для тестовой выборки

"ROC_train.png" - график ROC для тренировочной выборки.
Перекрестно-проверочная roc_auc = 0.846.

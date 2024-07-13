# random_forest.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

def train_rf(X_train, X_test, y_train, y_test,label_encoder):
    rf_model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [300, 350],
        'max_depth': [15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_rf_model = grid_search.best_estimator_

    # 评估随机森林模型
    y_pred = best_rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred)
    rf_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rf_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    rf_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    rf_class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)

    rf_log = {
        'model': 'RandomForest',
        'best_params': grid_search.best_params_,
        'accuracy': rf_accuracy,
        'precision': rf_precision,
        'recall': rf_recall,
        'f1_score': rf_f1,
        'classification_report': rf_class_report
    }

    joblib.dump(best_rf_model, 'trained_models/rf_model.pkl')

    return rf_log

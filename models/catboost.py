# catboost.py
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import joblib

def train_catboost(X_train, X_test, y_train, y_test, label_encoder, iterations=1000, learning_rate=0.1, depth=6):
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        verbose=100
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    y_pred = model.predict(X_test)
    y_pred = [int(x) for x in y_pred]

    catboost_accuracy = accuracy_score(y_test, y_pred)
    catboost_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    catboost_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    catboost_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    catboost_class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)

    catboost_log = {
        'model': 'CatBoost',
        'accuracy': catboost_accuracy,
        'precision': catboost_precision,
        'recall': catboost_recall,
        'f1_score': catboost_f1,
        'classification_report': catboost_class_report
    }

    model.save_model('trained_models/catboost_model.cbm')

    return catboost_log

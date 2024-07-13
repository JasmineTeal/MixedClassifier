# lgbm.py
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import joblib

def train_lgbm(X_train, X_test, y_train, y_test, label_encoder, num_leaves=31, learning_rate=0.05, num_boost_round=100):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': len(label_encoder.classes_),
        'metric': 'multi_logloss',
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
    }

    evals_result = {}
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=num_boost_round,
                    valid_sets=[lgb_eval],
                    evals_result=evals_result)

    # 获取最佳迭代次数
    best_iter = len(evals_result['valid_0']['multi_logloss']) - 1
    min_logloss = min(evals_result['valid_0']['multi_logloss'])
    for i, logloss in enumerate(evals_result['valid_0']['multi_logloss']):
        if logloss == min_logloss:
            best_iter = i
            break

    y_pred = gbm.predict(X_test, num_iteration=best_iter)
    y_pred_max = [list(x).index(max(x)) for x in y_pred]

    lgbm_accuracy = accuracy_score(y_test, y_pred_max)
    lgbm_precision = precision_score(y_test, y_pred_max, average='weighted', zero_division=0)
    lgbm_recall = recall_score(y_test, y_pred_max, average='weighted', zero_division=0)
    lgbm_f1 = f1_score(y_test, y_pred_max, average='weighted', zero_division=0)
    lgbm_class_report = classification_report(y_test, y_pred_max, target_names=label_encoder.classes_, zero_division=0)

    lgbm_log = {
        'model': 'LightGBM',
        'accuracy': lgbm_accuracy,
        'precision': lgbm_precision,
        'recall': lgbm_recall,
        'f1_score': lgbm_f1,
        'classification_report': lgbm_class_report
    }

    joblib.dump(gbm, 'trained_models/lgbm_model.pkl')

    return lgbm_log

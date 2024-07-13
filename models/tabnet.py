# tabnet.py
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import joblib

def train_tabnet(X_train, X_test, y_train, y_test, label_encoder, batch_size=256, lr=2e-2, num_epochs=300):
    tabnet_model = TabNetClassifier(
        n_d=64, n_a=64, n_steps=5, gamma=1.5, n_independent=2, n_shared=2,
        lambda_sparse=1e-3, momentum=0.3, clip_value=2.0,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=lr),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15
    )
    tabnet_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=num_epochs,
        patience=30,  # 设置patience为0，避免早停
        batch_size=batch_size,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    # 预测
    y_pred = tabnet_model.predict(X_test)
    y_proba = tabnet_model.predict_proba(X_test)

    # 评估TabNet模型
    tabnet_accuracy = accuracy_score(y_test, y_pred)
    tabnet_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    tabnet_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    tabnet_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    tabnet_class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)

    tabnet_log = {
        'model': 'TabNet',
        'accuracy': tabnet_accuracy,
        'precision': tabnet_precision,
        'recall': tabnet_recall,
        'f1_score': tabnet_f1,
        'classification_report': tabnet_class_report
    }

    # 保存TabNet模型
    joblib.dump(tabnet_model, 'trained_models/tabnet_model.pkl')

    return tabnet_log

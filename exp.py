import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import preprocess_data, create_dataloaders
from models.model import CNN1D, TransformerModel, MLP
from sklearn.linear_model import LogisticRegression
from pytorch_tabnet.tab_model import TabNetClassifier

# 检查CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 读取数据
file_path = '../dataset/result_data_train.csv'
data = pd.read_csv(file_path)
X_train, X_test, y_train, y_test, label_encoder, scaler = preprocess_data(data, fit_scaler=True, fit_label_encoder=True)

# 创建数据加载器
train_loader, test_loader = create_dataloaders(X_train, X_test, y_train, y_test, batch_size=8)

# 训练CNN模型
input_dim = X_train.shape[1]
num_classes = len(label_encoder.classes_)
cnn_model = CNN1D(input_dim, num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=5e-4)
num_epochs = 300
best_loss = float('inf')

for epoch in range(num_epochs):
    cnn_model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = cnn_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

    cnn_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = cnn_model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    class_report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, output_dict=True, zero_division=0)

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_cnn_log = {
            'model': 'CNN1D',
            'best_epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report
        }
        torch.save(cnn_model.state_dict(), 'trained_models/cnn1d_model.pth')

# 训练Transformer模型
transformer_model = TransformerModel(input_dim, num_classes).to(device)
optimizer = optim.Adam(transformer_model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
best_loss = float('inf')

for epoch in range(num_epochs):
    transformer_model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = transformer_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    # scheduler.step(avg_loss)  # Update learning rate based on validation loss
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

    transformer_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = transformer_model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    class_report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, output_dict=True, zero_division=0)

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_transformer_log = {
            'model': 'TransformerModel',
            'best_epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report
        }
        torch.save(transformer_model.state_dict(), 'trained_models/transformer_model.pth')

# MLP模型
mlp_model = MLP(input_dim, num_classes).to(device)
optimizer = optim.Adam(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)

num_epochs = 100
best_loss = float('inf')

for epoch in range(num_epochs):
    mlp_model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = mlp_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

    mlp_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = mlp_model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    class_report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, output_dict=True, zero_division=0)

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_mlp_log = {
            'model': 'MLP',
            'best_epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report
        }
        torch.save(mlp_model.state_dict(), 'trained_models/mlp_model.pth')

# 训练TabNet模型
tabnet_model = TabNetClassifier(
    n_d=64, n_a=64, n_steps=5, gamma=1.5, n_independent=2, n_shared=2,
    lambda_sparse=1e-3, momentum=0.3, clip_value=2.0,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15
)
tabnet_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    max_epochs=300,
    patience=30,  # 增加耐心值
    batch_size=256,
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
tabnet_class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)

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

# 训练随机森林模型
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
joblib.dump(best_rf_model, 'trained_models/rf_model.pkl')

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

# 堆叠特征
# RF
rf_proba = best_rf_model.predict_proba(X_test)

# CNN
cnn_model.eval()
cnn_proba = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = cnn_model(inputs)
        proba = torch.nn.functional.softmax(outputs, dim=1)
        cnn_proba.extend(proba.cpu().numpy())
cnn_proba = np.array(cnn_proba)

# Transformer
transformer_model.eval()
transformer_proba = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = transformer_model(inputs)
        proba = torch.nn.functional.softmax(outputs, dim=1)
        transformer_proba.extend(proba.cpu().numpy())
transformer_proba = np.array(transformer_proba)

# MLP
mlp_model.eval()
mlp_proba = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = mlp_model(inputs)
        proba = torch.nn.functional.softmax(outputs, dim=1)
        mlp_proba.extend(proba.cpu().numpy())
mlp_proba = np.array(mlp_proba)

# TabNet
tabnet_proba = y_proba

# Ensure all probabilities have the same length
assert rf_proba.shape[0] == cnn_proba.shape[0] == transformer_proba.shape[0] == mlp_proba.shape[0] == tabnet_proba.shape[0], "Mismatched shapes in prediction probabilities."

stacked_features = np.hstack((rf_proba, cnn_proba, transformer_proba, mlp_proba, tabnet_proba))

# 训练元学习器
meta_model = LogisticRegression(random_state=42)
meta_model.fit(stacked_features, y_test)
joblib.dump(meta_model, 'pickles/meta_model.pkl')

# 保存日志
log_data = {
    'best_cnn_log': best_cnn_log,
    'best_transformer_log': best_transformer_log,
    'best_mlp_log': best_mlp_log,
    'tabnet_log': tabnet_log,
    'rf_log': rf_log
}

with open('logs/training_log.json', 'w') as log_file:
    json.dump(log_data, log_file, indent=4)


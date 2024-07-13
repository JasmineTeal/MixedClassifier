import argparse
import json
import os
import joblib
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from utils import preprocess_test_data, create_test_dataloader, SklearnWrapper, predict_proba
from models.model import CNN1D, TransformerModel, MLP
from catboost import CatBoostClassifier

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

MODEL_CLASSES = {
    'cnn': CNN1D,
    'transformer': TransformerModel,
    'mlp': MLP
}

MODEL_PATHS = {
    'cnn': 'trained_models/cnn_model.pth',
    'transformer': 'trained_models/transformer_model.pth',
    'mlp': 'trained_models/mlp_model.pth',
    'tabnet': 'trained_models/tabnet_model.pkl',
    'rf': 'trained_models/rf_model.pkl',
    'catboost': 'trained_models/catboost_model.cbm'
}

# 加载数据和预处理
def load_data(file_path):
    data = pd.read_csv(file_path)
    X_test, y_test = preprocess_test_data(data)
    return data, X_test, y_test

# 自动检测和加载模型进行预测
def load_and_predict(model_name, X_test, device, input_dim, num_classes):
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unsupported model: {model_name}")
    model_path = MODEL_PATHS[model_name]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_name == 'rf':
        model = joblib.load(model_path)
        proba = predict_proba(model, X_test, model_type='sklearn')
        print(f"RF proba shape: {proba.shape}")
        return proba
    elif model_name in MODEL_CLASSES:
        model = SklearnWrapper(model_class=MODEL_CLASSES[model_name], input_dim=input_dim, num_classes=num_classes, model_type='pytorch', device=device)
        proba = model.predict_proba(X_test)
        print(f"{model_name.upper()} proba shape: {proba.shape}")
        return proba
    elif model_name == 'tabnet':
        model = joblib.load(model_path)
        proba = predict_proba(model, X_test, model_type='tabnet')
        print(f"TabNet proba shape: {proba.shape}")
        return proba
    elif model_name == 'catboost':
        model = CatBoostClassifier()
        model.load_model(model_path)
        proba = predict_proba(model, X_test, model_type='catboost')
        print(f"CatBoost proba shape: {proba.shape}")
        return proba
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def main(args):
    # 加载数据
    original_data, X_test, y_test = load_data(args.data_path)

    # 加载Label Encoder
    label_encoder = joblib.load('pickles/label_encoder.pkl')

    # 加载设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 固定要加载的模型
    fixed_models = ['rf', 'cnn', 'tabnet', 'catboost', 'mlp', 'transformer']

    # 特征堆叠
    stacked_proba = []
    test_loader = create_test_dataloader(X_test, batch_size=8)
    for model_name in fixed_models:
        proba = load_and_predict(model_name, X_test, device, input_dim=X_test.shape[1], num_classes=len(label_encoder.classes_))
        stacked_proba.append(proba)
    stacked_features = np.hstack(stacked_proba)
    print(f"Stacked features shape: {stacked_features.shape}")

    # 加载元学习器
    meta_model = joblib.load('pickles/meta_model.pkl')

    # 使用元学习器进行最终预测
    final_predictions = meta_model.predict(stacked_features)

    # 模型评估
    accuracy = accuracy_score(y_test, final_predictions)
    precision = precision_score(y_test, final_predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, final_predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, final_predictions, average='weighted', zero_division=0)
    classification_rep = classification_report(y_test, final_predictions, target_names=label_encoder.classes_, output_dict=True, zero_division=0)

    # 保存日志
    log_data = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': classification_rep
    }

    with open('logs/prediction_log.json', 'w') as log_file:
        json.dump(log_data, log_file, indent=4)

    print("预测准确率:", accuracy)
    print("分类报告:\n", classification_rep)

    # 输出预测标签和apk_name
    apk_names = original_data['apk_name'].values
    predicted_labels = label_encoder.inverse_transform(final_predictions)
    output_df = pd.DataFrame({'apk_name': apk_names, 'predicted_label': predicted_labels})
    output_df.to_csv('logs/predicted_labels.csv', index=False)
    print("预测标签已保存到 logs/predicted_labels.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using stacked models')
    parser.add_argument('--data_path', default='dataset/result_data_test.csv', type=str, help='Path to the test data')
    args = parser.parse_args()
    main(args)

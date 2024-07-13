import argparse
import json
import os
import joblib
import torch
import pandas as pd
import numpy as np
from utils import preprocess_test_data, SklearnWrapper, predict_proba
from models.model import CNN1D, TransformerModel, MLP
from catboost import CatBoostClassifier

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

# 加载模型和进行预测的函数
def load_and_predict(model_name, sample, device, input_dim, num_classes):
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unsupported model: {model_name}")
    model_path = MODEL_PATHS[model_name]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_name == 'rf':
        model = joblib.load(model_path)
        proba = model.predict_proba(sample)
        print(f"RF proba shape: {proba.shape}")
        return proba
    elif model_name in MODEL_CLASSES:
        model = SklearnWrapper(model_class=MODEL_CLASSES[model_name], input_dim=input_dim, num_classes=num_classes, model_type='pytorch', device=device)
        proba = model.predict_proba(sample)
        print(f"{model_name.upper()} proba shape: {proba.shape}")
        return proba
    elif model_name == 'tabnet':
        model = joblib.load(model_path)
        proba = model.predict_proba(sample)
        print(f"TabNet proba shape: {proba.shape}")
        return proba
    elif model_name == 'catboost':
        model = CatBoostClassifier()
        model.load_model(model_path)
        proba = model.predict_proba(sample)
        print(f"CatBoost proba shape: {proba.shape}")
        return proba
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# 加载数据和预处理
def load_data(file_path):
    data = pd.read_csv(file_path)
    X_test, y_test = preprocess_test_data(data)
    return X_test, y_test

# 加载Label Encoder
label_encoder = joblib.load('pickles/label_encoder.pkl')

# 加载设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 固定要加载的模型
fixed_models = ['cnn', 'transformer', 'mlp', 'tabnet', 'rf', 'catboost']

# 处理单个样本的预测
def predict_single_sample(sample):
    # 预处理单个样本
    sample_df = pd.DataFrame([sample])
    X_sample, _ = preprocess_test_data(sample_df)

    # 转换为合适的输入格式（保持二维）
    sample_np = np.array(X_sample).reshape(1, -1)

    # 获取各个模型的预测概率
    stacked_proba = []
    for model_name in fixed_models:
        proba = load_and_predict(model_name, sample_np, device, input_dim=sample_np.shape[1], num_classes=len(label_encoder.classes_))
        stacked_proba.append(proba[0])
    stacked_sample_features = np.hstack(stacked_proba)
    print(f"Stacked sample features shape: {stacked_sample_features.shape}")

    # 加载元学习器
    meta_model = joblib.load('pickles/meta_model.pkl')

    # 使用元学习器进行最终预测
    final_sample_prediction_proba = meta_model.predict_proba([stacked_sample_features])[0]
    final_sample_prediction = meta_model.predict([stacked_sample_features])[0]

    # 将预测结果转为标签
    predicted_label = label_encoder.inverse_transform([final_sample_prediction])[0]

    # 输出所有标签及其对应的概率
    predicted_probabilities = dict(zip(label_encoder.classes_, final_sample_prediction_proba))

    return predicted_label, predicted_probabilities

def main(args):
    # 加载数据
    data = pd.read_csv(args.data_path)

    # 测试单个样本预测
    sample = data.iloc[0].to_dict()  # 示例，使用数据集中的第一个样本
    predicted_label, predicted_probabilities = predict_single_sample(sample)
    log_data = {
        'predicted_label': predicted_label,
        'predicted_probabilities': predicted_probabilities
    }

    with open('logs/single_prediction.json', 'w') as f:
        json.dump(log_data, f, indent=4)

    print("单个样本的预测结果:", predicted_label)
    print("单个样本的预测概率:", json.dumps(predicted_probabilities, indent=4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict a single sample using stacked models')
    parser.add_argument('--data_path', default='dataset/tmp.csv', type=str, help='Path to the test data')
    args = parser.parse_args()
    main(args)

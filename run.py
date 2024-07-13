import argparse
import json

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from models.model import CNN1D, TransformerModel, MLP
from utils import preprocess_data, create_dataloaders, SklearnWrapper, predict_proba
from models.cnn import train_cnn
from models.transformer import train_transformer
from models.mlp import train_mlp
from models.tabnet import train_tabnet
from models.rf import train_rf
from models.catBoost import train_catboost
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

def train_and_log_model(model_name, X_train, X_test, y_train, y_test, label_encoder, args):
    if model_name == 'cnn':
        log = train_cnn(X_train, X_test, y_train, y_test, label_encoder, batch_size=args.batch_size, lr=args.lr, num_epochs=args.epochs)
    elif model_name == 'transformer':
        log = train_transformer(X_train, X_test, y_train, y_test, label_encoder, batch_size=args.batch_size, lr=args.lr, num_epochs=args.epochs)
    elif model_name == 'mlp':
        log = train_mlp(X_train, X_test, y_train, y_test, label_encoder, batch_size=args.batch_size, lr=args.lr, num_epochs=args.epochs)
    elif model_name == 'tabnet':
        log = train_tabnet(X_train, X_test, y_train, y_test, label_encoder, batch_size=args.batch_size, lr=args.lr, num_epochs=args.epochs)
    elif model_name == 'rf':
        log = train_rf(X_train, X_test, y_train, y_test, label_encoder)
    elif model_name == 'catboost':
        log = train_catboost(X_train, X_test, y_train, y_test, label_encoder, iterations=args.epochs, learning_rate=args.lr, depth=args.depth)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    log_filename = f'logs/{model_name}_training_log.json'
    with open(log_filename, 'w') as log_file:
        json.dump(log, log_file, indent=4)

def load_model(model_name, input_dim, num_classes, device):
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unsupported model: {model_name}")
    model_path = MODEL_PATHS[model_name]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_name == 'cnn':
        model = CNN1D(input_dim=input_dim, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    elif model_name == 'transformer':
        model = TransformerModel(input_dim=input_dim, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    elif model_name == 'mlp':
        model = MLP(input_dim=input_dim, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    elif model_name == 'tabnet':
        model = joblib.load(model_path)
        return model
    elif model_name == 'rf':
        model = joblib.load(model_path)
        return model
    elif model_name == 'catboost':
        model = CatBoostClassifier()
        model.load_model(model_path)
        return model
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def train_meta_model(X_train, X_test, y_train, y_test, input_dim, num_classes, device):
    # RF
    rf_model = load_model('rf', input_dim, num_classes, device)
    rf_proba = rf_model.predict_proba(X_test)
    print(f"RF proba shape: {rf_proba.shape}")

    # CNN
    cnn_model = load_model('cnn', input_dim, num_classes, device)
    cnn_model.eval()
    cnn_proba = []
    _, test_loader = create_dataloaders(X_train, X_test, y_train, y_test, batch_size=8)
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = cnn_model(inputs)
            proba = torch.nn.functional.softmax(outputs, dim=1)
            cnn_proba.extend(proba.cpu().numpy())
    cnn_proba = np.array(cnn_proba)
    print(f"CNN proba shape: {cnn_proba.shape}")

    # Transformer
    transformer_model = load_model('transformer', input_dim, num_classes, device)
    transformer_model.eval()
    transformer_proba = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = transformer_model(inputs)
            proba = torch.nn.functional.softmax(outputs, dim=1)
            transformer_proba.extend(proba.cpu().numpy())
    transformer_proba = np.array(transformer_proba)
    print(f"Transformer proba shape: {transformer_proba.shape}")

    # MLP
    mlp_model = load_model('mlp', input_dim, num_classes, device)
    mlp_model.eval()
    mlp_proba = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = mlp_model(inputs)
            proba = torch.nn.functional.softmax(outputs, dim=1)
            mlp_proba.extend(proba.cpu().numpy())
    mlp_proba = np.array(mlp_proba)
    print(f"MLP proba shape: {mlp_proba.shape}")

    # TabNet
    tabnet_model = load_model('tabnet', input_dim, num_classes, device)
    tabnet_proba = tabnet_model.predict_proba(X_test)
    print(f"TabNet proba shape: {tabnet_proba.shape}")

    # CatBoost
    catboost_model = load_model('catboost', input_dim, num_classes, device)
    catboost_proba = catboost_model.predict_proba(X_test)
    print(f"CatBoost proba shape: {catboost_proba.shape}")

    # Ensure all probabilities have the same length
    assert rf_proba.shape[0] == cnn_proba.shape[0] == transformer_proba.shape[0] == mlp_proba.shape[0] == \
           tabnet_proba.shape[0] == catboost_proba.shape[0], "Mismatched shapes in prediction probabilities."

    stacked_features = np.hstack((rf_proba, cnn_proba, transformer_proba, mlp_proba, tabnet_proba, catboost_proba))
    print(f"Stacked features shape: {stacked_features.shape}")

    # 训练元学习器
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(stacked_features, y_test)
    joblib.dump(meta_model, 'pickles/meta_model.pkl')
    return meta_model

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.only_stack:
        # 加载数据
        file_path = args.data_path
        data = pd.read_csv(file_path)
        X_train, X_test, y_train, y_test, label_encoder, scaler = preprocess_data(data, fit_scaler=True, fit_label_encoder=True)

        input_dim = X_train.shape[1]
        num_classes = len(label_encoder.classes_)

        print("Training meta model...")
        meta_model = train_meta_model(X_train, X_test, y_train, y_test, input_dim, num_classes, device)
        joblib.dump(meta_model, 'pickles/meta_model.pkl')
        print("Meta model saved as 'pickles/meta_model.pkl'")
    else:
        # 读取数据
        file_path = args.data_path
        data = pd.read_csv(file_path)
        X_train, X_test, y_train, y_test, label_encoder, scaler = preprocess_data(data, fit_scaler=True, fit_label_encoder=True)

        model_names = args.models.split(',')
        for model_name in model_names:
            print(f"Training model: {model_name}")
            train_and_log_model(model_name, X_train, X_test, y_train, y_test, label_encoder, args)

        if args.stack:
            input_dim = X_train.shape[1]
            num_classes = len(label_encoder.classes_)

            print("Training meta model...")
            meta_model = train_meta_model(X_train, X_test, y_train, y_test, input_dim, num_classes, device)
            joblib.dump(meta_model, 'pickles/meta_model.pkl')
            print("Meta model saved as 'pickles/meta_model.pkl'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models')
    parser.add_argument('--data_path', type=str, default='dataset/result_data_train.csv', help='Path to the training data')
    parser.add_argument('--models', type=str, help='Comma separated list of models to train (e.g., "cnn,transformer,mlp")')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training')
    parser.add_argument('--num_leaves', type=int, default=31, help='Number of leaves for LightGBM')
    parser.add_argument('--depth', type=int, default=6, help='Depth for CatBoost')
    parser.add_argument('--stack', action='store_true', help='Flag to enable stacking of trained models')
    parser.add_argument('--only_stack', action='store_true', help='Flag to only perform stacking of trained models')

    args = parser.parse_args()
    main(args)

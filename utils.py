import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import joblib
import os
from sklearn.base import BaseEstimator, ClassifierMixin

def preprocess_data(data, fit_scaler=False, fit_label_encoder=False):
    X = data.drop(columns=['apk_name', 'label']).values.astype(np.float32)
    y = data['label'].values

    if fit_label_encoder:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        joblib.dump(label_encoder, 'pickles/label_encoder.pkl')
    else:
        label_encoder = joblib.load('pickles/label_encoder.pkl')
        y = label_encoder.transform(y)

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        joblib.dump(scaler, 'pickles/scaler.pkl')
    else:
        scaler = joblib.load('pickles/scaler.pkl')
        X = scaler.transform(X)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder, scaler

def preprocess_test_data(data):
    X = data.drop(columns=['apk_name', 'label']).values.astype(np.float32)
    y = data['label'].values

    label_encoder = joblib.load('pickles/label_encoder.pkl')
    y = label_encoder.transform(y)

    scaler = joblib.load('pickles/scaler.pkl')
    X = scaler.transform(X)

    return X, y

def create_dataloaders(X_train, X_test, y_train, y_test, batch_size):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def create_test_dataloader(X_test, batch_size):
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def predict_proba(model, X_test, model_type='sklearn', device=None):
    if model_type == 'sklearn':
        return model.predict_proba(X_test)
    elif model_type == 'pytorch':
        model.to(device)
        model.eval()
        proba = []
        test_loader = create_test_dataloader(X_test, batch_size=8)
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs[0].to(device)  # Unpack the tuple
                outputs = model(inputs)
                proba_batch = torch.nn.functional.softmax(outputs, dim=1)
                proba.extend(proba_batch.cpu().numpy())
        return np.array(proba)
    elif model_type == 'tabnet':
        return model.predict_proba(X_test)
    elif model_type == 'catboost':
        return model.predict_proba(X_test)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

class SklearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_class, input_dim, num_classes, model_type, device='cpu'):
        self.model_class = model_class
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model_type = model_type
        self.device = device
        self.model = self.model_class(input_dim=self.input_dim, num_classes=self.num_classes)
        self.model.to(self.device)
        self.classes_ = np.arange(self.num_classes)

    def fit(self, X, y=None):
        # This fit method is a placeholder and will not be used
        return self

    def predict_proba(self, X):
        return predict_proba(self.model, X, model_type=self.model_type, device=self.device)

# MixedClassifier
> This code is for the "China Software Cup" competition's A2 topic, 
> written by team "South Leaf" from Sun Yueqi College, CUMT

## Project Overview

The `MixedClassifier` project is a machine learning framework designed to classify software samples using an ensemble of various machine learning models. The project includes training, stacking, and predicting functionalities with multiple models, including CNN, Transformer, MLP, TabNet, RandomForest, and CatBoost.

## Models

The project includes the following models:
1. **CNN (Convolutional Neural Network)**:
   - Implemented in `models/cnn.py`.
   - Suitable for extracting spatial features from data.
   
2. **Transformer**:
   - Implemented in `models/transformer.py`.
   - Effective for sequential data due to self-attention mechanisms.

3. **MLP (Multilayer Perceptron)**:
   - Implemented in `models/mlp.py`.
   - A basic feedforward neural network for general purposes.

4. **TabNet**:
   - Implemented in `models/tabnet.py`.
   - Utilizes sequential attention to perform feature selection in tabular data.

5. **RandomForest**:
   - Implemented in `models/rf.py`.
   - An ensemble of decision trees to improve predictive performance.

6. **CatBoost**:
   - Implemented in `models/catboost.py`.
   - A gradient boosting algorithm that handles categorical features well.

## Data Input Format

The input data should be in CSV format with the following columns:
- `apk_name`: The name of the software sample.
- `label`: The class label of the sample.
- Other columns should contain the features used for classification.

Example:
```
apk_name,feature1,feature2,feature3,...,featureN,label
sample1,0.1,0.2,0.3,...,0.4,class1
sample2,0.5,0.6,0.7,...,0.8,class2
```

## Usage

### Training

To train the models, run the `run.py` script with the appropriate arguments:

```
python run.py --data_path <path_to_training_data> --models <comma_separated_model_list> --batch_size <batch_size> --lr <learning_rate> --epochs <num_epochs> --stack 
```

Example:
```
python run.py --data_path dataset/result_data_train.csv --models cnn,transformer,mlp --batch_size 8 --lr 5e-4 --epochs 300 --stack 
```

### Prediction

To predict using the trained models, run the `predict.py` script:

```
python predict.py --data_path <path_to_test_data>
```

Example:
```
python predict.py --data_path dataset/result_data_test.csv
```

This will output the prediction accuracy, classification report, and save the predicted labels along with the apk_name to `logs/predicted_labels.csv`.

### Single Sample Prediction

To predict a single sample using the trained models, run the `predict_single.py` script:

```
python predict_single.py --data_path <path_to_test_data>
```

Example:
```
python predict_single.py --data_path ../dataset/tmp.csv
```

The `predict_single.py` script will output both the predicted label and the probabilities for all labels.

### Utility Functions

Utility functions for data preprocessing and model handling are implemented in `utils.py`. Some key functions include:
- `preprocess_data`: Preprocesses training data.
- `preprocess_test_data`: Preprocesses test data.
- `create_test_dataloader`: Creates a DataLoader for test data.
- `SklearnWrapper`: A wrapper to integrate PyTorch models with scikit-learn.

## Logging

Training and prediction logs are stored in the `logs/` directory. Each model has its own training log, and prediction results are logged in `prediction_log.json`.

## Pickle Files

The `pickles/` directory contains serialized objects such as label encoders, scalers, and the meta-model used for stacking.

## Trained Models

The `trained_models/` directory contains the serialized weights and models for each trained model.

## Notes

- Ensure all required libraries are installed, including PyTorch, scikit-learn, pandas, numpy, and CatBoost, etc.
- Code is written in PyTorch 2.3.1(with CUDA 12.1) and Python 3.8
- The code is designed to run on both CPU and GPU. Ensure the proper device is set in the scripts.


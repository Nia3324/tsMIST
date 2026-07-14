import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt

# Sklearn & Preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import OneHotEncoder

# Tensorflow / Keras
import tensorflow as tf
import keras
from keras import initializers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Time Series Libraries
import pycatch22
from sktime.transformations.panel.rocket import Rocket
from aeon.classification.deep_learning import InceptionTimeClassifier 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_keras_verbose'] = '0'

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

def create_lstm(sequence_length: int, n_features: int, n_classes: int, seed: int = 42) -> Sequential:

    set_seeds(seed)

    kernel_initializer = initializers.GlorotUniform(seed=seed)
    recurrent_initializer = initializers.Orthogonal(seed=seed)
    bias_initializer = initializers.Zeros() 

    if n_classes == 2:
        output_units = 1
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        output_units = n_classes
        activation = 'softmax'
        loss = 'categorical_crossentropy'

    model = Sequential([
        LSTM(32, input_shape=(sequence_length, n_features), 
             return_sequences=True,
             kernel_initializer=kernel_initializer,
             recurrent_initializer=recurrent_initializer,
             bias_initializer=bias_initializer),
        Dropout(0.2, seed=42),
        LSTM(16,
             kernel_initializer=kernel_initializer,
             recurrent_initializer=recurrent_initializer,
             bias_initializer=bias_initializer),
        Dropout(0.2, seed=42),
        Dense(8, activation='relu',
              kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer),
        Dense(output_units, activation=activation,
              kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer)
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'], 
    )
    return model

def compute_catch22_features(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        X = X.values

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Get feature names from first valid time series
    try:
        feature_names = pycatch22.catch22_all(X[0].flatten())['names']
    except Exception as e:
        # Fallback if first series fails
        feature_names = [f"feature_{i}" for i in range(22)]

    all_features = np.zeros((len(X), len(feature_names)))

    for i, series in enumerate(X):
        try:
            series = series.flatten()
            features = pycatch22.catch22_all(series)['values']
            all_features[i] = features
        except Exception as e:
            print(f"Error processing series {i}: {str(e)}")
            all_features[i] = np.nan

    df_features = pd.DataFrame(all_features, columns=feature_names)
    return df_features


class Models:
    def __init__(self, model_name : str, 
                 X_train : np.array, y_train : np.array, seed : int = 42) -> None:

        valid_models = ['lstm', 'catch22', 'rocket', 'inception']
        if model_name not in valid_models:
            raise ValueError(f"Invalid model name. Choose from {valid_models}")
            
        self.model_name = model_name
        self.X_train = X_train
        
        self.original_classes = np.unique(y_train)
        self.n_classes = len(self.original_classes)

        # Store y_train as 2D for OneHot/LSTM, but we will flatten it for others
        self.y_train = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoded_y = None

        self.model = None
        self.catch22_train = None
        self.rocket_kernels = None

        self.seed = seed
        return

    # LSTM model ==============================================================================
    def train_lstm(self, epochs=30, batch_size=8, validation_split=0.2, verbose=False) -> None:
        n_features = self.X_train.shape[2]
        sequence_length = self.X_train.shape[1]

        self.encoded_y = self.encoder.fit_transform(self.y_train)

        model = create_lstm(sequence_length, n_features, self.n_classes, self.seed)
        target = self.encoded_y if self.n_classes > 2 else self.y_train

        history = model.fit(
            self.X_train, 
            target, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split,
            verbose=1 if verbose else 0
        )
        self.model = model
        
        if verbose:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('LSTM Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('LSTM Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.tight_layout()
            plt.show()
        return
    
    def index_to_original_label(self, indices: np.array) -> np.array:
        return np.array([self.original_classes[idx] for idx in indices])
    
    # Random Forest w/ Catch22 Features =======================================================
    def train_catch22(self) -> None:
        self.catch22_train = compute_catch22_features(self.X_train)
        self.model = RandomForestClassifier(random_state=self.seed)
        # Flatten y_train for sklearn
        self.model.fit(self.catch22_train, self.y_train.ravel())
        return

    # Rocket w/ Ridge Classifier ==============================================================
    def train_rocket(self, n_kernels=5000) -> None: 
        self.rocket_kernels = Rocket(num_kernels=n_kernels, random_state=self.seed) 
        self.rocket_kernels.fit(self.X_train) 

        X_train_transform = self.rocket_kernels.transform(self.X_train)
        self.model = RidgeClassifier(random_state=self.seed)
        self.model.fit(X_train_transform, self.y_train.ravel())
        return

    # InceptionTime (Aeon) ====================================================================
    def train_inception(self, n_epochs=100, batch_size=64, verbose=False) -> None:
        """
        Trains the InceptionTimeClassifier from Aeon.
        Note: Aeon expects (n_samples, n_channels, n_timepoints).
        Input X_train is (n_samples, n_timepoints, n_channels/features).
        We must transpose dim 1 and 2.
        """
        # Transpose: (N, T, C) -> (N, C, T)
        X_train_transposed = self.X_train.transpose(0, 2, 1)
        
        self.model = InceptionTimeClassifier(
            n_epochs=n_epochs,
            batch_size=batch_size,
            verbose=0 if not verbose else 1,
            random_state=self.seed
        )
        # InceptionTime expects 1D y array
        self.model.fit(X_train_transposed, self.y_train.ravel())
        return

    # Predictions ==============================================================================
    def predict(self, X_test : np.array) -> tuple:
        if(self.model_name == 'lstm'):  
            y_proba = self.model.predict(X_test, verbose=0)

            if self.n_classes == 2:
                    y_pred_indices = np.where(y_proba > 0.5, 1, 0).flatten()
            else:
                y_pred_indices = np.argmax(y_proba, axis=1)
            y_pred = self.index_to_original_label(y_pred_indices)
    
            if self.n_classes == 2 and y_proba.shape[1] == 1:
                y_proba = np.hstack([1-y_proba, y_proba])
            
        elif(self.model_name == 'catch22'):
            catch22_test = compute_catch22_features(X_test)
            y_pred = self.model.predict(catch22_test)
            y_proba = self.model.predict_proba(catch22_test)
        
        elif(self.model_name == 'rocket'): 
            X_test_transform  = self.rocket_kernels.transform(X_test)
            y_pred = self.model.predict(X_test_transform)
            y_proba = self.model.decision_function(X_test_transform)

        elif(self.model_name == 'inception'):
            # Transpose: (N, T, C) -> (N, C, T)
            X_test_transposed = X_test.transpose(0, 2, 1)
            y_pred = self.model.predict(X_test_transposed)
            y_proba = self.model.predict_proba(X_test_transposed)

        return y_pred, y_proba
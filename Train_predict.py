""" The objective of this script is to create functions training the different reweighting methods (binning, NN, GBRw) and giving weights.
This will assume that the input data is given as numpy arrays, already sorted in training and validation sets.
The output will be the trained model, which can be used to predict the weights. Also, since the different methods need a scaling constant,
for example, the NN method needs a scaling constant that is proportional to the ratio of events from the two different distributions,
the function will also return the scaling constant, which can be used to rescale the predicted weights."""

#imports 
import numpy as np
import pandas as pd
import tensorflow as tf
from hep_ml import reweight
from xgboost import XGBClassifier

## binning

def train_binning(original_train, target_train, n_bins, n_neighbours, original_train_weight=None, target_train_weight=None):
    """This function trains the binning reweighter given as arguments the training data, the number of bins and neighbours, 
    and the weights of the two distributions if they are given as arguments, otherwise it will assume that they are all 1s. 
    The output is the trained model."""
    if original_train_weight is None:
        original_train_weight = np.ones(original_train.shape[0])/original_train.shape[0]
    if target_train_weight is None:
        target_train_weight = np.ones(target_train.shape[0])/target_train.shape[0]
    model = reweight.BinsReweighter(n_bins=n_bins, n_neighs=n_neighbours)
    model.fit(original_train, target_train,
              original_train_weight/np.sum(original_train_weight),
              target_train_weight/np.sum(target_train_weight))
    return model


def predict_binning(model, original_val, original_val_weight=None):
    """This function returns the weights predicted by the binning reweighter given as argument
    the scaling is already included in the output, and is given by the ratio of the number of events in the two """
    weights = model.predict_weights(original_val, original_val_weight/np.sum(original_val_weight) if original_val_weight is not None else None)
    reweighting_scale = 1 / np.sum(weights)
    return weights * reweighting_scale

# NN reweighter

def train_NN(original_train, original_val,
             target_train, target_val, 
             n_layers, n_neurons, epochs, batch_size, activation_function = 'relu',
             original_train_weight=None, original_val_weight=None, target_train_weight=None, target_val_weight=None):
    
    """ This function trains a NN to distinguish between the "original" and "target" distributions. The outputs are the training history and the 
    trained model that can be used to give weights to the "original" distribution to make it look like the "target" distribution."""

    # Reformat the data for training

    X_train = np.concatenate((original_train, target_train), axis=0)
    Y_train = np.concatenate((tf.keras.utils.to_categorical(np.zeros(len(original_train)), num_classes=2), 
                              tf.keras.utils.to_categorical(np.ones(len(target_train)), num_classes=2)), axis=0)
    X_val = np.concatenate((original_val, target_val), axis=0)
    Y_val = np.concatenate((tf.keras.utils.to_categorical(np.zeros(len(original_val)), num_classes=2), 
                            tf.keras.utils.to_categorical(np.ones(len(target_val)), num_classes=2)), axis=0)
    
    # Create the model architecture
    
    model = tf.keras.Sequential()
    for _ in range(n_layers):
        model.add(tf.keras.layers.Dense(n_neurons, activation= activation_function))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], weighted_metrics = [tf.keras.metrics.AUC(name='auc')])

    #Ensure best generalization by stopping at a min val_loss.
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience = 10,
                              restore_best_weights=True)
    if original_train_weight is not None and target_train_weight is not None:
        train_weights = np.concatenate((original_train_weight/np.sum(original_train_weight), target_train_weight/np.sum(target_train_weight)), axis=0)
        target_weights = np.concatenate((original_val_weight/np.sum(original_val_weight), target_val_weight/np.sum(target_val_weight)), axis=0)
        history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, sample_weight=train_weights,
                             validation_data=(X_val, Y_val, target_weights), callbacks=[earlystopping], verbose = 0)
        
        print(f"There is {np.sum(train_weights[Y_train[:, 0] == 1])} normalized samples of class 0 \
              and {np.sum(train_weights[Y_train[:, 1] == 1])} normalized samples of class 1 in the training set.")
        train_scale_factor = np.sum(train_weights[Y_train[:, 1] == 1]) / np.sum(train_weights[Y_train[:, 0] == 1])
    else:
        original_train_weight = np.ones(original_train.shape[0])/original_train.shape[0]
        original_val_weight = np.ones(original_val.shape[0])/original_val.shape[0]
        target_train_weight = np.ones(target_train.shape[0])/target_train.shape[0]
        target_val_weight = np.ones(target_val.shape[0])/target_val.shape[0]
        train_weights = np.concatenate((original_train_weight/np.sum(original_train_weight), target_train_weight/np.sum(target_train_weight)), axis=0)
        target_weights = np.concatenate((original_val_weight/np.sum(original_val_weight), target_val_weight/np.sum(target_val_weight)), axis=0)
        history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, sample_weight=train_weights,
                             validation_data=(X_val, Y_val, target_weights), callbacks=[earlystopping], verbose = 0)
        
        print(f"There is {np.sum(train_weights[Y_train[:, 0] == 1])/ (np.sum(train_weights[Y_train[:, 0] == 1]) + np.sum(train_weights[Y_train[:, 1] == 1]))} normalized samples of class 0 \
              and {np.sum(train_weights[Y_train[:, 1] == 1])/ (np.sum(train_weights[Y_train[:, 0] == 1]) + np.sum(train_weights[Y_train[:, 1] == 1]))} normalized samples of class 1 in the training set.")
        train_scale_factor = np.sum(train_weights[Y_train[:, 1] == 1]) / np.sum(train_weights[Y_train[:, 0] == 1])

    return history, model, train_scale_factor


def predict_NN(original, model, train_scale_factor):

    """This function uses the trained model to return the weights used for the reweighting process.
        if you give weights for one distribution, please give weights for both (can be 1s)"""
    weights = model.predict(original, verbose = 0)
    return (1 / train_scale_factor) * weights[:, 1] / weights[:, 0]

## Hep_ml - GBRw reweighter

def train_GBR(original_train, target_train, n_estimators, learning_rate, max_depth, min_samples_leaf, loss_regularization,
               original_train_weight=None, target_train_weight=None):
    """This function trains a GBReweighter to reweight the 'original' distribution into the 'target' distribution. 
    The output is a model that can be used to give weights to the 'original' distribution to make it look like the 'target' distribution."""
    if original_train_weight is None:
        original_train_weight = np.ones(original_train.shape[0])/original_train.shape[0]
    if target_train_weight is None:
        target_train_weight = np.ones(target_train.shape[0])/target_train.shape[0]
    reweighter = reweight.GBReweighter(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, 
                min_samples_leaf=min_samples_leaf, loss_regularization=loss_regularization)
    reweighter.fit(original_train, target_train, original_train_weight/np.sum(original_train_weight), target_train_weight/np.sum(target_train_weight))
    
    return reweighter

def predict_GBR(original_train, target_train, reweighter, original_train_weight=None, target_train_weight=None):
    """This function uses the trained reweighter to return the weights used for the reweighting process
        Note that as the output of a GBReweighter is not a probability, a rescaling is necessary to match the target distribution"""
    weights = reweighter.predict_weights(original_train, original_train_weight)
    reweighting_scale = 1 / np.sum(weights)
    return reweighting_scale * weights

def train_XGB(original_train, original_val, target_train, target_val, 
              hparams = {"n_estimators": 100,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "subsample": 1, 
                    "gamma": 1,
                    "early_stopping_rounds": 10}, original_train_weight=None, original_val_weight=None, target_train_weight=None, target_val_weight=None, 
                    header = None):
    """This function trains an XGBReweighter to reweight the 'original' distribution into the 'target' distribution. 
    The output is a model that can be used to give weights to the 'original' distribution to make it look like the 'target' distribution."""
    if original_train_weight is None:
        original_train_weight = np.ones(original_train.shape[0])/original_train.shape[0]
    if target_train_weight is None:
        target_train_weight = np.ones(target_train.shape[0])/target_train.shape[0]
       
    X_train = np.concatenate((original_train, target_train), axis=0)
    Y_train = np.concatenate((np.zeros(len(original_train)), np.ones(len(target_train))), axis=0)
    X_val = np.concatenate((original_val, target_val), axis=0)
    Y_val = np.concatenate((np.zeros(len(original_val)), np.ones(len(target_val))), axis=0)

    if header is not None and "Mode_v2" in header:
        print("Mode_v2 is in the header, converting it to categorical.")
        X_train_df = pd.DataFrame(X_train, columns=header)
        X_val_df   = pd.DataFrame(X_val, columns=header)

        X_train_df["Mode_v2"] = X_train_df["Mode_v2"].astype("category")
        X_val_df["Mode_v2"]   = X_val_df["Mode_v2"].astype("category")

        bst = XGBClassifier(objective='binary:logistic',
                        eval_metric=['logloss', 'auc'],
                        scale_pos_weight = len(original_train) / len(target_train),
                        enable_categorical=True,
                        **hparams)

        bst.fit(
            X_train_df, Y_train,
            eval_set=[(X_val_df, Y_val), (X_train_df, Y_train)],
            verbose=False
        )

    else:
        bst = XGBClassifier(objective='binary:logistic',
                        eval_metric=['logloss', 'auc'],
                        scale_pos_weight = len(original_train) / len(target_train),
                        **hparams)

        bst.fit(
            X_train, Y_train,
            eval_set=[(X_val, Y_val), (X_train, Y_train)],
            verbose=False
        )

    return bst

def predict_XGB(original, model, header = None):
    """This function uses the trained model to return the weights used for the reweighting process."""
    if header is not None and "Mode_v2" in header:
        original_df = pd.DataFrame(original, columns=header)
        original_df["Mode_v2"] = original_df["Mode_v2"].astype("category")
        predictions = model.predict_proba(original_df, iteration_range=(0, model.best_iteration + 1))
    else:
        predictions = model.predict_proba(original, iteration_range=(0, model.best_iteration + 1))
    weights = predictions[:, 1] / predictions[:, 0]
    reweighting_scale = 1 / np.sum(weights)
    return reweighting_scale * weights

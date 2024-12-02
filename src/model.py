import os
import pandas as pd
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import json
import matplotlib.pyplot as plt
from config import DATA_PATH, MODELS_PATH
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Define the objective function for hyperparameter tuning
def objective(params):
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': params['booster'],
        'eta': params['eta'],
        'max_depth': int(params['max_depth']),
        'min_child_weight': params['min_child_weight'],
        'subsample': params['subsample'],
        'colsample_bytree': params['colsample_bytree'],
        'gamma': params['gamma'],
        'n_estimators': int(params['n_estimators']),
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Train XGBoost model
    cv_result = xgb.cv(params, dtrain,
                       num_boost_round=params['n_estimators'],
                       nfold=5, stratified=True, early_stopping_rounds=10, 
                       seed=42, verbose_eval=False)
    
    # Return the loss from cross-validation
    loss = cv_result['test-logloss-mean'].min()
    return {'loss': loss, 'status': STATUS_OK}

# Define the search space for hyperparameters
def get_space():
    space = {
        'booster': hp.choice('booster', ['gbtree', 'dart']),
        'eta': hp.uniform('eta', 0.01, 0.3),
        'max_depth': hp.quniform('max_depth', 3, 15, 1),
        'min_child_weight': hp.uniform('min_child_weight', 1, 10),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'gamma': hp.uniform('gamma', 0, 5),
        'n_estimators': hp.quniform('n_estimators', 50, 250, 10),
    }
    return space

def model_tune():
    space = get_space()
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=50, trials=trials, rstate=None)
    best_params = {'objective': 'binary:logistic',
                   'eval_metric': 'logloss',
                   'booster': ['gbtree', 'dart'][best['booster']],
                   'eta': best['eta'],
                   'max_depth': int(best['max_depth']),
                   'min_child_weight': best['min_child_weight'],
                   'subsample': best['subsample'],
                   'colsample_bytree': best['colsample_bytree'],
                   'gamma': best['gamma'],
                   'n_estimators': int(best['n_estimators']),
                  }

    return best_params

def predict(resolution,
            training_data,
            predict_data):
    """
    This function predicts forest locations

    This function load tunned model parameteres ans train 

    Parameters
    ----------
    resolution : _type_
        _description_
    training_data : _type_
        _description_
    predict_data : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    model_params_path = os.path.join(MODELS_PATH, f'{resolution}_m_model_params.json')
    with open(model_params_path) as pf:
        best_params = json.load(pf)
    X_train = training_data.iloc[:, :-1]
    y_train = training_data.iloc[:, -1]
    try:
        model = xgb.XGBClassifier()
        model.load_model(os.path.join(MODELS_PATH, f'{res}m_model.bin'))
    except xgb.XGBoostError as e:
        print(e, "Traning a model from scratch...")
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)
    pred = model.predict(predict_data)
    prob = model.predict_proba(predict_data)
    return pred, prob


if __name__ == "__main__":

    res = 10
    to_save = True
    # Load dataset
    data = pd.read_csv(os.path.join(DATA_PATH, f"{res}_m_res.csv"))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], 
                                                            data['target'], 
                                                            test_size=.2, 
                                                            random_state=42)    
    # Train final model using the best hyperparameters
    try:
        with open(f'../models/{res}_m_model_params.bin', 'r') as pf:
            best_params = json.load(pf)
    except FileNotFoundError:
        # Run hyperparameter optimization
        best_params = model_tune()
        os.makedirs('../models', exist_ok=True)
        with open(f'../models/{res}_m_model_params.bin', 'w') as pf:
            json.dump(best_params, pf,indent=4)

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)
    if to_save:
        final_model.fit(data.iloc[:, :-1], data['target'])
        final_model.save_model(os.path.join(MODELS_PATH, f'{res}m_model.json'))
    
    # Evaluate the model
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)
    eval_results = y_test.reset_index().loc[:, "target"]
    eval_results = pd.concat([eval_results, pd.DataFrame(y_pred), pd.DataFrame(y_prob)], axis=1)
    eval_results.columns = ["y_test", "y_pred_NT", "y_pred_YT", "y_prob"]
    eval_results.to_csv(f"{res}_m_results_eval.csv")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Train set size: {len(y_train)} (Tree size: {sum(y_train)}, {sum(y_train)/len(y_train): .2f})")
    print(f"Test set size: {len(y_test)} (Tree size: {sum(y_test)}, {sum(y_test)/len(y_test): .2f})")
    f1 = f1_score(y_test, y_pred)
    print("Final model accuracy:", accuracy)
    print("final F1 score:", f1)
    # This part of the code is evaluating the performance of the final model by generating and
    # displaying a confusion matrix. Here's a breakdown of what each step does:
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Trees', 'Trees'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

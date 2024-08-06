import mlflow 

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


def objective(trial, X_train, y_train, experiment_id):
    """
    Optimize hyperparameters for a regressor using Optuna.

    Parameters:
    -----------
    trial : optuna.trial.Trial
        A trial is a process of evaluating an objective function.
    X_train : pandas.DataFrame
        Input features for training.
    y_train : pandas.Series
        Target variable for training.
    experiment_id : int
        ID of the MLflow experiment where results will be logged.

    Returns:
    --------
    float
        MAE of the regressor after cross-validation.
    """
    # Comienza el run de MLflow. Este run debería ser el hijo del run padre, 
    # así se anidan los diferentes experimentos.
    with mlflow.start_run(experiment_id=experiment_id, 
                          run_name=f"Trial: {trial.number}", nested=True):

        # Sugiere un valor categórico para model_name de la lista de modelos proporcionada. 
        # Esto significa que Optuna probará cada uno de estos modelos durante la optimización.
        model_name = trial.suggest_categorical('model', ['LinearRegression', 
                                                         'Ridge',
                                                         'DecisionTree',
                                                         'XGBoost' ])
   
        params = {}

        # Dependiendo del modelo seleccionado, se definen los hiperparámetros específicos

        # Para el modelo de regresión lineal no hay hiperparámetros que ajustar
        if model_name == 'LinearRegression':
            params["model"] = 'LinearRegression'
            model = LinearRegression()

        # Para la regresión Ridge, se ajusta el parámetro alpha
        elif model_name == 'Ridge':
            alpha = trial.suggest_float('ridge_alpha', 1e-3, 10.0)
            params["model"] = 'Ridge'
            params["alpha"] = alpha
            model = Ridge(alpha=alpha)
            
        # Para el modelo SVR, se ajustan los parámetros C, gamma y kernel.
        elif model_name == 'SVR':
            C = trial.suggest_float('svr_C', 0.1, 10.0)
            gamma = trial.suggest_float('svr_gamma', 0.1, 10.0)
            kernel = trial.suggest_categorical('svr_kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            params["model"] = 'SVR'
            params["C"] = C
            params["gamma"] = gamma
            params["kernel"] = kernel
            model = SVR(C=C, gamma=gamma, kernel=kernel)
            
        # Para el árbol de decisión, se ajusta el parámetro max_depth
        elif model_name == 'DecisionTree':
            max_depth = trial.suggest_int('dt_max_depth', 1, 20)
            params["model"] = 'DecisionTree'
            params["max_depth"] = max_depth
            model = DecisionTreeRegressor(max_depth=max_depth)

        # Para el bosque aleatorio, se ajustan los parámetros n_estimators y max_depth    
        elif model_name == 'RandomForest':
            n_estimators = trial.suggest_int('rf_n_estimators', 10, 200)
            max_depth = trial.suggest_int('rf_max_depth', 1, 20)
            params["model"] = 'RandomForest'
            params["max_depth"] = max_depth
            params["n_estimators"] = n_estimators
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

        # Para el modelo AdaBoost, se ajustan los parámetros n_estimators y learning_rate    
        elif model_name == 'AdaBoost':
            n_estimators = trial.suggest_int('ab_n_estimators', 10, 200)
            learning_rate = trial.suggest_float('ab_learning_rate', 1e-3, 1.0)
            params["model"] = 'AdaBoost'
            params["learning_rate"] = learning_rate
            params["n_estimators"] = n_estimators
            model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate)

        # Para el modelo XGBoost, se ajustan los parámetros n_estimators, learning_rate y max_depth 
        elif model_name == 'XGBoost':
            n_estimators = trial.suggest_int('xgb_n_estimators', 10, 200)
            learning_rate = trial.suggest_float('xgb_learning_rate', 1e-3, 1.0)
            max_depth = trial.suggest_int('xgb_max_depth', 1, 20)
            params["model"] = 'XGBoost'
            params["learning_rate"] = learning_rate
            params["n_estimators"] = n_estimators
            params["max_depth"] = max_depth
            model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, eval_metric='logloss')

        # Realizamos validación cruzada y calculamos el mae
        score = cross_val_score(model, X_train, y_train.to_numpy().ravel(), cv=5, scoring='neg_mean_absolute_error').mean()
        
        # Log los hiperparámetros a MLflow
        mlflow.log_params(params)
        
        # Y el MAE de la validación cruzada.
        mlflow.log_metric("MAE", -score)

    return -score

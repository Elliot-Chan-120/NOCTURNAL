import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, make_scorer, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

from pathlib import Path
import pickle as pkl
from b01_utility import *


class ModelBuilder:
    """
    Builds machine learning model based on data produced by the DataWorker class
    \n cleans the data once the class is initialized, ready for machine learning model generation, assessment and comparison against other models
    """
    def __init__(self, model_name):
        self.model_name = model_name

        # Verify that all appropriate config keys and folders are present
        # Load cfg
        self.cfg = validate_config()

        # NAVIGATION
        self.bioact_folder_path = Path(self.cfg['database'])
        self.fpdf_path = self.bioact_folder_path / self.cfg['fingerprintdf_4']  # fingerprint filepath

        # make folder to hold ML model and its settings
        self.model_storage = Path(self.cfg['model_folder'])  # model folder
        self.model_path = self.model_storage / f"{self.model_name}"
        self.model_path.mkdir(exist_ok=True)

        # make folder to hold the numerous assessment files
        self.assess_storage = Path(self.cfg['assessments'])     # graph folder
        self.assess_path = self.assess_storage / f"{self.model_name}_assessment_files"
        self.assess_path.mkdir(exist_ok=True)

        # Data - model and settings
        self.best_params = None
        self.model_type = self.cfg['ml_model_type']
        self.model = None
        self.selected_cols = []

    # =====[Performs the Model Building Function Sequence]=====
    def build(self):
        # ===== FPDF_LOADING =====
        # generate the input feature axes
        try:
            df = pd.read_csv(self.fpdf_path, index_col=0)
        except FileNotFoundError:
            raise ModelBuilderError(f"Fingerprint + pIC50 DataFrame File Missing")
        except Exception as e:
            raise ModelBuilderError(f"Failed to load fingerprint dataframe: {e}")

        X = df.drop('pIC50', axis=1)
        Y = df.pIC50

        if 'pIC50' not in df.columns:
            raise DataProcessingError(
                f"Required column 'pIC50' not found in {self.cfg['bioactivity_folder']} file: {self.cfg['fingerprintdf_4']}")

        # ========= DATA CLEANING =========
        # remove low variance features
        selection = VarianceThreshold(threshold=(.8 * (1 - 0.8)))
        selection.fit(X)

        # save the columns that were kept
        kept = selection.get_support()
        self.selected_cols = X.columns[kept].to_list()

        # carry on with the data cleaning
        X = selection.transform(X)

        # split data
        tts_cfg = self.cfg['train_test_split']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, **tts_cfg)

        if self.model_type == 'RandomForestRegressor':
            self.ens_randomfor(X_train, X_test, y_train, y_test)
        elif self.model_type == 'XGBoosting':
            self.ens_boost(X_train, X_test, y_train, y_test)
        elif self.model_type == 'Stacking':
            self.ens_stack(X_train, X_test, y_train, y_test)
        else:
            raise ConfigurationError(f"{self.model_type} is not a valid config option")

    # =====[RANDOMFOREST]=====
    def ens_randomfor(self, X_train, X_test, y_train, y_test):
        """
        Build a singular regression model using random forests
        """
        print("Training RandomForestRegressor")
        # ========= OPTIMIZE RESULTS ========= (and save as self.best_params)
        try:
            self.optimize_rf(X_train, y_train)
        except Exception as e:
            raise ModelBuilderError(f"Hyperparameter optimization failed: {e}")


        # ========= CREATE OPTIMIZED MODEL =========
        # build regression model using random forests and optimized parameters
        try:
            if self.cfg['use_best_params'] is True:
                print("Using Optimized Parameters")
                self.model = RandomForestRegressor(**self.best_params, random_state=42)
            else:
                default_model_cfg = self.cfg['random_forest_params']
                self.model = RandomForestRegressor(**default_model_cfg)
        except Exception as e:
            raise ModelBuilderError(f"Failed to create RandomForestRegressor: {e}")

        # ===[train + eval RandomForestRegressor model]===
        self.train_eval(X_train, X_test, y_train, y_test)

    # =====[BOOSTING]=====
    def ens_boost(self, X_train, X_test, y_train, y_test):
        """Trains XGBoost on the Fingerprint + pIC50 DataFrame"""
        print("Training XGBoost Regressor")
        try:
            self.optimize_xgb(X_train, y_train)
        except Exception as e:
            raise ModelBuilderError(f"Hyperparameter optimization failed: {e}")

        try:
            if self.cfg['use_best_params'] is True:
                print("Using Optimized Parameters")
                self.model = XGBRegressor(**self.best_params, random_state=42)
            else:
                default_model_cfg = self.cfg['xgb_params']
                self.model = XGBRegressor(**default_model_cfg)
        except Exception as e:
            raise ModelBuilderError(f"Failed to create XGBRegressor: {e}")

        self.train_eval(X_train, X_test, y_train, y_test)

    # =====[STACKING]=====
    def ens_stack(self, X_train, X_test, y_train, y_test):
        """
        Utilizes Stacking with rf, xgb, and svr as base models + Ridge as the meta-learner
        """
        print("Training Stacking Regressor")

        # Create base estimators good for molecular fingerprints - non-linear relationships
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
            ('svr', SVR(kernel='rbf', gamma='scale'))
        ]

        # Create meta-learner w/ Ridge regression
        meta_estimator = Ridge(alpha=1.0)

        try:
            self.model = StackingRegressor(
                estimators = estimators,
                final_estimator= meta_estimator,
                cv = 5,
                n_jobs=-1
            )
        except Exception as e:
            raise ModelBuilderError(f"Failed to create StackingRegressor: {e}")

        self.train_eval(X_train, X_test, y_train, y_test)


# TRAINS AND EVALUATES MODEL -> SAVES IT
    def train_eval(self, X_train, X_test, y_train, y_test):
        # ========= TRAINING EVALUATION =========
        # Evaluate optimized model with cross-validation on training set
        scoring = {
            'r2': 'r2',
            'rmse': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': 'neg_mean_absolute_error'
        }

        try:
            train_results = cross_validate(
                self.model, X_train, y_train, cv=5,
                scoring = scoring, return_estimator=True)
        except ValueError as e:
            raise ModelBuilderError(f"Cross-validation failed w/ ValueError: {e}")
        except Exception as e:
            raise ModelBuilderError(f"Cross-validation failed: {e}")

        # get file evaluation on trained results
        self.evaluate_cv_model(train_results, file_suffix='Cross_Validation')

        # train model on training data
        try:
            self.model.fit(X_train, y_train)
        except Exception as e:
            raise ModelBuilderError(f"Failed to train model: {e}")


        # ========= FINAL EVALUATION =========
        # evaluation on untouched 'test' set
        try:
            y_predict = self.model.predict(X_test)
            final_results = {
                'test_r2': r2_score(y_test, y_predict),
                'test_rmse':  np.sqrt(mean_squared_error(y_test, y_predict)),
                'test_mae': mean_absolute_error(y_test, y_predict)
            }

            # check for poor performance
            if final_results['test_r2'] < 0.3:
                print("Warning: Poor R^2 performance")
        except Exception as e:
            raise ModelBuilderError(f"Model Evaluation Failed: {e}")

        # get file evaluation on final results
        self.evaluate_final_model(final_results, file_suffix="Final")

        # get regression plot for experimental vs. predicted pIC50s
        self.regression_plot(exp=y_test, pred=y_predict)

        # feature importance data visualization - check if model has it
        if hasattr(self.model, 'feature_importances_'):
            self.importance_plot(self.model.feature_importances_, self.selected_cols)
        elif hasattr(self.model, 'estimators_') and hasattr(self.model.estimators_[0], 'feature_importances_'):
            # we will get the feature importances from the first model in the stack
            self.importance_plot(self.model.estimators_[0].feature_importances_, self.selected_cols)


        # ========= SAVE MODEL AND SETTINGS =========
        # Save the selected feature names / settings as dataframe to be used later
        if self.cfg.get("auto_save_model", False):
            print("Auto-saving")
            self.save()
        else:
            x = True
            while x:
                answer = input("Save Model? y/n: ")
                if answer.lower() == 'y':
                    self.save()
                    x = False
                elif answer.lower() == 'n':
                    print("[Model Rejected]")
                    x = False
                else:
                    print("invalid character submitted: please input either 'y' or 'n' ")


# OPTIMIZATION FUNCTIONS
    def optimize_rf(self, copyX, copyY):
        """use GridSearchCV to find optimal hyperparameters for RandomForestRegressor"""
        print("Optimizing RandomForest Hyperparameters")

        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]}

        # baseline model
        basemodel = RandomForestRegressor(random_state=42)

        grid_cfg = self.cfg['grid_search']
        search = GridSearchCV(**grid_cfg, param_grid=param_grid ,estimator=basemodel)

        search.fit(copyX, copyY)

        self.best_params = search.best_params_

        print(f"Best Parameters: {self.best_params}")
        print(f"Best RMSE: {np.sqrt(-search.best_score_):.4f}")

        results_df = pd.DataFrame(search.cv_results_)
        results_df.to_csv(self.assess_path / f"{self.model_name}_gridsearch_results.csv")

        return self.best_params

    def optimize_xgb(self, copyX, copyY):
        """use GridSearchCV to find optimal hyperparams for XGBRegressor"""
        print("Optimizing XGBoost Hyperparameters")

        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 6, 9],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'gamma': [0, 0.1]
        }

        base_model = XGBRegressor(random_state=42)

        grid_cfg = self.cfg['grid_search']
        search = GridSearchCV(**grid_cfg, param_grid=param_grid, estimator=base_model)

        search.fit(copyX, copyY)

        self.best_params = search.best_params_

        results_df = pd.DataFrame(search.cv_results_)
        results_df.to_csv(self.assess_path / f"{self.model_name}_gridsearch_results.csv")

        return self.best_params


# EVALUATION FUNCTIONS
    def evaluate_cv_model(self, results, file_suffix=None, vocal=True):
        r2 = results['test_r2']
        rmse = results['test_rmse']
        mae = -results['test_mae']  # sklearn uses "neg_mean_absolute_error"

        if vocal:
            print(f"{file_suffix} assessment")
            print(f"Mean R^2 Score: {np.mean(r2):.4f}")
            print(f"Mean RMSE: {np.mean(rmse):.4f}")
            print(f"Mean MAE: {np.mean(mae):.4f}")
            print("-" * 40)

        content = (
            f"{self.model_name} {file_suffix} Assessment"
            f"\n{self.model_type}"
            f"\nOptimal Hyperparameters: {self.best_params}"
            f"\nMean R^2 Score: {np.mean(r2)} "
            f"\nMean RMSE: {np.mean(rmse)} "
            f"\nMean MAE: {np.mean(mae)}"
        )
        with open(self.assess_path / f"{self.model_name}.{file_suffix}_assessment.txt", 'w') as file:
            file.write(content)

    def evaluate_final_model(self, results, file_suffix=None, vocal=True):
        r2 = results['test_r2']
        rmse = results['test_rmse']
        mae = results['test_mae']  # sklearn uses "neg_mean_absolute_error"

        if vocal:
            print(f"{file_suffix} assessment")
            f"\n{self.model_type}"
            print(f"Mean R^2 Score: {r2:.4f}")
            print(f"Mean RMSE: {rmse:.4f}")
            print(f"Mean MAE: {mae:.4f}")
            print("-" * 40)

        content = (
            f"{self.model_name} {file_suffix} Assessment"
            f"\n{self.model_type}"
            f"\nOptimal Hyperparameters: {self.best_params}"
            f"\nR^2 Score: {r2:.4f}"
            f"\nRMSE: {rmse:.4f}"
            f"\nMAE: {mae:.4f}"
        )

        with open(self.assess_path / f"{self.model_name}.{file_suffix}_assessment.txt", 'w') as file:
            file.write(content)


# DATA VISUALIZATION FUNCTIONS
    def regression_plot(self, exp, pred):
        sns.set_theme(color_codes=True)
        sns.set_style("white")

        ax = sns.regplot(x=exp, y=pred, scatter_kws={'alpha': 0.4})
        ax.set_xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
        ax.set_ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
        plt.title(f"Experimental vs Predicted pIC50 values - {self.model_name}")
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 12)
        ax.figure.set_size_inches(5, 5)
        plt.savefig(self.assess_path / f"{self.model_name}_regression_plot.pdf")
        plt.close()

    def importance_plot(self, importances, feature_IDs):
        feature_count = self.cfg["n_features"]

        important_df = pd.DataFrame({
            'Feature': feature_IDs,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(feature_count)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=important_df,
                    x='Importance', y='Feature',
                    hue='Feature',
                    legend=False,  # add hue for color variation
                    dodge=False,   # avoid bars being side by side
                    palette='viridis'
                    )
        plt.title(f"Top {feature_count} Feature Importances - {self.model_name}")
        plt.tight_layout()

        plt.savefig(self.assess_path / f"{self.model_name}_feature_importance_plot.pdf")
        plt.close()


# SAVE FUNC.
    def save(self):
        """Save the machine learning model and its settings for predictions"""
        if self.model is None:
            raise ModelBuilderError(f"No model to save")

        print("Saving current model...")

        try:
            path = Path(self.model_path / f"{self.model_name}.pkl")
            with open(path, 'wb') as handle:  # write and save model using 'wb' - write binary
                pkl.dump(self.model, handle)

            print("[Model Saved]")

            print("Saving model settings...")
            with open(self.model_path / f"{self.model_name}_settings.txt", 'w') as f:
                for col in self.selected_cols:
                    f.write(f"{col}\n")

        except Exception as e:
            raise ModelBuilderError(f"Model save failure: {e}")

        print("[Model Settings Saved]")

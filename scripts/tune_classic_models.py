import os
import pickle
from abc import ABC
import seaborn as sns
from sklearn.svm import SVR
from optuna.trial import Trial
import matplotlib.pyplot as plt
from Research.util.tune import *
from sklearn.svm import LinearSVC
from imblearn.over_sampling import *
from Research.util.evaluate import *
from sklearn.linear_model import Lars
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from Research.util.data_loading import *
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

"""# Data"""
def load_data(path, channels, duration):
    data = load_features(path, channels, duration)
    labels = ["categorical", "valance", "arousal", "dominance", "positive"]
    y = data[labels]
    X = data.drop(columns=["session", "time"] + labels)
    X["dominate_window"] = X["dominate_window"].astype('category').cat.codes.astype(np.uint32)
    X["dominate_task"] = X["dominate_task"].astype('category').cat.codes.astype(np.uint32)
    for column in X:
        if "Turn" in column or \
                "count" in column or \
                "switches" in column or \
                "mode_key" in column or \
                "Direction" in column or \
                "unique_events" in column or \
                "error_corrections" in column:
            X[column] = X[column].astype(np.uint32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
    return X, X_train, X_test, y, y_train, y_test


"""# Models"""
"""## models tuning"""
class SklearnTuner(Tuner, ABC):
    def __init__(self, name, X_train, X_test, y_train, y_test, labeling_type):
        super().__init__(project="capi", name=name, monitor="mujority-model-dif_sum",
                         default_params={})
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train[labeling_type]
        self.y_test = y_test[labeling_type]
        self.best_score = float('inf')
        self.best_model = None
        self.weights = None
        self.direction = "minimize"

        baseline_preds = self._get_baseline_prediction()
        self.majority_eval = self._get_prediction_score(baseline_preds, self.y_test)

    def objective(self, callbacks=()):
        model = self.build_model()
        model.fit(self.X_train, self.y_train)
        return self.__score_model(model)

    def run(self, trials=100, use_logger=False):
        super().run(trials, self.direction, use_logger)

    def get_best_results(self):
        eval = self.__get_model_results(self.best_model)
        eval["model"] = self.name
        return eval

    def __score_model(self, model):
        eval = self.__get_model_results(model)
        score = 0
        for param in eval:
            score += self.weights[param] * (self.majority_eval[param] - eval[param])

        best = False
        if self.direction == "minimize":
            if score < self.best_score:
                best = True
        elif self.direction == "maximize":
            if score > self.best_score:
                best = True
        if best:
            self.best_score = score
            self.best_model = model
        return score

    def __get_model_results(self, model):
        predictions = model.predict(self.X_test)
        eval = self._get_prediction_score(predictions, self.y_test)
        return eval

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def show_best_results(self, path_to_save=None):
        pass

    @abstractmethod
    def _get_baseline_prediction(self):
        pass

    @abstractmethod
    def _get_prediction_score(self, y_pred, y_true):
        pass

"""### classification"""
class ClassificationTuner(SklearnTuner, ABC):
    def __init__(self, name, X_train, X_test, y_train, y_test, labeling_type, weights):
        super().__init__(name, X_train, X_test, y_train, y_test, labeling_type)
        self.weights = weights
        self.best_score = float('inf')
        self.direction = "minimize"

    def show_best_results(self, path_to_save=None):
        scores = pd.DataFrame(columns=['model', 'accuracy', 'balanced accuracy',
                                       'precision', 'recall', 'f1 score'])
        majority_eval = self.majority_eval
        majority_eval["model"] = "majority rule"
        scores = scores.append(majority_eval, ignore_index=True)

        best_results = self.get_best_results()
        scores = scores.append(best_results, ignore_index=True)

        if path_to_save is not None:
            os.mkdir(f"{path_to_save}/{self.name}")
            with open(f'{path_to_save}/{self.name}/pickled_scores.df', 'wb') as res_file:
                pickle.dump(scores, res_file)
            scores.to_html(f'{path_to_save}/{self.name}/scores.html')
            with open(f'{path_to_save}/{self.name}/model.md', 'wb') as model_file:
                pickle.dump(self.best_model, model_file)
            return

        return scores

    def _get_baseline_prediction(self):
        mode_label = self.y_train.mode()[0]
        baseline_prediction = [mode_label for _ in range(len(self.y_test))]
        return baseline_prediction

    def _get_prediction_score(self, y_pred, y_true):
        return dict(clf_scores(np.array(y_true), np.array(y_pred)))

    @abstractmethod
    def build_model(self):
        pass
"""#### linear svm"""
class LinearSVMTuner(ClassificationTuner):
    def __init__(self, X_train, X_test, y_train, y_test, labeling_type,
                 weights={"accuracy": 1, "balanced accuracy": 1, "precision": 1,
                          "recall": 1, "f1 score": 1}):
        super().__init__("linear_svm", X_train, X_test, y_train, y_test, labeling_type, weights)

    def build_model(self):
        model = make_pipeline(MinMaxScaler(),
                              LinearSVC(penalty=self.params["penalty"],
                                        loss="squared_hinge",
                                        dual=False,
                                        C=self.params["C"],
                                        multi_class=self.params["multi_class"],
                                        fit_intercept=True,
                                        intercept_scaling=self.params["intercept_scaling"],
                                        class_weight=None,
                                        verbose=0,
                                        random_state=42,
                                        max_iter=50000))
        return model

    def update_params(self, trial: Trial):
        self.params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
        self.params["multi_class"] = trial.suggest_categorical("multi_class", ["ovr", "crammer_singer"])
        self.params["C"] = trial.suggest_float("C", 0, 1)
        self.params["intercept_scaling"] = trial.suggest_float("intercept_scaling", 0, 10)

    def run(self, trials=100, use_logger=False):
        super().run(trials, use_logger)
"""#### KNeighbors"""
class KNeighborsTuner(ClassificationTuner):
    def __init__(self, X_train, X_test, y_train, y_test, labeling_type,
                 weights={"accuracy": 1, "balanced accuracy": 1, "precision": 1,
                          "recall": 1, "f1 score": 1}):
        super().__init__("knn", X_train, X_test, y_train, y_test, labeling_type, weights)

    def build_model(self):
        model = KNeighborsClassifier(n_neighbors=self.params["n_neighbors"],
                                     weights=self.params["weights"],
                                     algorithm=self.params["algorithm"],
                                     p=self.params["p"],
                                     metric=self.params["metric"])
        return model

    def update_params(self, trial: Trial):
        self.params["n_neighbors"] = trial.suggest_int("n_neighbors", 1, 20)
        self.params["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
        self.params["algorithm"] = trial.suggest_categorical("algorithm", ["ball_tree", "kd_tree", "brute"])
        self.params["p"] = trial.suggest_int("p", 1, 5)
        self.params["metric"] = trial.suggest_categorical("metric",
                                                          ["euclidean", "manhattan", "chebyshev", "minkowski"])

    def run(self, trials=100, use_logger=False):
        super().run(trials, use_logger)
"""#### DecisionTrees"""
class DecisionTreeTuner(ClassificationTuner):
    def __init__(self, X_train, X_test, y_train, y_test, labeling_type,
                 weights={"accuracy": 1, "balanced accuracy": 1, "precision": 1,
                          "recall": 1, "f1 score": 1}):
        super().__init__("decision_tree", X_train, X_test, y_train, y_test, labeling_type, weights)

    def build_model(self):
        model = DecisionTreeClassifier(criterion=self.params["criterion"],
                                       splitter=self.params["splitter"],
                                       max_depth=self.params["max_depth"],
                                       min_samples_split=self.params["min_samples_split"],
                                       min_samples_leaf=self.params["min_samples_leaf"],
                                       max_features=self.params["max_features"],
                                       random_state=42,
                                       max_leaf_nodes=self.params["max_leaf_nodes"],
                                       class_weight=self.params["class_weight"],
                                       ccp_alpha=self.params["ccp_alpha"])
        return model

    def update_params(self, trial: Trial):
        self.params["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy"])
        self.params["splitter"] = trial.suggest_categorical("splitter", ["best", "random"])
        self.params["max_depth"] = trial.suggest_int("max_depth", 3, 40)
        self.params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 6)
        self.params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 2, 6)
        self.params["max_features"] = trial.suggest_int("max_features", 1, len(self.X_train.columns))
        self.params["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 2, 30)
        self.params["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])
        self.params["ccp_alpha"] = trial.suggest_float("ccp_alpha", 0, 1)

    def run(self, trials=300, use_logger=False):
        super().run(trials, use_logger)
"""#### Logistic Regression"""
class LogisticRegressionTuner(ClassificationTuner):
    def __init__(self, X_train, X_test, y_train, y_test, labeling_type,
                 weights={"accuracy": 1, "balanced accuracy": 1, "precision": 1,
                          "recall": 1, "f1 score": 1}):
        super().__init__("logistic_regression", X_train, X_test, y_train, y_test, labeling_type, weights)

    def build_model(self):
        model = make_pipeline(MinMaxScaler(),
                              LogisticRegression(penalty=self.params["penalty"],
                                                 dual=False,
                                                 tol=self.params["tol"],
                                                 C=self.params["C"],
                                                 fit_intercept=self.params["fit_intercept"],
                                                 intercept_scaling=self.params["intercept_scaling"],
                                                 class_weight=self.params["class_weight"],
                                                 random_state=42,
                                                 solver=self.params["solver"],
                                                 max_iter=25000,
                                                 multi_class=self.params["multi_class"]))
        return model

    def objective(self, callbacks: List[Callback] = ()) -> float:
        if (self.params["solver"] in ["newton-cg", "lbfgs", "saga", "sag"] and self.params["penalty"] == "l1") or \
                (self.params["solver"] == "liblinear" and self.params["penalty"] == None) or \
                (self.params["solver"] == "liblinear" and self.params["multi_class"] == "multinomial") or \
                (self.params["penalty"] == "elasticnet" and self.params["solver"] != "saga"):
            return float('inf')
        return super().objective(callbacks)

    def update_params(self, trial: Trial):
        self.params["penalty"] = trial.suggest_categorical("penalty", ["l1", "l2"])
        self.params["tol"] = trial.suggest_float("tol", 0, 0.01)
        self.params["C"] = trial.suggest_float("C", 0, 1)
        self.params["fit_intercept"] = trial.suggest_categorical("fit_intercept", [True, False])
        self.params["intercept_scaling"] = trial.suggest_float("intercept_scaling", 0, 2)
        self.params["class_weight"] = trial.suggest_categorical("class_weight", ["balanced", None])
        self.params["solver"] = trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"])
        self.params["multi_class"] = trial.suggest_categorical("multi_class", ["ovr", "multinomial"])

    def run(self, trials=500, use_logger=False):
        super().run(trials, use_logger)
"""#### random forest"""
class RandomForestTuner(ClassificationTuner):
    def __init__(self, X_train, X_test, y_train, y_test, labeling_type,
                 weights={"accuracy": 1, "balanced accuracy": 1, "precision": 1,
                          "recall": 1, "f1 score": 1}):
        super().__init__("random_forest", X_train, X_test, y_train, y_test, labeling_type, weights)

    def build_model(self):
        model = RandomForestClassifier(n_estimators=150,
                                       criterion=self.params["criterion"],
                                       min_samples_split=self.params["min_samples_split"],
                                       max_features=self.params["max_features"],
                                       bootstrap=self.params["bootstrap"],
                                       random_state=42,
                                       ccp_alpha=self.params["ccp_alpha"])
        return model

    def update_params(self, trial: Trial):
        self.params["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy"])
        self.params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 5)
        self.params["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        self.params["bootstrap"] = trial.suggest_categorical("bootstrap", [True, False])
        self.params["ccp_alpha"] = trial.suggest_float("ccp_alpha", 0, 1)

    def run(self, trials=100, use_logger=False):
        super().run(trials, use_logger)
"""#### Bagging"""
class BaggingTuner(ClassificationTuner):
    def __init__(self, X_train, X_test, y_train, y_test, labeling_type,
                 weights={"accuracy": 1, "balanced accuracy": 1, "precision": 1,
                          "recall": 1, "f1 score": 1}):
        super().__init__("bagging", X_train, X_test, y_train, y_test, labeling_type, weights)

    def build_model(self):
        model = BaggingClassifier(n_estimators=100,
                                  max_samples=self.params["max_samples"],
                                  max_features=self.params["max_features"],
                                  bootstrap=self.params["bootstrap"],
                                  bootstrap_features=self.params["bootstrap_features"],
                                  random_state=42)
        return model

    def update_params(self, trial: Trial):
        self.params["max_samples"] = trial.suggest_float("max_samples", 0.1, 1.0)
        self.params["max_features"] = trial.suggest_int("max_features", 1, len(self.X_train.columns))
        self.params["bootstrap"] = trial.suggest_categorical("bootstrap", [True, False])
        self.params["bootstrap_features"] = trial.suggest_categorical("bootstrap_features", [True, False])

    def run(self, trials=100, use_logger=False):
        super().run(trials, use_logger)
"""#### AdaBoost"""
class AdaBoostTuner(ClassificationTuner):
    def __init__(self, X_train, X_test, y_train, y_test, labeling_type,
                 weights={"accuracy": 1, "balanced accuracy": 1, "precision": 1,
                          "recall": 1, "f1 score": 1}):
        super().__init__("AdaBoost", X_train, X_test, y_train, y_test, labeling_type, weights)

    def build_model(self):
        model = AdaBoostClassifier(n_estimators=self.params["n_estimators"],
                                   learning_rate=self.params["learning_rate"],
                                   algorithm=self.params["algorithm"],
                                   random_state=42)
        return model

    def update_params(self, trial: Trial):
        self.params["n_estimators"] = trial.suggest_int("n_estimators", 10, 50)
        self.params["learning_rate"] = trial.suggest_float("learning_rate", 0, 2)
        self.params["algorithm"] = trial.suggest_categorical("algorithm", ["SAMME", "SAMME.R"])

    def run(self, trials=100, use_logger=False):
        super().run(trials, use_logger)
"""#### GradientBoosting"""
class GradientBosstingTuner(ClassificationTuner):
    def __init__(self, X_train, X_test, y_train, y_test, labeling_type,
                 weights={"accuracy": 1, "balanced accuracy": 1, "precision": 1,
                          "recall": 1, "f1 score": 1}):
        super().__init__("GradientBossting", X_train, X_test, y_train, y_test, labeling_type, weights)

    def build_model(self):
        model = GradientBoostingClassifier(
            learning_rate=self.params["learning_rate"],
            criterion=self.params["criterion"],
            min_samples_split=self.params["min_samples_split"],
            max_depth=self.params["max_depth"],
            random_state=42,
            max_features=self.params["max_features"],
        )
        return model

    def update_params(self, trial: Trial):
        self.params["learning_rate"] = trial.suggest_float("learning_rate", 0, 1)
        self.params["criterion"] = trial.suggest_categorical("criterion", ["friedman_mse", "mse"])
        self.params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 5)
        self.params["max_depth"] = trial.suggest_int("max_depth", 2, 20)
        self.params["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

    def run(self, trials=300, use_logger=False):
        super().run(trials, use_logger)

"""### regression"""
class RegressionTuner(SklearnTuner, ABC):
    def __init__(self, name, X_train, X_test, y_train, y_test, labeling_type, weights):
        super().__init__(name, X_train, X_test, y_train, y_test, labeling_type)
        self.weights = weights
        self.best_score = -float('inf')
        self.direction = "maximize"

    def show_best_results(self, path_to_save=None):
        scores = pd.DataFrame(columns=['model', 'mean absolute error',
                                       'mean squared error'])
        majority_eval = self.majority_eval
        majority_eval["model"] = "majority rule"
        scores = scores.append(majority_eval, ignore_index=True)

        best_results = self.get_best_results()
        scores = scores.append(best_results, ignore_index=True)

        predictions = self.best_model.predict(self.X_test)
        baseline_predictions = self._get_baseline_prediction()

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(1, 1, 1)

        sns.lineplot(x=list(range(len(self.y_test))), y=predictions, legend='full', label="predictions", ax=ax)
        sns.lineplot(x=list(range(len(self.y_test))), y=baseline_predictions, legend='full', label="baseline", ax=ax)
        sns.lineplot(x=list(range(len(self.y_test))), y=self.y_test, legend='full', label="true value", ax=ax)

        if path_to_save is not None:
            plt.close(fig)
            os.mkdir(f"{path_to_save}/{self.name}")
            fig.savefig(f"{path_to_save}/{self.name}/results.png")
            with open(f'{path_to_save}/{self.name}/pickled_scores.df', 'wb') as res_file:
                pickle.dump(scores, res_file)
            scores.to_html(f'{path_to_save}/{self.name}/scores.html')
            with open(f'{path_to_save}/{self.name}/model.md', 'wb') as model_file:
                pickle.dump(self.best_model, model_file)
            return

        return scores

    def _get_baseline_prediction(self):
        mode_label = self.y_train.mean()
        baseline_prediction = [mode_label for _ in range(len(self.y_test))]
        return baseline_prediction

    def _get_prediction_score(self, y_pred, y_true):
        return dict(reg_scores(np.array(y_true), np.array(y_pred)))

    @abstractmethod
    def build_model(self):
        pass
"""#### Ridge"""
class RidgeTuner(RegressionTuner):
    def __init__(self, X_train, X_test, y_train, y_test, labeling_type,
                 weights={"accuracy": 1, "balanced accuracy": 1, "precision": 1,
                          "recall": 1, "f1 score": 1}):
        super().__init__("ridge", X_train, X_test, y_train, y_test, labeling_type, weights)

    def build_model(self):
        model = Ridge(
            alpha=self.params["alpha"],
            fit_intercept=self.params["fit_intercept"],
            normalize=self.params["normalize"],
            max_iter=25000,
            solver=self.params["solver"],
            random_state=42)
        return model

    def update_params(self, trial: Trial):
        self.params["alpha"] = trial.suggest_float("alpha", 1, 50)
        self.params["fit_intercept"] = trial.suggest_categorical("fit_intercept", [True, False])
        self.params["normalize"] = trial.suggest_categorical("normalize", [True, False])
        self.params["tol"] = trial.suggest_float("tol", 1e-5, 1e-1)
        self.params["solver"] = trial.suggest_categorical("solver",
                                                          ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"])

    def run(self, trials=500, use_logger=False):
        super().run(trials, use_logger)
"""#### Lasso"""
class LassoTuner(RegressionTuner):
    def __init__(self, X_train, X_test, y_train, y_test, labeling_type,
                 weights={"accuracy": 1, "balanced accuracy": 1, "precision": 1,
                          "recall": 1, "f1 score": 1}):
        super().__init__("lasso", X_train, X_test, y_train, y_test, labeling_type, weights)

    def build_model(self):
        model = Lasso(alpha=self.params["alpha"],
                      fit_intercept=self.params["fit_intercept"],
                      normalize=self.params["normalize"],
                      max_iter=-1,
                      tol=self.params["tol"],
                      positive=self.params["positive"],
                      random_state=42,
                      selection=self.params["selection"])
        return model

    def update_params(self, trial: Trial):
        self.params["alpha"] = trial.suggest_float("alpha", 0.00001, 1)
        self.params["fit_intercept"] = trial.suggest_categorical("fit_intercept", [True, False])
        self.params["normalize"] = trial.suggest_categorical("normalize", [True, False])
        self.params["tol"] = trial.suggest_float("tol", 1e-6, 1e-2)
        self.params["positive"] = trial.suggest_categorical("positive", [True, False])
        self.params["selection"] = trial.suggest_categorical("selection", ["cyclic", "random"])

    def run(self, trials=500, use_logger=False):
        super().run(trials, use_logger)
"""#### ElasticNet"""
class ElasticNetTuner(RegressionTuner):
    def __init__(self, X_train, X_test, y_train, y_test, labeling_type,
                 weights={"accuracy": 1, "balanced accuracy": 1, "precision": 1,
                          "recall": 1, "f1 score": 1}):
        super().__init__("ElasticNet", X_train, X_test, y_train, y_test, labeling_type, weights)

    def build_model(self):
        model = ElasticNet(alpha=self.params["alpha"],
                           l1_ratio=self.params["l1_ratio"],
                           fit_intercept=self.params["fit_intercept"],
                           normalize=self.params["normalize"],
                           max_iter=-1,
                           tol=self.params["tol"],
                           positive=self.params["positive"],
                           random_state=42,
                           selection=self.params["selection"])
        return model

    def update_params(self, trial: Trial):
        self.params["alpha"] = trial.suggest_float("alpha", 0.00001, 1)
        self.params["l1_ratio"] = trial.suggest_float("l1_ratio", 0, 0.1)
        self.params["fit_intercept"] = trial.suggest_categorical("fit_intercept", [True, False])
        self.params["normalize"] = trial.suggest_categorical("normalize", [True, False])
        self.params["tol"] = trial.suggest_float("tol", 1e-6, 1e-2)
        self.params["positive"] = trial.suggest_categorical("positive", [True, False])
        self.params["selection"] = trial.suggest_categorical("selection", ["cyclic", "random"])

    def run(self, trials=500, use_logger=False):
        super().run(trials, use_logger)
"""#### Lars"""
class LarsTuner(RegressionTuner):
    def __init__(self, X_train, X_test, y_train, y_test, labeling_type,
                 weights={"accuracy": 1, "balanced accuracy": 1, "precision": 1,
                          "recall": 1, "f1 score": 1}):
        super().__init__("Lars", X_train, X_test, y_train, y_test, labeling_type, weights)

    def build_model(self):
        model = Lars(fit_intercept=self.params["fit_intercept"],
                     normalize=self.params["normalize"],
                     precompute=self.params["precompute"],
                     n_nonzero_coefs=self.params["n_nonzero_coefs"],
                     eps=self.params["eps"],
                     jitter=self.params["jitter"],
                     random_state=42)
        return model

    def update_params(self, trial: Trial):
        self.params["fit_intercept"] = trial.suggest_categorical("fit_intercept", [True, False])
        self.params["normalize"] = trial.suggest_categorical("normalize", [True, False])
        self.params["precompute"] = trial.suggest_categorical("precompute", [True, False])
        self.params["n_nonzero_coefs"] = trial.suggest_int("n_nonzero_coefs", 1, 500)
        self.params["eps"] = trial.suggest_float("eps", 0, 1)
        self.params["jitter"] = trial.suggest_float("jitter", 0, 10)

    def run(self, trials=500, use_logger=False):
        super().run(trials, use_logger)
"""#### KNeighbors Regressor"""
class KNeighborsRegressorTuner(RegressionTuner):
    def __init__(self, X_train, X_test, y_train, y_test, labeling_type,
                 weights={"accuracy": 1, "balanced accuracy": 1, "precision": 1,
                          "recall": 1, "f1 score": 1}):
        super().__init__("knn_reg", X_train, X_test, y_train, y_test, labeling_type, weights)

    def build_model(self):
        model = KNeighborsRegressor(n_neighbors=self.params["n_neighbors"],
                                    weights=self.params["weights"],
                                    algorithm=self.params["algorithm"],
                                    p=self.params["p"],
                                    metric=self.params["metric"])
        return model

    def update_params(self, trial: Trial):
        super().update_params(trial)
        self.params["n_neighbors"] = trial.suggest_int("n_neighbors", 1, 40)
        self.params["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
        self.params["algorithm"] = trial.suggest_categorical("algorithm", ["ball_tree", "kd_tree", "brute"])
        self.params["p"] = trial.suggest_int("p", 1, 5)
        self.params["metric"] = trial.suggest_categorical("metric",
                                                          ["euclidean", "manhattan", "chebyshev", "minkowski"])

    def run(self, trials=300, use_logger=False):
        super().run(trials, use_logger)
"""#### SVR"""
class SVRTuner(RegressionTuner):
    def __init__(self, X_train, X_test, y_train, y_test, labeling_type,
                 weights={"accuracy": 1, "balanced accuracy": 1, "precision": 1,
                          "recall": 1, "f1 score": 1}):
        super().__init__("svr", X_train, X_test, y_train, y_test, labeling_type, weights)

    def build_model(self):
        model = make_pipeline(MinMaxScaler(),
                              SVR(kernel=self.params["kernel"],
                                  degree=self.params["degree"],
                                  gamma=self.params["gamma"],
                                  coef0=self.params["coef0"],
                                  C=self.params["C"],
                                  epsilon=self.params["epsilon"],
                                  max_iter=200000))
        return model

    def update_params(self, trial: Trial):
        self.params["kernel"] = trial.suggest_categorical("kernel", ['linear', 'poly', 'rbf', 'sigmoid'])
        self.params["degree"] = trial.suggest_int("degree", 2, 10)
        self.params["gamma"] = trial.suggest_float("gamma", 0, 1)
        self.params["coef0"] = trial.suggest_float("coef0", 0, 1)
        self.params["C"] = trial.suggest_float("C", 0, 1)
        self.params["epsilon"] = trial.suggest_float("epsilon", 0.01, 0.1)

    def run(self, trials=500, use_logger=False):
        super().run(trials, use_logger)
"""#### Random Forest Regression"""
class RandomForestRegressorTuner(RegressionTuner):
    def __init__(self, X_train, X_test, y_train, y_test, labeling_type,
                 weights={"accuracy": 1, "balanced accuracy": 1, "precision": 1,
                          "recall": 1, "f1 score": 1}):
        super().__init__("random_forest_reg", X_train, X_test, y_train, y_test, labeling_type, weights)

    def build_model(self):
        model = RandomForestRegressor(
            n_estimators=150,
            criterion=self.params["criterion"],
            max_depth=self.params["max_depth"],
            min_samples_split=self.params["min_samples_split"],
            max_features=self.params["max_features"],
            bootstrap=self.params["bootstrap"],
            random_state=42,
            ccp_alpha=self.params["ccp_alpha"]
        )
        return model

    def update_params(self, trial: Trial):
        self.params["criterion"] = trial.suggest_categorical("criterion", ["mse", "mae"])
        self.params["max_depth"] = trial.suggest_int("max_depth", 2, 50)
        self.params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 5)
        self.params["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        self.params["bootstrap"] = trial.suggest_categorical("bootstrap", [True, False])
        self.params["ccp_alpha"] = trial.suggest_float("ccp_alpha", 0, 0.1)

    def run(self, trials=500, use_logger=False):
        super().run(trials, use_logger)
"""#### Bagging Regeressor"""
class BaggingRegressorTuner(RegressionTuner):
    def __init__(self, X_train, X_test, y_train, y_test, labeling_type,
                 weights={"accuracy": 1, "balanced accuracy": 1, "precision": 1,
                          "recall": 1, "f1 score": 1}):
        super().__init__("bagging_reg", X_train, X_test, y_train, y_test, labeling_type, weights)

    def build_model(self):
        model = BaggingRegressor(
            n_estimators=self.params["n_estimators"],
            max_samples=self.params["max_samples"],
            max_features=self.params["max_features"],
            bootstrap=self.params["bootstrap"],
            bootstrap_features=self.params["bootstrap_features"],
            random_state=42)
        return model

    def update_params(self, trial: Trial):
        self.params["n_estimators"] = trial.suggest_int("n_estimators", 50, 150)
        self.params["max_samples"] = trial.suggest_float("max_samples", 0.1, 1)
        self.params["max_features"] = trial.suggest_int("max_features", 1, len(self.X_train.columns))
        self.params["bootstrap"] = trial.suggest_categorical("bootstrap", [True, False])
        self.params["bootstrap_features"] = trial.suggest_categorical("bootstrap_features", [True, False])

    def run(self, trials=500, use_logger=False):
        super().run(trials, use_logger)
"""#### AdaBoost Regressor"""
class AdaBoostRegressionTuner(RegressionTuner):
    def __init__(self, X_train, X_test, y_train, y_test, labeling_type,
                 weights={"accuracy": 1, "balanced accuracy": 1, "precision": 1,
                          "recall": 1, "f1 score": 1}):
        super().__init__("AdaBoost_reg", X_train, X_test, y_train, y_test, labeling_type, weights)

    def build_model(self):
        model = AdaBoostRegressor(n_estimators=self.params["n_estimators"],
                                  learning_rate=self.params["learning_rate"],
                                  loss=self.params["loss"],
                                  random_state=42)
        return model

    def update_params(self, trial: Trial):
        self.params["n_estimators"] = trial.suggest_int("n_estimators", 10, 100)
        self.params["learning_rate"] = trial.suggest_float("learning_rate", 0, 2)
        self.params["loss"] = trial.suggest_categorical("loss", ["linear", "square", "exponential"])

    def run(self, trials=500, use_logger=False):
        super().run(trials, use_logger)

"""## evaluation"""
def get_best_models(tuners, X_train, X_test, y_train, y_test,
                    labeling_type, weights, path_to_save, verbose=True, trials=None):
    optuna.logging.set_verbosity(optuna.logging.FATAL)
    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)

    untrained_best_models = {}
    if labeling_type == "categorical" or labeling_type == "positive":
        task = "classification"
        majority = y_train[labeling_type].mode()[0]
        majority_predictions = [majority for _ in range(len(y_test[labeling_type]))]
        majority_eval = dict(clf_scores(np.array(y_test[labeling_type]), np.array(majority_predictions)))
    else:
        task = "regression"
        majority = y_train[labeling_type].mean()
        majority_predictions = [majority for _ in range(len(y_test[labeling_type]))]
        majority_eval = dict(reg_scores(np.array(y_test[labeling_type]), np.array(majority_predictions)))

    for tuner in tuners:
        tuner = tuner(X_train, X_test, y_train, y_test, labeling_type, weights)
        print(f"~~~~~~~~~~~~~~~~~~~~{tuner.name}~~~~~~~~~~~~~~~~~~~~")
        if trials is not None:
            tuner.run(trials, use_logger=False)
        else:
            tuner.run(use_logger=False)

        if tuner.best_model is not None:
            untrained_best_models[tuner.name] = tuner.build_model()
            tuner.show_best_results(path_to_save)

    print(f"~~~~~~~~~~~~~~~~~~~~voting~~~~~~~~~~~~~~~~~~~~")
    estimators = list(untrained_best_models.items())
    if task == "classification":
        voting_model = VotingClassifier(estimators, voting='hard', weights=None,
                                        n_jobs=-1, flatten_transform=True,
                                        verbose=False)
        voting_model.fit(X_train, y_train[labeling_type])
        voting_preds = voting_model.predict(X_test)
        voting_eval = dict(clf_scores(np.array(y_test[labeling_type]), np.array(voting_preds)))
        measurements = ["model", "accuracy", "balanced accuracy", "precision", "recall", "f1 score"]
    elif task == "regression":
        voting_model = VotingRegressor(estimators, weights=None, n_jobs=-1, verbose=False)
        voting_model.fit(X_train, y_train[labeling_type])
        voting_preds = voting_model.predict(X_test)
        voting_eval = dict(reg_scores(np.array(y_test[labeling_type]), np.array(voting_preds)))
        measurements = ["model", "mean absolute error", "mean squared error"]

    voting_score = 0
    for param in voting_eval:
        voting_score += weights[param] * (majority_eval[param] - voting_eval[param])
    print(f"best value (mujority-model-dif_sum): {voting_score}")

    os.mkdir(f"{path_to_save}/voting")
    majority_eval["model"] = "majority rule"
    voting_eval["model"] = "voting ensamble"
    voting_scores = pd.DataFrame(columns=measurements)
    voting_scores = voting_scores.append(majority_eval, ignore_index=True)
    voting_scores = voting_scores.append(voting_eval, ignore_index=True)

    if task == "regression":
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(1, 1, 1)
        sns.lineplot(x=list(range(len(y_test[labeling_type]))), y=voting_preds, legend='full', label="predictions",
                     ax=ax)
        sns.lineplot(x=list(range(len(y_test[labeling_type]))), y=majority_predictions, legend='full', label="baseline",
                     ax=ax)
        sns.lineplot(x=list(range(len(y_test[labeling_type]))), y=y_test[labeling_type], legend='full',
                     label="true value", ax=ax)
        plt.close(fig)
        fig.savefig(f"{path_to_save}/voting/results.png")

    with open(f'{path_to_save}/voting/pickled_scores.df', 'wb') as res_file:
        pickle.dump(voting_scores, res_file)
    voting_scores.to_html(f'{path_to_save}/voting/scores.html')
    with open(f'{path_to_save}/voting/model.md', 'wb') as model_file:
        pickle.dump(voting_model, model_file)

clf_tuners = [BaggingTuner,
              AdaBoostTuner,
              LinearSVMTuner,
              KNeighborsTuner,
              DecisionTreeTuner,
              RandomForestTuner,
              GradientBosstingTuner,
              LogisticRegressionTuner]
clf_weights = {"accuracy": 1, "balanced accuracy": 0.5,
               "precision": 0, "recall": 0, "f1 score": 0}

reg_tuners = [SVRTuner,
              LarsTuner,
              RidgeTuner,
              LassoTuner,
              ElasticNetTuner,
              BaggingRegressorTuner,
              AdaBoostRegressionTuner,
              KNeighborsRegressorTuner,
              RandomForestRegressorTuner]
reg_weights = {"mean absolute error": 1, "mean squared error": 1}

durations = [1, 2, 5, 10, 15]
channels = {"Mouse": "Mouse",
            "Keyboard": "Keyboard",
            "Both": ["Mouse", "Keyboard"]}
labels = ["categorical", "positive", "valance", "arousal", "dominance"]
testees = ["yuval", "ron", "liraz", "niv", "shira-no_images",
           "shiran", "tal", "amit_dabash-no_images", "shoham"]
path_to_save = f"/content/drive/MyDrive/capi/results"

def tune_all_models(trials=None):
    for testee in testees:
        tune_testee_models(testee, trials)


def tune_testee_models(name, trials=None):
    print(f"**********tuning {name}'s data**********")
    if not os.path.isdir(f"{path_to_save}/{name}"):
        os.mkdir(f"{path_to_save}/{name}")
    for duration in durations:
        tune_testee_duration_models(name, duration, trials)


def tune_testee_duration_models(name, duration, trials=None):
    print(f"------{name}__{duration}------")
    if not os.path.isdir(f"{path_to_save}/{name}"):
        os.mkdir(f"{path_to_save}/{name}")
    if not os.path.isdir(f"{path_to_save}/{name}/{duration}"):
        os.mkdir(f"{path_to_save}/{name}/{duration}")
    for channel_name in channels:
        tune_testee_channel_models(name, duration, channel_name, trials)


def tune_testee_channel_models(name, duration, channel_name, trials=None):
    print(f"------{name}__{duration}__{channel_name}------")
    if not os.path.isdir(f"{path_to_save}/{name}"):
        os.mkdir(f"{path_to_save}/{name}")
    if not os.path.isdir(f"{path_to_save}/{name}/{duration}"):
        os.mkdir(f"{path_to_save}/{name}/{duration}")
    if not os.path.isdir(f"{path_to_save}/{name}/{duration}/{channel_name}"):
        os.mkdir(f"{path_to_save}/{name}/{duration}/{channel_name}")
    print(f"reading {name}'s {channel_name} data")
    X, X_train, X_test, y, y_train, y_test = load_data(f"/content/drive/MyDrive/capi/data/public/{name}.db",
                                                       channels[channel_name], duration)
    print("finished reading the data")
    for labeling_type in labels:
        tune_testee_label_models(name, duration, channel_name, labeling_type, (X, X_train, X_test, y, y_train, y_test),
                                 trials)


def tune_testee_label_models(name, duration, channel_name, labeling_type, data=None, trials=None):
    print(f"------{name}__{duration}__{channel_name}__{labeling_type}------")
    if not os.path.isdir(f"{path_to_save}/{name}"):
        os.mkdir(f"{path_to_save}/{name}")
    if not os.path.isdir(f"{path_to_save}/{name}/{duration}"):
        os.mkdir(f"{path_to_save}/{name}/{duration}")
    if not os.path.isdir(f"{path_to_save}/{name}/{duration}/{channel_name}"):
        os.mkdir(f"{path_to_save}/{name}/{duration}/{channel_name}")
    if os.path.isdir(f"{path_to_save}/{name}/{duration}/{channel_name}/{labeling_type}"):
        print(f"!!!!! been there done that !!!!! \n  Error in {name} {duration} {channel_name} {labeling_type}\n\n")
        return
    os.mkdir(f"{path_to_save}/{name}/{duration}/{channel_name}/{labeling_type}")

    if labeling_type == "categorical" or labeling_type == "positive":
        weights = clf_weights
        tuners = clf_tuners
    else:
        weights = reg_weights
        tuners = reg_tuners

    if data is not None:
        X, X_train, X_test, y, y_train, y_test = data
    else:
        print(f"reading {name}'s {channel_name} data")
        X, X_train, X_test, y, y_train, y_test = load_data(f"/content/drive/MyDrive/capi/data/public/{name}.db",
                                                           channels[channel_name], duration)
        print("finished reading the data")

    get_best_models(tuners, X_train, X_test, y_train, y_test, labeling_type, weights,
                    path_to_save=f"{path_to_save}/{name}/{duration}/{channel_name}/{labeling_type}",
                    verbose=False, trials=trials)
    print("\n\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", choices=testees)
    parser.add_argument("-d", "--duration", type=int, choices=durations)
    parser.add_argument("-c", "--channel", choices=list(channels.keys()))
    parser.add_argument("-l", "--labeling_type", choices=labels)
    parser.add_argument("-t", "--trials", type=int)
    args = parser.parse_args()

    if args.name is None:
        tune_all_models(args.trials)
    elif args.duration is None:
        tune_testee_models(args.name, args.trials)
    elif args.channel is None:
        tune_testee_duration_models(args.name, args.duration, args.trials)
    elif args.labeling_type is None:
        tune_testee_channel_models(args.name, args.duration, channels[args.channel], args.trials)
    else:
        tune_testee_label_models(args.name, args.duration, args.channel, args.labeling_type, trials=args.trials)

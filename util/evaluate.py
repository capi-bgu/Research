from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Any, Union, Dict, NewType, List, Callable

import numpy as np
import sklearn.base
import wandb
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, \
    recall_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.python.keras.callbacks import ModelCheckpoint
from wandb.keras import WandbCallback

NdArrayLike = NewType("NdArray", Any)
CapiModel = NewType("CapiModel", Union[keras.Model, sklearn.base.BaseEstimator, Any])
Scorer = NewType("Scorer", Callable[[NdArrayLike, NdArrayLike], List[Tuple[str, Any]]])


def clf_scores(true_values, pred_probs):
    pred_values = np.argmax(pred_probs, axis=1)

    acc = accuracy_score(true_values, pred_values)
    bacc = balanced_accuracy_score(true_values, pred_values)
    precision = precision_score(true_values, pred_values, average='micro')
    recall = recall_score(true_values, pred_values, average='micro')
    f1 = f1_score(true_values, pred_values, average='micro')

    return [
        ("accuracy", acc),
        ("balanced accuracy", bacc),
        ("precision", precision),
        ("recall", recall),
        ("f1 score", f1)
    ]


def reg_scores(true_values, pred_values):
    mae = mean_absolute_error(true_values, pred_values)
    mse = mean_squared_error(true_values, pred_values)
    mape = mean_absolute_percentage_error(true_values, pred_values)

    return [
        ("mean absolute error", mae),
        ("mean squared error", mse),
        ("mean absolute percentage error", mape)
    ]


class Task(Enum):
    CLF = 1
    REG = 2


class Evaluator(ABC):

    def __init__(self, project: str, name: str, task: Task = Task.CLF,
                 use_logger=False, use_ensemble=True, model_path: str = None, scorer: Scorer = None):
        super().__init__()
        self.project = project
        self.name = name
        self.task = task
        self.scorer = clf_scores
        self.use_logger = use_logger
        self.use_ensemble = use_ensemble
        self.test_predictions = []
        self.fold = 0

        if not model_path:
            model_path = f"./{self.project}/{self.name}"

        self.callbacks = [ModelCheckpoint(path=model_path, save_best_only=True)]

        if use_logger:
            self.callbacks += [WandbCallback(save_model=False)]

        if self.task == Task.REG:
            self.scorer = reg_scores

        if scorer:
            self.scorer = scorer

    def disable_logger(self):
        self.use_logger = False

    def enable_logger(self):
        self.use_logger = True

    @abstractmethod
    def train(self, train_data: Tuple[NdArrayLike, NdArrayLike],
              validation_data: Tuple[NdArrayLike, NdArrayLike],
              callbacks: List[keras.callbacks.Callback]) -> CapiModel:
        """
        This method is meant to create, and train a model using the training and validation data provided and
        the model configuration defined in `model_config`.

        The method also accepts a list of keras callbacks, if the model being trained is a tensorflow model,
        these callbacks provide some useful defaults for wandb and model checkpointing.
        """
        pass

    @abstractmethod
    def predict(self, model: CapiModel, data: NdArrayLike) -> NdArrayLike:
        """
        This method accepts a model created by the `train` method in addition to a data set
        and runs the model on the given data set, returning the predictions.
        """
        pass

    @abstractmethod
    def model_config(self) -> Union[None, Dict[str, Any]]:
        """
        Defines the model configuration and hyper-parameters, if logging is enabled this method is used
        for the model configuration displayed in the log.
        """
        pass

    def predict_ensemble(self) -> NdArrayLike:
        """
        By default we use a mean classifier/regression as the ensemble method.
        This method can be overridden in order to use a more complex ensemble method.

        to calculate the ensemble predictions you can use `self.test_predictions` an array containing,
        the predictions of all the models trained in all the folds.
        The method must return an NdarrayLike object that has the same dimensions as a single prediction
        of the model.
        """
        stacked_predictions = np.stack(self.test_predictions)
        return np.mean(stacked_predictions, axis=0)

    def _init_logger(self, ensemble=False):
        """
        If logging is enabled this method initializes a new the logging for a new fold.
        This method is called once at the beginning of every fold.
        If using the ensemble model this method will be called once again before the prediction.
        """
        if not self.use_logger:
            return

        run_name = f"{self.name}-fold-{self.fold}"
        if ensemble:
            run_name = self.name + "-ensemble"

        wandb.init(
            project=self.project,
            group=f"{self.name}-eval",
            name=run_name,
            config=self.model_config()
        )

    def log_validation(self, model: CapiModel, x_val: NdArrayLike, y_val: NdArrayLike,
                       label_map: Dict[int, str] = None) -> NdArrayLike:
        """
        If logging is enabled this method will log the validation metrics.
        This method is called once at the end of each fold, after calculating the validation metrics.
        This method also returns the validation predictions generated by the model.
        """
        val_pred = self.predict(model, x_val)
        val_scores = self.scorer(val_pred, y_val)
        print("----------------------------------------")
        print(f"validation results for fold number {self.fold}:")
        for metric, score in val_scores:
            print(f"  {metric}: {score}")
        print()

        if not self.use_logger:
            wandb.log({"validation " + metric: score for (metric, score) in val_scores}, commit=False)
            if self.task == Task.CLF:
                val_class_pred = np.argmax(val_pred, axis=1)
                wandb.log({"validation confusion matrix": wandb.plot.confusion_matrix(val_class_pred,
                                                                                      y_val,
                                                                                      list(label_map.values()))},
                          commit=False)

        return val_pred

    def log_testing(self, model: CapiModel, testing_data: NdArrayLike, testing_labels: NdArrayLike,
                    label_map: Dict[int, str] = None, ensemble=False) -> NdArrayLike:
        """
        If logging is enabled this method will log the testing metrics.
        This method is called once at the end of each fold, after calculating the validation metrics.
        This method also returns the test predictions generated by the model.
        """
        test_pred = self.predict(model, testing_data)
        test_scores = self.scorer(testing_labels, test_pred)
        if not ensemble:
            print(f"test results for fold number {self.fold}:")

        for metric, score in test_scores:
            print(f"  {metric}: {score}")

        if not ensemble:
            print("----------------------------------------")

        if not self.use_logger:
            wandb.log({"test " + metric: score for (metric, score) in test_scores}, commit=False)
            if self.task == Task.CLF:
                test_class_pred = np.argmax(test_pred, axis=1)
                wandb.log({"testing confusion matrix": wandb.plot.confusion_matrix(test_class_pred,
                                                                                   testing_labels,
                                                                                   list(label_map.values()))},
                          commit=False)

        return test_pred

    def log_examples(self, model: CapiModel, test_data: NdArrayLike, test_labels: NdArrayLike, test_pred: NdArrayLike):
        """
        This method is empty by default, it is called at the end of each fold with the model trained in said fold,
        and the testing data, labels, and predictions. Using this information the implementer is expected to somehow
        log example results from the model that will help understand what the model has learned.

        For example we can log a correct and incorrect classification to see when the model fails and when it succeeds,
        or if we are using a deep learning model we could try and reflect the gradients of some predictions on the input
        images and see what the models considers important in the image.
        """
        pass

    def _end_fold(self, test_pred: NdArrayLike):
        if self.use_logger:
            wandb.log({}, commit=True)
            wandb.finish()

        self.test_predictions.append(test_pred)
        self.fold += 1

    def evaluate(self, training: Tuple[NdArrayLike, NdArrayLike],
                 testing: Tuple[NdArrayLike, NdArrayLike],
                 label_map: Dict[int, str] = None, splits=5):
        """
        Evaluates the model given training and testing data.
        This method also accepts `label_map`. If you are evaluating a classification model,
        `label_map` should be a dictionary mapping the target values to their human readable labels.
        The label map will be used by the logger if enabled.
        This method also accepts `splits` - the number of folds to evaluate the model with.
        """

        print("----------------------------------------")
        print(f"evaluating {self.name} model")
        print("----------------------------------------")

        training_data, training_labels = training
        testing_data, testing_labels = testing
        kf = KFold(n_splits=splits)

        for train_index, val_index in kf.split(training_data):
            self._init_logger()
            x_train = training_data[train_index]
            y_train = training_labels[train_index]

            x_val = training_data[val_index]
            y_val = training_labels[val_index]

            model = self.train((x_train, y_train), (x_val, y_val), self.callbacks)

            self.log_validation(model, x_val, y_val, label_map)
            test_pred = self.log_testing(model, testing_data, testing_labels, label_map)
            self.log_examples(model, testing_data, testing_labels, test_pred)
            self._end_fold(test_pred)

        if self.use_ensemble:
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            print(f"mean classifier test results:")
            self._init_logger()
            ensemble_pred = self.predict_ensemble()
            self.log_testing(ensemble_pred, testing_labels, label_map)
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

            self._end_fold(ensemble_pred)

        self.test_predictions = []
        self.fold = 0

from abc import ABC, abstractmethod
from typing import Any, Dict, List, NewType

import optuna
import wandb
from optuna import Trial
from optuna.integration import TFKerasPruningCallback
from optuna.structs import TrialState
from tensorflow.keras.callbacks import Callback
from wandb.keras import WandbCallback

Params = NewType("Params", Dict[str, Any])


class Tuner(ABC):

    def __init__(self, project: str, name: str, monitor: str, default_params: Params):
        super().__init__()
        self.project = project
        self.name = name
        self.monitor = monitor
        self.params = default_params

    @abstractmethod
    def objective(self, callbacks: List[Callback] = ()) -> float:
        """
        This is the tuners objective function, it should create and train a model using the parameters
        defined in the classes initializer.
        The parameters can be accessed using `self.params`.
        This method returns a float value representing the value of the trained model to the optimizer.
        For example it could return the best accuracy, of the model.
        Note: in the `run` function you most specify a direction that fits with the value returned here.
        If you use accuracy as the value then the direction should be "maximize", while if you are using something like
        mse the direction should be "minimize"
        """
        pass

    @abstractmethod
    def update_params(self, trial: Trial):
        """
        This method is called right before the `objective` method is called, here the optimizer updates
        the hyper-parameters the `objective` method will use to create the model.

        This method accepts a `trial` variable which allows us to specify the bounds of the optimization.
        For example if you want to optimize an integer value, you can use
        `trial.suggest_int("some_name", lower_bound, upper_bound)`.
        You can learn more about how to use trial to pick other kinds of values here :
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
        """
        pass

    def run(self, trials: int, direction="minimize", use_logger=True) -> Params:
        """
        Runs the tuner with `trials` number of trials, and the given direction.
        If `use_logger` is enabled the results of each of the trials will be logged to wandb.
        """

        def objective_wrapper(trial: Trial):
            self.update_params(trial)
            callbacks: List = [TFKerasPruningCallback(trial, self.monitor)]
            if use_logger:
                wandb.init(
                    project=self.project,
                    group=f"{self.name}-sweep",
                    name=f"{self.name}-sweep-{trial.number}",
                    config=self.params
                )
                callbacks += [WandbCallback(save_model=False)]

            target_value = self.objective(callbacks)

            if use_logger:
                wandb.finish()

            return target_value

        study = optuna.create_study(direction=direction, pruner=optuna.pruners.HyperbandPruner())
        study.optimize(objective_wrapper, n_trials=trials)

        self.params.update(study.best_params)
        pruned_trials = len(study.get_trials(deepcopy=False, states=(TrialState.PRUNED,)))

        print("hyperparameter optimization results:")
        print(f"\tnumber of trials: {trials}")
        print(f"\tnumber of pruned trials: {pruned_trials}")
        print(f"\tbest value ({self.monitor}): {study.best_value}")
        print(f"\toptimized parameters:")
        for key, value in study.best_params.items():
            print(f"\t\t{key}: {value}")
        print(f"\tfinal parameters:")
        for key, value in self.params.items():
            print(f"\t\t{key}: {value}")

        return self.params

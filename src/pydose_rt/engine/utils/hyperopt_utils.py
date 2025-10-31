import os
import pickle
import hyperopt
import numpy as np
import pandas as pd
import abc
from abc import abstractmethod
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from termcolor import colored
import time
import random
import engine.utils.path_utils as path_utils
from engine.config import config
import scipy.stats as stat


class IncrementalObjectiveOptimization(abc.ABC):
    def __init__(self, hyperopt_project_path, is_print=True):
        self.hyperopt_project_path = hyperopt_project_path
        self.fill_inf_nan = 10_000_000_000
        self.is_print = is_print

    def compute_ranks(self, X, method="average"):
        if method not in ["average", "min", "max", "dense", "ordinal"]:
            raise ValueError(
                'Method must be one of "average", "min", "max", '
                '"dense" and "ordinal".'
            )

        n = X.shape[0]
        R = np.zeros(X.shape)
        for i in range(n):
            r = stat.rankdata(X[i, :], method=method)
            R[i, :] = r

        return R

    def update_trials_losses(self, evaluations: list):
        with open(self.hyperopt_project_path, "rb") as f:
            trials = pickle.load(f)

        if self.is_print:
            print()
            print("Before updating losses")
            for i, trial in enumerate(trials.trials):
                print(f"Trial {i} loss: {trial['result']['loss']}")

        d = {}
        for trial in trials:
            if trial["result"]["reason"] == "success":  # we ignore failed trials
                tid = trial["tid"]
                d[tid] = []
                for evaluation in evaluations:
                    row = trial["result"][f"scores_{evaluation}"]
                    row = pd.DataFrame([row])
                    row = self.update_metric_higher_is_better(row, evaluation)
                    d[tid].extend(row.values.tolist()[0])

        df_score = pd.DataFrame.from_dict(d, orient="index").T
        df_score = df_score.fillna(self.fill_inf_nan)

        df_data = df_score.iloc[:, :]
        X = df_data.values

        tids = list(df_score.columns.values)
        ranks = self.compute_ranks(X)
        ranks_mean = ranks.mean(axis=0)

        newloss_dict = dict(zip(tids, ranks_mean))

        # update the loss according to mean of ranks and given tid (id)
        for trial in trials:
            if trial["result"]["reason"] != "success":
                trial["result"]["loss"] = np.inf
            else:
                tid = trial["tid"]
                trial["result"]["loss"] = -newloss_dict[
                    tid
                ]  # the loss is lower is better thus we need to change the sign here

        with open(self.hyperopt_project_path, "wb") as f:
            pickle.dump(trials, f)

        if self.is_print:
            print()
            print("After updating losses")
            for i, trial in enumerate(trials.trials):
                print(f"Trial {i} loss: {trial['result']['loss']}")

    def update_row_metric(self, metric_series, is_higher_is_better):
        # Apply the checks element-wise
        is_nan_or_inf = metric_series.apply(lambda x: np.isnan(x) or np.isinf(x))

        # Replace NaN or Inf with the fill value based on is_higher_is_better
        if is_higher_is_better:
            return np.where(is_nan_or_inf, -self.fill_inf_nan, metric_series)
        else:
            return np.where(is_nan_or_inf, self.fill_inf_nan, -metric_series)

    def get_row_fmin_evaluation(self, loss, evaluation):
        d = {}
        fmin = get_best_set_params(self.hyperopt_project_path)
        row = fmin[f"scores_{evaluation}"]
        row = pd.DataFrame([row])
        row = self.update_metric_higher_is_better(row, evaluation)
        d[f"metric"] = row.columns.tolist()
        d[f"{loss}"] = row.values.tolist()[0]
        df_score = pd.DataFrame.from_dict(d, orient="index").T
        return df_score

    def get_value_from_metric(self, metric, dictionary):
        # Check for exact match
        if metric in dictionary:
            return dictionary[metric]

        # Check for substring match
        for key, value in dictionary.items():
            if key in metric:
                return value

        # Raise ValueError if no match found
        raise ValueError("No matching key found in the dictionary.")

    @abstractmethod
    def update_metric_higher_is_better(self, row, evaluation):
        raise NotImplementedError


class IncrementalObjectiveOptimizationMLMethod(IncrementalObjectiveOptimization):
    def update_metric_higher_is_better(self, row, evaluation):
        for metric in row.columns:
            if evaluation == "statistics":
                if config.DICT_MAPPING_METRICS[metric] == "lower":
                    row[metric] = self.update_row_metric(row[metric], False)
                else:
                    row[metric] = self.update_row_metric(row[metric], True)
            else:
                if "mae" in metric or "mse" in metric:
                    row[metric] = self.update_row_metric(row[metric], False)
                else:
                    row[metric] = self.update_row_metric(row[metric], True)
        return row


class IncrementalObjectiveOptimizationGenerativeModel(IncrementalObjectiveOptimization):
    def update_metric_higher_is_better(self, row, evaluation):
        row = row.replace([np.inf], self.fill_inf_nan)
        row = row.replace([np.nan], self.fill_inf_nan)

        for metric in row.columns:
            if evaluation == "statistics":
                if config.DICT_MAPPING_METRICS[metric] == "lower":
                    row[metric] = self.update_row_metric(row[metric], False)
                else:
                    row[metric] = self.update_row_metric(row[metric], True)
            elif evaluation == "ml":  # lower is better
                row[metric] = self.update_row_metric(row[metric], False)
            elif (
                evaluation == "ml_augment"
            ):  # higher is better except for regression task (mae + mse)
                if "mae" in metric or "mse" in metric:
                    row[metric] = self.update_row_metric(row[metric], False)
                else:
                    row[metric] = self.update_row_metric(row[metric], True)
            elif evaluation == "dp":
                higher_or_lower = self.get_value_from_metric(
                    metric, config.DICT_MAPPING_METRICS
                )
                if higher_or_lower == "lower":
                    row[metric] = self.update_row_metric(row[metric], False)
                else:
                    row[metric] = self.update_row_metric(row[metric], True)
        return row


class StandardObjectiveOptimization(abc.ABC):
    def __init__(self, hyperopt_project_path, agg="mean", is_print=True):
        self.hyperopt_project_path = hyperopt_project_path
        self.fill_inf_nan = 10_000_000_000
        self.agg = agg
        self.is_print = is_print

    def compute_objective_function(self, X):
        if self.agg == "mean":
            return np.mean(X, axis=0)
        elif self.agg == "median":
            return np.median(X, axis=0)

    def update_trials_losses(self, evaluations: list):
        with open(self.hyperopt_project_path, "rb") as f:
            trials = pickle.load(f)

        if self.is_print:
            print()
            print("Before updating losses")
            for i, trial in enumerate(trials.trials):
                print(f"Trial {i} loss: {trial['result']['loss']}")

        d = {}
        for trial in trials:
            if trial["result"]["reason"] == "success":  # we ignore failed trials
                tid = trial["tid"]
                d[tid] = []
                for evaluation in evaluations:
                    row = trial["result"][f"scores_{evaluation}"]
                    row = pd.DataFrame([row])
                    row = self.update_metric_higher_is_better(row, evaluation)
                    d[tid].extend(row.values.tolist()[0])

        df_score = pd.DataFrame.from_dict(d, orient="index").T
        df_score = df_score.fillna(self.fill_inf_nan)

        df_data = df_score.iloc[:, :]
        X = df_data.values

        tids = list(df_score.columns.values)
        ranks_mean = self.compute_objective_function(X)

        newloss_dict = dict(zip(tids, ranks_mean))

        # update the loss according to mean of ranks and given tid (id)
        for trial in trials:
            if trial["result"]["reason"] != "success":
                trial["result"]["loss"] = np.inf
            else:
                tid = trial["tid"]
                trial["result"]["loss"] = -newloss_dict[
                    tid
                ]  # the loss is lower is better thus we need to change the sign here

        with open(self.hyperopt_project_path, "wb") as f:
            pickle.dump(trials, f)

        if self.is_print:
            print()
            print("After updating losses")
            for i, trial in enumerate(trials.trials):
                print(f"Trial {i} loss: {trial['result']['loss']}")

    def update_row_metric(self, metric_series, is_higher_is_better):
        # Apply the checks element-wise
        is_nan_or_inf = metric_series.apply(lambda x: np.isnan(x) or np.isinf(x))

        # Replace NaN or Inf with the fill value based on is_higher_is_better
        if is_higher_is_better:
            return np.where(is_nan_or_inf, -self.fill_inf_nan, metric_series)
        else:
            return np.where(is_nan_or_inf, self.fill_inf_nan, -metric_series)

    def get_row_fmin_evaluation(self, loss, evaluation):
        d = {}
        fmin = get_best_set_params(self.hyperopt_project_path)
        row = fmin[f"scores_{evaluation}"]
        row = pd.DataFrame([row])
        row = self.update_metric_higher_is_better(row, evaluation)
        d[f"metric"] = row.columns.tolist()
        d[f"{loss}"] = row.values.tolist()[0]
        df_score = pd.DataFrame.from_dict(d, orient="index").T
        return df_score

    def get_value_from_metric(self, metric, dictionary):
        # Check for exact match
        if metric in dictionary:
            return dictionary[metric]

        # Check for substring match
        for key, value in dictionary.items():
            if key in metric:
                return value

        # Raise ValueError if no match found
        raise ValueError("No matching key found in the dictionary.")

    @abstractmethod
    def update_metric_higher_is_better(self, row, evaluation):
        raise NotImplementedError


class StandardObjectiveOptimizationMLMethod(StandardObjectiveOptimization):
    def update_metric_higher_is_better(self, row, evaluation):
        for metric in row.columns:
            if evaluation == "statistics":
                if config.DICT_MAPPING_METRICS[metric] == "lower":
                    row[metric] = self.update_row_metric(row[metric], False)
                else:
                    row[metric] = self.update_row_metric(row[metric], True)
            else:
                if "mae" in metric or "mse" in metric:
                    row[metric] = self.update_row_metric(row[metric], False)
                else:
                    row[metric] = self.update_row_metric(row[metric], True)
        return row


class StandardObjectiveOptimizationGenerativeModel(StandardObjectiveOptimization):
    def update_metric_higher_is_better(self, row, evaluation):
        row = row.replace([np.inf], self.fill_inf_nan)
        row = row.replace([np.nan], self.fill_inf_nan)

        for metric in row.columns:
            if evaluation == "statistics":
                if config.DICT_MAPPING_METRICS[metric] == "lower":
                    row[metric] = self.update_row_metric(row[metric], False)
                else:
                    row[metric] = self.update_row_metric(row[metric], True)
            elif evaluation == "ml":  # lower is better
                row[metric] = self.update_row_metric(row[metric], False)
            elif (
                evaluation == "ml_augment"
            ):  # higher is better except for regression task (mae + mse)
                if "mae" in metric or "mse" in metric:
                    row[metric] = self.update_row_metric(row[metric], False)
                else:
                    row[metric] = self.update_row_metric(row[metric], True)
            elif evaluation == "dp":
                higher_or_lower = self.get_value_from_metric(
                    metric, config.DICT_MAPPING_METRICS
                )
                if higher_or_lower == "lower":
                    row[metric] = self.update_row_metric(row[metric], False)
                else:
                    row[metric] = self.update_row_metric(row[metric], True)
        return row


def is_project_exist(project_path):
    return os.path.exists(project_path)


def load_project(project_path, trials_step=1, is_print=True):
    """
    trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
    """
    if is_project_exist(project_path):
        is_loaded = False
        while not is_loaded:
            try:
                time_wait = random.random() * 0.1
                time.sleep(time_wait)
                if is_print:
                    print(
                        colored(f"Found saved Trials! Loading {project_path}", "yellow")
                    )
                trials = pickle.load(open(project_path, "rb"))
                is_loaded = True
            except:
                is_loaded = False

    else:
        if is_print:
            print(colored(f"Create new project", "yellow"))
        trials = Trials()

    return trials


def merge_trials(master_trials, trials_new_run):
    max_tid = max([trial["tid"] for trial in master_trials.trials])

    for trial in trials_new_run:
        tid = trial["tid"] + max_tid + 1
        hyperopt_trial = Trials().new_trial_docs(
            tids=[None], specs=[None], results=[None], miscs=[None]
        )
        hyperopt_trial[0] = trial
        hyperopt_trial[0]["tid"] = tid
        hyperopt_trial[0]["misc"]["tid"] = tid
        for key in hyperopt_trial[0]["misc"]["idxs"].keys():
            hyperopt_trial[0]["misc"]["idxs"][key] = [tid]
        master_trials.insert_trial_docs(hyperopt_trial)
        master_trials.refresh()
    return master_trials


def write_lockfile_multiple_jobs(trials, project_path):
    lockfile = project_path + ".lock"
    with open(lockfile, "w") as f:
        while True:
            try:
                # Try to acquire the lock
                os.mkdir(lockfile + ".lock")
                break
            except OSError:
                # If the lock is already held, wait and try again
                time.sleep(random.random())

        # Do some work on the shared file
        with open(project_path, "wb") as f:
            pickle.dump(trials, f)

        # Release the lock
        os.rmdir(lockfile + ".lock")


def save_project(project_path, trials, trials_step=1):
    """When multiple jobs are reading and writing to the same file, it can lead to errors
    such as race conditions, deadlocks, and data corruption. To avoid these issues,
    we can use a file lock to ensure that only one job can access the file at a time.
    """
    if is_project_exist(project_path):
        print(colored(f"Merge master and new run", "green"))
        master_trials = load_project(project_path, trials_step=1)
        master_trials = merge_trials(master_trials, trials.trials[-trials_step:])
        print(colored(f"Save merge trials", "green"))
        is_saved = False
        while not is_saved:
            try:
                write_lockfile_multiple_jobs(master_trials, project_path)
                is_saved = True
            except:
                is_saved = False

        return len(master_trials.trials)
    else:
        print(colored(f"Save new trial", "green"))
        write_lockfile_multiple_jobs(trials, project_path)
        return len(trials.trials)


def get_number_successful_trials(trials, success_value="success"):
    results = trials.results
    s = 0
    for i in range(len(results)):
        if results[i]["reason"] == success_value:
            s = s + 1
    print(colored(f"Number of successful trials: {s}", "green"))
    return s


def get_number_trials(trials):
    return len(trials.results)


def run_trials(
    project_path,
    objective,
    space,
    trials_step=1,
    algo="tpe",
    ckpt_save=2_000,
):
    # load the trials object
    trials = load_project(project_path)

    # initialize
    max_trials = len(trials.trials) + trials_step
    print(
        colored(
            "Continue from {} trials to {} (+{}) trials".format(
                len(trials.trials), max_trials, trials_step
            ),
            "yellow",
        )
    )
    n_trials = len(trials.trials)

    if algo == "tpe":
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=n_trials + 1,
            trials=trials,
        )
    elif algo == "random":
        best = fmin(
            fn=objective,
            space=space,
            algo=hyperopt.rand.suggest,
            max_evals=n_trials + 1,
            trials=trials,
        )

    print("Best:", best)

    # save the trials object
    n_trials = save_project(project_path, trials)
    n_successful_trials = get_number_successful_trials(trials, success_value="success")

    # save a ckpt for verifying
    if n_trials % ckpt_save == 0:
        filename_ckpt = (
            path_utils.get_filename_without_extension(project_path)
            + f"_{n_trials:07}.hyperopt"
        )
        parent_dir = path_utils.get_parent_dir(project_path)
        ckpt_project_path = os.path.join(parent_dir, filename_ckpt)
        save_project(ckpt_project_path, trials)

    return n_trials, n_successful_trials


def get_best_set_params(project_path):
    trials = load_project(project_path, is_print=False)

    fmin = np.inf
    trial_fmin = None
    for trial in trials.results:
        # print(trial["loss"])
        if trial["loss"] < fmin:
            trial_fmin = trial
            fmin = trial["loss"]

    return trial_fmin


def test_objective(params):
    x = params["x"]
    y = params["y"]
    return {"loss": x**2 - y**3, "status": STATUS_OK}


def test():
    # loop indefinitely and stop whenever you like
    is_continue = True
    max_trials = 10

    # search space
    space = {"x": hp.uniform("x", 0, 10), "y": hp.uniform("y", -10, 10)}

    while is_continue:
        print(colored("=" * 100, "red"))
        n_trials, n_successful_trials = run_trials(
            objective=test_objective, space=space
        )
        if n_trials >= max_trials:
            is_continue = False
            print(colored("=" * 100, "red"))
            print(colored("Done", "red"))
            print(colored("=" * 100, "red"))


if __name__ == "__main__":
    path = "database/optimization_ml_method/abalone_bagging.hyperopt"
    I = IncrementalObjectiveOptimization(path)
    I.update_trials_losses([])

from comet_ml import API
from database.confidential.pwd import Confidential
from pydose_rt.engine.config import config as PARAMS


def log_args_to_comet(args, experiment):
    """Logs all arguments in args to Comet."""
    for arg_name, arg_value in vars(args).items():
        experiment.log_parameter(arg_name, arg_value)


def log_dict_to_met(d, experiment):
    experiment.log_parameters(parameters=d)


def count_experiments_with_name(
    api_key,
    workspace,
    project_name,
    experiment_name_to_count,
):
    """
    Counts the number of experiments with a specific name in a Comet project.

    Args:
        api_key: Your Comet API key.
        project_name: The name of your Comet project.
        experiment_name_to_count: The experiment name to search for.
        workspace: The workspace where your project resides. If None, it will search in all workspaces.

    Returns:
        The number of experiments with the specified name.
    """
    api = API(api_key=api_key)
    experiments = api.get_experiments(workspace, project_name)

    count = 0
    for exp in experiments:
        if experiment_name_to_count in exp._name:
            count += 1
    return count


# ===========================================================================
# main
# ===========================================================================
if __name__ == "__main__":
    # Example Usage:
    api_key = Confidential.comet_api_key
    project_name = PARAMS.comet_project_name
    workspace = PARAMS.comet_workspace
    experiment_name_to_count = "stiff_redshift_6106"  # replace with your target name
    # experiment_name_to_count = "abc"  # replace with your target name

    experiment_count = count_experiments_with_name(
        api_key, project_name, experiment_name_to_count, workspace
    )
    print(
        f"Number of experiments with name '{experiment_name_to_count}': {experiment_count}"
    )

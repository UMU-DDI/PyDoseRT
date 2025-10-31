import os
import json
import pickle
import datetime
import argparse


def get_string_datetime():
    now = datetime.datetime.now()
    if now.month < 10:
        month_string = "0" + str(now.month)
    else:
        month_string = str(now.month)
    if now.day < 10:
        day_string = "0" + str(now.day)
    else:
        day_string = str(now.day)
    yearmonthdate_string = str(now.year) + month_string + day_string
    return yearmonthdate_string


def write_list_to_file(my_list, path):
    with open(path, "w+") as f:
        for item in my_list:
            f.write("%s\n" % item)


def read_file_to_list(path):
    with open(path, "r") as f:
        x = f.readlines()
    return x


def write_pickle(data, path):
    with open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    return data


def convert_args_to_dict(args):
    return vars(args)


# Convert Tuple String to Integer Tuple
# Using tuple() + int() + replace() + split()
def convert_string_to_tuple(s):
    res = tuple(
        int(num)
        for num in s.replace("(", "").replace(")", "").replace(" ", "").split(",")
    )
    return res


def convert_to_tuple(value, n=2):
    """Converts a single value to a tuple of n identical values.

    Args:
      value: The value to be converted.
      n: The number of times to repeat the value.

    Returns:
      A tuple of n identical values.
    """

    return (value,) * n


def args_to_json(args, folder_path, filename="args.json"):
    """
    Converts argparse arguments to JSON and saves them to a file.

    Args:
        args: argparse.Namespace object containing the arguments.
        folder_path: Path to the folder where the JSON file will be saved.
        filename: Name of the JSON file (default: "args.json").
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create the folder if it doesn't exist

    file_path = os.path.join(folder_path, filename)

    with open(file_path, "w") as f:
        json.dump(vars(args), f, indent=4)  # Convert to dict and dump as JSON

    print(f"Arguments saved to {file_path}")


def parse_constraints(parser, constraints):
    """
    Adds command-line arguments to the parser using the keys and default values
    from the provided constraints dictionary.

    Parameters:
      parser (argparse.ArgumentParser): The parser to which arguments will be added.
      constraints (dict): A dictionary containing constraint sub-dictionaries.
          Expected keys are: "lower_bound_gy", "higher_bound_gy",
                             "lower_bound_target_percent",
                             "higher_bound_target_percent", and "weight".
          Each sub-dictionary should map region names (e.g. "PTV", "ROI1", etc.)
          to a default value.

    Returns:
      argparse.ArgumentParser: The parser with the additional arguments.
    """
    # Iterate over each constraint type and its sub-dictionary.
    for constraint_type, region_dict in constraints.items():
        for region, default_value in region_dict.items():
            arg_name = f"--{constraint_type}_{region}"
            # Use the type of the default_value for the argument type.
            parser.add_argument(
                arg_name,
                type=type(default_value),
                default=default_value,
                help=f"{constraint_type} for {region} (default: {default_value})",
            )
    return parser


def construct_constraints(args, constraints_keys):
    constructed = {}
    for key in constraints_keys:
        constructed[key] = {}
        for region in constraints_keys[key]:
            arg_name = f"{key}_{region}"
            constructed[key][region] = getattr(args, arg_name)
    return constructed


def str_to_tuple(arg_str):
    """Converts a comma-separated string to a tuple of integers."""
    try:
        # Remove any parentheses and whitespace, then split by comma
        cleaned_str = arg_str.strip("() ")
        return tuple(map(int, cleaned_str.split(",")))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid tuple format: '{arg_str}'. "
            "Please use a comma-separated string like '2,2,2'."
        )

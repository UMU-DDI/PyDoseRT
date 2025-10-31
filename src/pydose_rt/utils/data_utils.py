import inspect


def append_to_history(loss_dict, value):
    """Appends the given value to the corresponding key in the loss history dictionary."""
    # Get the name of the variable passed as `value`
    frame = inspect.currentframe().f_back
    var_name = [name for name, val in frame.f_locals.items() if val is value]

    if not var_name:
        raise ValueError(
            "Could not determine variable name. Make sure the variable is directly passed."
        )

    key = var_name[0]  # Get the first matching variable name

    if key in loss_dict:
        loss_dict[key].append(value)
    else:
        raise KeyError(f"Key '{key}' not found in the loss history dictionary.")

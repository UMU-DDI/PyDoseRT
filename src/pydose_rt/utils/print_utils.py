def print_section(print_string):
    print("\n" * 4)
    print("=" * 100)
    print("working on:", print_string)
    print("=" * 100)


def print_processing(print_string):
    print()
    print(">> processing", print_string)


def print_separator():
    # print("\n")
    print("-" * 100)


def print_training_summary(model, config):
    print("\n" * 2)
    print("=" * 100)
    print("TRAINING SUMMARY")
    print("=" * 100)
    print("project:", config["project"])
    print("model: {} \t dimension: {}".format(config["model"], config["model_dim"]))
    try:
        print(
            "model input: {} \t model output: {}".format(
                model.input.get_shape().as_list(), model.output.get_shape().as_list()
            )
        )
    except:
        pass
    print("-" * 100)
    print(
        "number of train: {} \t val: {} \t test: {}".format(
            config["n_training_patient"],
            config["n_validation_patient"],
            config["n_testing_patient"],
        )
    )

    print("training on labels:", config["labels"])
    print("-" * 100)
    print(
        "initial learning rate: {} \t learning rate drop: {}".format(
            config["initial_learning_rate"], config["learning_rate_drop"]
        )
    )
    print("-" * 100)
    print("data file:", config["data_file"])
    print("model file:", config["model_file"])
    print("training file:", config["training_file"])
    print("validation file:", config["validation_file"])
    print("testing file:", config["testing_file"])
    print("=" * 100)


def beautify_print(
    iteration,
    total_loss,
    loss_lower_bound_gy,
    loss_higher_bound_gy,
    loss_lower_bound_target,
    loss_higher_bound_target,
    l2_loss_oars_and_background,
    elapsed_time_epoch,
    total_iterations,
):
    """Beautifies the print statement with fixed character lengths."""

    iteration_str = f"{iteration + 1:>{len(str(total_iterations))}}/{total_iterations}"
    total_loss_str = f"{total_loss.numpy():>10.6f}"
    loss_lower_bound_gy_str = f"{loss_lower_bound_gy.numpy():>10.6f}"
    loss_higher_bound_gy_str = f"{loss_higher_bound_gy.numpy():>10.6f}"
    loss_lower_bound_target_str = f"{loss_lower_bound_target.numpy():>10.6f}"
    loss_higher_bound_target_str = f"{loss_higher_bound_target.numpy():>10.6f}"
    l2_loss_str = f"{l2_loss_oars_and_background.numpy():>10.6f}"
    elapsed_time_str = f"{elapsed_time_epoch:>6.2f}"

    print(
        f"Iteration {iteration_str} \t Total Loss: {total_loss_str} \t L.Gy: {loss_lower_bound_gy_str} \t H.Gy: {loss_higher_bound_gy_str} \t L.Tar: {loss_lower_bound_target_str} \t H.Tar: {loss_higher_bound_target_str} \t L2: {l2_loss_str} \t Epoch Time: {elapsed_time_str}"
    )

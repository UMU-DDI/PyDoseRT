import glob
import engine.utils.path_utils as path_utils
import random
import os
import pprint


pp = pprint.PrettyPrinter(indent=4)


# mode_run: 0 - just run
# mode_run: 1 - run if file not exists
# mode_run: 2 - run if file not exists and no gpu is running
# mode_run: 3 - run if no gpu is running
MODE_RUN = 3


def run(filename_or_directory, cmd, mode_run=0):
    # mode_run: 0 - just run
    # mode_run: 1 - run if file not exists
    # mode_run: 2 - run if file not exists and no gpu is running
    # mode_run: 3 - run if no gpu is running

    storage_dir = "database"

    def is_running_in_another_gpu(model_filename):
        path_loop = os.path.join(storage_dir, "loop")
        path_utils.make_dir(path_loop)
        path = os.path.join(storage_dir, "loop/running.txt")
        is_running = False
        if not os.path.exists(path):
            f = open(path, "w")
            f.write("{}\n".format(model_filename))
            f.close()
        else:
            f = open(path, "r")
            lines = f.readlines()
            f.close()
            for line in lines:
                if model_filename in line:
                    is_running = True
            if not is_running:
                f = open(path, "a+")
                f.write("{}\n".format(model_filename))
                f.close()
        return is_running

    def is_model_trained(model_path):
        folder = path_utils.get_parent_dir(model_path)
        model_name = path_utils.get_filename(model_path)
        is_found = False
        if os.path.exists(folder):
            list_files = glob.glob(os.path.join(folder, "*"))
            for path in list_files:
                if model_name in path:
                    is_found = True
                    break
        return is_found

    print("=" * 120)
    model_path = filename_or_directory

    is_run = False
    if mode_run == 0:
        is_run = True
    elif mode_run == 1 and not is_model_trained(model_path):
        is_run = True
    elif (
        mode_run == 2
        and not is_model_trained(model_path)
        and not is_running_in_another_gpu(filename_or_directory)
    ):
        is_run = True
    # finetune
    elif mode_run == 3 and not is_running_in_another_gpu(filename_or_directory):
        is_run = True

    if is_run:
        os.system(cmd)
    else:
        print("{} exists or in training. Will skip!!".format(model_path))

    return is_run


def is_added(model_filename, list_train=None):
    is_add = False
    if list_train is None:
        is_add = True
    else:
        for model in list_train:
            if model in model_filename:
                is_add = True
    return is_add


def run_loop(args):
    model_list = list()
    cmd_list = list()

    # full_filters = get_full_list_filters()

    filters = [
        "db3",
        "haar",
    ]
    compression_methods = ["wt"]
    compression_ratios = [0.1, 0.3, 0.6, 0.9, 0.99, 0.999]
    for model_name in [get_model(args.dataset, is_compressed=True)]:
        for compression_method in compression_methods:
            for wave in filters:
                for n_levels in [3]:
                    for compressed_blocks in get_list_compressed_blocks(args.dataset):
                        for compression_ratio in compression_ratios:
                            cmd, model_filename = generate_yaml(
                                dataset=args.dataset,
                                model_name=model_name,
                                compression_method=compression_method,
                                compression_parameters={
                                    "wave": wave,
                                    "compression_ratio": compression_ratio,
                                    "n_levels": n_levels,
                                },
                                compressed_blocks=compressed_blocks,
                                is_exp_epoch_time=args.is_exp_epoch_time,
                            )

                            if is_added(model_filename, LIST_TRAIN):
                                model_list.append(model_filename)
                                cmd_list.append(cmd)

    combined = list(zip(model_list, cmd_list))
    random.shuffle(combined)
    model_list, cmd_list = zip(*combined)

    count_time_running = 0

    for i in range(len(model_list)):
        model_filename = model_list[i]
        cmd = cmd_list[i]
        is_run = run(filename_or_directory=model_filename, cmd=cmd, mode_run=MODE_RUN)

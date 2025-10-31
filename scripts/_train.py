from comet_ml import Experiment
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # 0 = all messages; 1 = filter out INFO; 2 = filter out INFO and WARNING; 3 = only errors
)

import time

# from rich import print
# from rich.pretty import Pretty
import json
import argparse
import numpy as np
import uuid
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

tf.get_logger().setLevel("ERROR")


# from models.data import DataGenerator
import engine.utils.test_utils as test_utils
from pydose_rt.engine.config import config as PARAMS
from pydose_rt.engine._data import DataGenerator

# from models.layers import FluenceMapLayer
from pydose_rt.engine.model import build_unet_without_dose_layer, build_full_model
from pydose_rt import ModelConfig
from pydose_rt.engine.utils.plot_utils import Plotter
from pydose_rt.engine._loss import (
    dose_loss,
    auxiliary_loss,
    TrainableLossWeightsNormalized,
    leafs_loss,
    mus_loss,
)
from database.confidential.pwd import Confidential
import engine.utils.tf_utils as tf_utils
import engine.utils.path_utils as path_utils
import engine.utils.comet_utils as comet_utils
import engine.utils.io_utils as io_utils
import engine.utils.data_utils as data_utils


# tf.debugging.set_log_device_placement(True)  # Log device placement of operations
tf.debugging.set_log_device_placement(False)  # Log device placement of operations


# ======================================================================================
# Arguments
# ======================================================================================
# region
parser = argparse.ArgumentParser(description="Welcome.")
parser.add_argument("--lr", default=0.0001)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--epochs", default=150, type=int)
parser.add_argument("--kernel_size", default=15)
parser.add_argument("--number_of_cps", default=9, type=int)
parser.add_argument("--number_of_leaf_pairs", default=60, type=float)
parser.add_argument("--downsampling_factor", type=int, nargs=3, default=(2, 2, 4))
parser.add_argument("--gpu", default=0)

parser.add_argument("--is_debug", default=1, type=int)
parser.add_argument("--is_comet", default=0, type=int)
parser.add_argument("--verbose", default=0, type=int)
parser.add_argument("--memory_limit_mb", default=None, type=int)  # try to set to 12_288
parser.add_argument("--initial_filters", default=16, type=int)
parser.add_argument("--num_classes", default=1, type=int)
parser.add_argument("--depth", default=4, type=int)
parser.add_argument("--vae_n_filters", default=16, type=int)
parser.add_argument("--is_normalize_weight", default=0, type=int)
parser.add_argument(
    "--constraint_mode",
    default="fixed",
    type=str,
    choices=[
        "fixed",  # fix everything
        "rconstraint",  # random a single constraint from 8 predefined
        "rweight",  # fix bounds and random weight
    ],
)

parser.add_argument("--is_load_pretrained", default=0, type=int)
parser.add_argument("--epoch_start", default=0, type=int)


parser = io_utils.parse_constraints(parser, PARAMS.constraints)
args = parser.parse_args()
constraints_keys = {
    "lower_bound_gy": list(PARAMS.constraints["lower_bound_gy"].keys()),
    "higher_bound_gy": list(PARAMS.constraints["higher_bound_gy"].keys()),
    "lower_bound_target_percent": list(
        PARAMS.constraints["lower_bound_target_percent"].keys()
    ),
    "higher_bound_target_percent": list(
        PARAMS.constraints["higher_bound_target_percent"].keys()
    ),
    "weight": list(PARAMS.constraints["weight"].keys()),
}


final_constraints = io_utils.construct_constraints(args, constraints_keys)
print("args.is_debug:", args.is_debug)

if args.is_debug:
    args.verbose = 1
if args.is_debug == 2:
    args.verbose_gen = 1
    args.initial_filters = 1
    args.vae_n_filters = 1
else:
    args.verbose_gen = 0


# endregion


# ======================================================================================
# Setup tensorflow vram usage
# ======================================================================================
# region
if args.memory_limit_mb is not None:
    tf_utils.set_fixed_gpu_memory_limit_mb(args.memory_limit_mb)
# endregion


# ======================================================================================
# Setup comet and logging
# ======================================================================================
# region
experiment_name = path_utils.get_folder(args)  # name experiment
if args.is_comet:
    project_name = (
        "autoplan-test" if args.is_debug else f"autoplan-{args.constraint_mode}"
    )
    experiment = Experiment(
        api_key=Confidential.comet_api_key, project_name=project_name, disabled=False
    )
    count_comet = comet_utils.count_experiments_with_name(
        Confidential.comet_api_key,
        PARAMS.comet_workspace,
        PARAMS.comet_project_name,
        experiment_name + "_",
    )
else:
    experiment = None
    count_comet = 0

path_utils.make_dir("database/experiments/")
count_dir = sum(
    1
    for item in os.listdir("database/experiments/")
    if os.path.isdir(os.path.join("database/experiments/", item))
    and experiment_name + "_" in item
)
count_existing_experiment = (lambda a, b: a if a > b else b)(count_comet, count_dir)
experiment_name = f"{experiment_name}_{count_existing_experiment+1:02d}"


if args.is_debug:
    experiment_name = f"test_{experiment_name}"


args.experiment_name = experiment_name
args.data_path = "/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/DATASETS/ARTP/"
args.data_path = "database/AUTORPT/"

args.dir_output = f"database/experiments/{experiment_name}"
args.constraints = final_constraints

try:  # bug multiple processes make dir
    path_utils.make_dir(args.dir_output)
except:
    pass

if args.is_comet:
    experiment.set_name(experiment_name)
    comet_utils.log_args_to_comet(args, experiment)

io_utils.args_to_json(args, args.dir_output)
# endregion


# ======================================================================================
# Data generator
# ======================================================================================
# region
gen = DataGenerator(
    args.data_path,
    "training",
    True,
    int(args.batch_size),
    constraints=args.constraints,
    is_debug=args.is_debug,
    weight_ptv=args.weight_PTV,
    constraint_mode=args.constraint_mode,
    is_normalize_weight=args.is_normalize_weight,
    verbose=args.verbose_gen,
)
gen_val = DataGenerator(
    args.data_path,
    "validating",
    False,
    int(args.batch_size),
    constraints=args.constraints,
    is_debug=args.is_debug,
    weight_ptv=args.weight_PTV,
    constraint_mode=args.constraint_mode,
    is_normalize_weight=args.is_normalize_weight,
    verbose=args.verbose_gen,
)

list_gen_plot = []
if args.constraint_mode == "rconstraint":
    list_constraints_to_plot = PARAMS.list_constraints
    for c in list_constraints_to_plot:
        list_gen_plot.append(
            DataGenerator(
                args.data_path,
                "plotting",
                False,
                1,
                constraints=c,
                weight_ptv=args.weight_PTV,  # force weight PTV
                is_normalize_weight=args.is_normalize_weight,
                verbose=args.verbose_gen,
            )
        )
elif args.constraint_mode == "rweight":
    list_constraints_to_plot = PARAMS.list_constraints_weight
    for c in list_constraints_to_plot:
        list_gen_plot.append(
            DataGenerator(
                args.data_path,
                "plotting",
                False,
                1,
                constraints=c,
                weight_ptv=None,  # already have weight
                is_normalize_weight=args.is_normalize_weight,
                verbose=args.verbose_gen,
            )
        )
elif args.constraint_mode == "fixed":
    list_constraints_to_plot = [args.constraints]
    for c in list_constraints_to_plot:
        list_gen_plot.append(
            DataGenerator(
                args.data_path,
                "plotting",
                False,
                1,
                constraints=c,
                weight_ptv=None,  # already have weight
                is_normalize_weight=args.is_normalize_weight,
                verbose=args.verbose_gen,
            )
        )
else:
    raise ValueError

# endregion


# ======================================================================================
# Model config
# ======================================================================================
# region
# The input CT is always 128x128x320. To give the model physical dimensions, we need to know the resolution of the CT.
config = ModelConfig(
    ct_array_shape=(128, 128, 320),
    downsampling_factor=tuple(args.downsampling_factor),
    resolution=(0.3, 0.3, 0.3),
    field_size=(40, 40),
    number_of_leaf_pairs=float(args.number_of_leaf_pairs),
    tpr_20_10=0.72,
    number_of_cps=int(args.number_of_cps),
)
# endregion


# ======================================================================================
# Plotter
# ======================================================================================
# region
list_plotter = []
for gen_plot in list_gen_plot:
    plotter = Plotter(
        config,
        gen_plot,
        experiment,
        int(args.number_of_cps),
        fluence_layer=None,
        num_fluence_plots=int(args.number_of_cps),
        scaled_mu=1,
        out_path=args.dir_output,
    )
    list_plotter.append(plotter)
# endregion


# ======================================================================================
# Model
# ======================================================================================
# region
latest_model_path = tf_utils.get_latest_model_path(experiment_name)
if args.is_load_pretrained and latest_model_path is not None:
    model_without_dose_layer = tf_utils.load_model(latest_model_path)
    args.epoch_start = 1  # we want to start regularizing

    # Check and unfreeze layers in model_without_dose_layer if needed
    for layer in model_without_dose_layer.layers:
        layer.trainable = True
        # print(f"Unfreezing layer: {layer.name}")


else:
    model_without_dose_layer = build_unet_without_dose_layer(
        input_shape=PARAMS.input_shape,
        kernel_size=int(args.kernel_size),
        batch_size=int(args.batch_size),
        depth=args.depth,
        initial_filters=args.initial_filters,
        config=config,
        vae_n_filters=args.vae_n_filters,
        is_transformer_vae=False,
    )

model = build_full_model(
    model_without_dose_layer,
    config=config,
    kernel_size=int(args.kernel_size),
    depth=args.depth,
    initial_filters=args.initial_filters,
    use_bypass=True,
)

print()
print()
print()
model_without_dose_layer.summary()

print()
print()
print()
model.summary()


print()
print()
print()
# endregion


# ======================================================================================
# Training
# ======================================================================================
# region
opt = Adam(learning_rate=float(args.lr))
val_min = np.inf
patience = 40
patience_counter = 0
plot_every = 1
loss_weight_layer = TrainableLossWeightsNormalized(num_losses=10)

# After building the full model, you should compile it with your optimizer and loss
model.compile(optimizer=opt)


for epoch in range(args.epoch_start, args.epochs, 1):
    train_loss_history = {
        "total_loss": [],
        "loss_lower_bound_gy": [],
        "loss_higher_bound_gy": [],
        "loss_lower_bound_target": [],
        "loss_higher_bound_target": [],
        "l2_loss_oars_and_background": [],
        "aux_loss": [],
        "mu_rate_loss": [],
        "mu_complexity_loss": [],
        "leaf_opening_loss": [],
        "leaf_rate_loss": [],
    }

    gen.set_epoch(epoch)
    gen_val.set_epoch(epoch)

    if args.verbose:
        print()
        print()
        print()
        print("-" * 150)
        print("Epoch: ", epoch)
        print("-" * 150)

    for i in range(len(gen)):
        x, y_dose, masks, region_weights, constraints = gen[i]
        start_time_epoch = time.time()

        with tf.GradientTape() as tape:
            try:
                # print(json.dumps(constraints, indent=4))

                dose_pred, leafs, mus, dose_bypass = model(x)
                if args.verbose:
                    print()
                    print(
                        "\t\t\t\t\t\t\t\t\t leafs:",
                        f"{leafs.numpy().min():.4f}",
                        f"{leafs.numpy().max():.4f}",
                        f"{leafs.numpy().mean():.4f}",
                        "\t\t mus:",
                        f"{mus.numpy().min():.4f}",
                        f"{mus.numpy().max():.4f}",
                        f"{mus.numpy().mean():.4f}",
                    )
                    # print("-" * 100)
                    print()
            except:
                dose_pred = model(x)

            (
                loss_lower_bound_gy,
                loss_higher_bound_gy,
                loss_lower_bound_target,
                loss_higher_bound_target,
                l2_loss_oars_and_background,
            ) = dose_loss(x, dose_pred, constraints, masks, region_weights, None)
            aux_loss = auxiliary_loss(dose_pred, dose_bypass)

            mu_rate_loss, mu_complexity_loss = mus_loss(mus, config)
            leaf_opening_loss, leaf_rate_loss = leafs_loss(leafs, config)

            weight_leaf_mu_loss = min(epoch * 1, 1)

            leaf_opening_loss, leaf_rate_loss = (
                weight_leaf_mu_loss * leaf_opening_loss,
                weight_leaf_mu_loss * leaf_rate_loss,
            )

            mu_rate_loss, mu_complexity_loss = (
                0 * mu_rate_loss,
                0 * mu_complexity_loss,
            )

            total_loss = loss_weight_layer(
                [
                    loss_lower_bound_gy,
                    loss_higher_bound_gy,
                    loss_lower_bound_target,
                    loss_higher_bound_target,
                    l2_loss_oars_and_background,
                    aux_loss,
                    mu_rate_loss,
                    mu_complexity_loss,
                    leaf_opening_loss,
                    leaf_rate_loss,
                ]
            )

            if tf.reduce_any(tf.math.is_nan(total_loss)):
                tensors_to_check = {
                    "constraints": constraints,
                    "region_weights": region_weights,
                    "loss_lower_bound_gy": loss_lower_bound_gy,
                    "loss_higher_bound_gy": loss_higher_bound_gy,
                    "loss_lower_bound_target": loss_lower_bound_target,
                    "loss_higher_bound_target": loss_higher_bound_target,
                    "l2_loss_oars_and_background": l2_loss_oars_and_background,
                    "aux_loss": aux_loss,
                    "mu_rate_loss": mu_rate_loss,
                    "mu_complexity_loss": mu_complexity_loss,
                    "leaf_opening_loss": leaf_opening_loss,
                    "leaf_rate_loss": leaf_rate_loss,
                }

                test_utils.check_for_nan(**tensors_to_check)

            grads = tape.gradient(
                total_loss,
                model.trainable_variables + loss_weight_layer.trainable_variables,
            )
        opt.apply_gradients(
            zip(
                grads, model.trainable_variables + loss_weight_layer.trainable_variables
            )
        )

        train_loss_history["total_loss"].append(total_loss.numpy())
        train_loss_history["loss_lower_bound_gy"].append(loss_lower_bound_gy.numpy())
        train_loss_history["loss_higher_bound_gy"].append(loss_higher_bound_gy.numpy())
        train_loss_history["loss_lower_bound_target"].append(
            loss_lower_bound_target.numpy()
        )
        train_loss_history["loss_higher_bound_target"].append(
            loss_higher_bound_target.numpy()
        )
        train_loss_history["l2_loss_oars_and_background"].append(
            l2_loss_oars_and_background.numpy()
        )
        train_loss_history["aux_loss"].append(aux_loss.numpy())
        train_loss_history["mu_rate_loss"].append(mu_rate_loss.numpy())
        train_loss_history["mu_complexity_loss"].append(mu_complexity_loss.numpy())
        train_loss_history["leaf_opening_loss"].append(leaf_opening_loss.numpy())
        train_loss_history["leaf_rate_loss"].append(leaf_rate_loss.numpy())

        end_time_epoch = time.time()  # Record end time for entire epoch (if applicable)
        elapsed_time_epoch = end_time_epoch - start_time_epoch

        print_str = f"Iteration {i + 1}/{len(gen)} \t {PARAMS.print_dict['total_loss']}: {np.mean(train_loss_history['total_loss']):.5f}"
        for key, value in train_loss_history.items():
            if key != "total_loss":
                print_str += f" \t {PARAMS.print_dict[key]}: {np.mean(value):.5f}"
        print_str += f" \t time: {elapsed_time_epoch:.2f}"
        print(print_str)

    print(f"Epoch {epoch} - Train: {np.mean(train_loss_history['total_loss'])}")
    # gen.on_epoch_end()  # shuffle and recompute p_augment

    tf_utils.save_model(
        model_without_dose_layer,
        path=os.path.join(
            args.dir_output, f"model_without_dose_layer_{epoch:02d}.keras"
        ),
    )

    val_loss_history = {
        "total_loss": [],
        "loss_lower_bound_gy": [],
        "loss_higher_bound_gy": [],
        "loss_lower_bound_target": [],
        "loss_higher_bound_target": [],
        "l2_loss_oars_and_background": [],
        "aux_loss": [],
        "mu_rate_loss": [],
        "mu_complexity_loss": [],
        "leaf_opening_loss": [],
        "leaf_rate_loss": [],
    }

    for i in range(len(gen_val)):
        x, y_dose, masks, region_weights, constraints = gen_val[i]
        try:
            dose_pred, leafs, mus, dose_bypass = model(x)
        except:
            dose_pred = model(x)

        (
            loss_lower_bound_gy,
            loss_higher_bound_gy,
            loss_lower_bound_target,
            loss_higher_bound_target,
            l2_loss_oars_and_background,
        ) = dose_loss(x, dose_pred, constraints, masks, region_weights, None)
        aux_loss = auxiliary_loss(dose_pred, dose_bypass)

        mu_rate_loss, mu_complexity_loss = mus_loss(leafs, config)
        leaf_opening_loss, leaf_rate_loss = leafs_loss(leafs, config)

        weight_leaf_mu_loss = min(epoch * 0.01, 0.1)

        leaf_opening_loss, leaf_rate_loss = (
            weight_leaf_mu_loss * leaf_opening_loss,
            weight_leaf_mu_loss * leaf_rate_loss,
        )

        mu_rate_loss, mu_complexity_loss = (
            0 * mu_rate_loss,
            0 * mu_complexity_loss,
        )

        total_loss = loss_weight_layer(
            [
                loss_lower_bound_gy,
                loss_higher_bound_gy,
                loss_lower_bound_target,
                loss_higher_bound_target,
                l2_loss_oars_and_background,
                aux_loss,
                mu_rate_loss,
                mu_complexity_loss,
                leaf_opening_loss,
                leaf_rate_loss,
            ]
        )

        # total_loss = (
        #     loss_lower_bound_gy
        #     + loss_higher_bound_gy
        #     + loss_lower_bound_target
        #     + loss_higher_bound_target
        #     + aux_loss
        #     + mu_rate_loss
        #     + mu_complexity_loss
        #     + leaf_opening_loss
        #     + leaf_rate_loss
        #     # + l2_loss_oars_and_background
        # )

        val_loss_history["total_loss"].append(total_loss.numpy())
        val_loss_history["loss_lower_bound_gy"].append(loss_lower_bound_gy.numpy())
        val_loss_history["loss_higher_bound_gy"].append(loss_higher_bound_gy.numpy())
        val_loss_history["loss_lower_bound_target"].append(
            loss_lower_bound_target.numpy()
        )
        val_loss_history["loss_higher_bound_target"].append(
            loss_higher_bound_target.numpy()
        )
        val_loss_history["l2_loss_oars_and_background"].append(
            l2_loss_oars_and_background.numpy()
        )
        val_loss_history["aux_loss"].append(aux_loss.numpy())
        val_loss_history["mu_rate_loss"].append(mu_rate_loss.numpy())
        val_loss_history["mu_complexity_loss"].append(mu_complexity_loss.numpy())
        val_loss_history["leaf_opening_loss"].append(leaf_opening_loss.numpy())
        val_loss_history["leaf_rate_loss"].append(leaf_rate_loss.numpy())

    print(f"Epoch {epoch} - Val: {np.mean(val_loss_history['total_loss'])}")

    if experiment is not None:
        log_dict = {}
        log_dict["epoch"] = epoch
        for key, value in train_loss_history.items():
            log_dict[key] = np.mean(value)
        for key, value in val_loss_history.items():
            log_dict[f"val_{key}"] = np.mean(value)

        experiment.log_metrics(log_dict, epoch=epoch)

    plot_freq = 5
    if epoch % plot_freq == 0:
        for i_plotter, plotter in enumerate(list_plotter):
            plotter.make(model, epoch, text_constraint=f"c{i_plotter+1}")

    # if np.mean(val_loss_history["total_loss"]) < val_min:
    #     val_min = np.mean(val_loss_history["total_loss"])
    #     patience_counter = 0
    # else:
    #     patience_counter += 1
    #     if patience_counter > patience:
    #         print("Patience threshold reached. Ending training")
    #         if experiment is not None:
    #             experiment.end()
    #         break
# endregion

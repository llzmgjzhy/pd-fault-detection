import logging
from datetime import datetime
import random
import string
import os
from utils import utils
import json
from collections import OrderedDict
import torch
import time
import numpy as np

logger = logging.getLogger("__main__")

NEG_METRICS = {"loss", "mse_loss"}


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    # config = args.__dict__  # configuration dictionary

    # if args.config_filepath is not None:
    #     logger.info("Reading configuration ...")
    #     try:  # dictionary containing the entire configuration settings in a hierarchical fashion
    #         config.update(utils.load_config(args.config_filepath))
    #     except:
    #         logger.critical(
    #             "Failed to load configuration file. Check JSON syntax and verify that files exist"
    #         )
    #         traceback.print_exc()
    #         sys.exit(1)

    # Create output directory
    initial_timestamp = datetime.now()
    if not os.path.isdir(args.output_dir):
        raise IOError(
            f"Root directory '{args.output_dir}', where the directory of the experiment will be created, must exist"
        )

    output_dir = os.path.join(args.output_dir, args.experiment_name)

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    args.initial_timestamp = formatted_timestamp
    if (not args.no_timestamp) or (len(args.experiment_name) == 0):
        rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
        output_dir += f"_{formatted_timestamp}_{rand_suffix}"
    args.output_dir = output_dir
    args.save_dir = os.path.join(output_dir, "checkpoints")
    args.pred_dir = os.path.join(output_dir, "predictions")
    args.tensorboard_dir = os.path.join(output_dir, "tb_summaries")
    utils.create_dirs([args.save_dir, args.pred_dir, args.tensorboard_dir])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, "configuration.json"), "w") as fp:
        json.dump(vars(args), fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return args


def pipeline_factory(config):
    """For the task specified in the configuration returns the corresponding combination of
    Dataset class, collate function and Runner class."""

    task = config.task

    if task == "fault_detection":
        return UnSupervisedRunner
    else:
        raise NotImplementedError("Task '{}' not implemented".format(task))


class BaseRunner(object):

    def __init__(
        self,
        model,
        dataloader,
        device,
        loss_module,
        config,
        optimizer=None,
        l2_reg=None,
        print_interval=10,
        console=True,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.config = config
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)

        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError("Please override in child class")

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError("Please override in child class")

    def print_callback(self, i_batch, metrics, prefix=""):

        total_batches = len(self.dataloader)

        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class UnSupervisedRunner(BaseRunner):
    def __init__(self, *args, **kwargs):

        super(UnSupervisedRunner, self).__init__(*args, **kwargs)

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        for i, batch in enumerate(self.dataloader):
            self.optimizer.zero_grad()

            X, targets = batch
            X = X.to(device=self.device)
            targets = targets.to(device=self.device)
            predictions = self.model(X)

            loss = self.loss_module(predictions, targets)
            batch_loss = loss.sum()
            mean_loss = loss.mean()

            backward_loss = mean_loss

            backward_loss.backward()
            self.optimizer.step()

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Training " + ending)

            with torch.no_grad():
                total_samples += loss.numel()
                epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = (
            epoch_loss / total_samples
        )  # average loss per sample for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True, stage="val"):

        self.model = self.model.eval()

        epoch_mse_loss = 0  # total loss of epoch
        epoch_mae_loss = 0
        total_samples = 0  # total samples in epoch

        per_batch = {
            "target_masks": [],
            "targets": [],
            "predictions": [],
            "metrics": [],
        }
        for i, batch in enumerate(self.dataloader):

            X, targets, x_mask, y_mask = batch
            X = X.to(device=self.device, dtype=torch.bfloat16)
            targets = targets.to(device=self.device, dtype=torch.bfloat16)
            predictions = self.model(X, x_mask, targets, y_mask, stage, i)

            f_dim = -1 if self.config.features == "MS" else 0
            predictions = predictions[:, -self.config.pred_len :, f_dim:]
            targets = targets[:, -self.config.pred_len :, f_dim:]

            mse_loss = self.loss_module(predictions, targets)
            batch_mse_loss = mse_loss.sum()
            mean_mse_loss = mse_loss.mean()  # mean loss (over samples)
            # (batch_size,) loss for each sample in the batch
            mae_loss = self.mae_criterion(predictions, targets)
            batch_mae_loss = mae_loss.sum()
            mean_mae_loss = mae_loss.mean()

            per_batch["targets"].append(targets.half().cpu().numpy())
            per_batch["predictions"].append(predictions.half().cpu().numpy())
            per_batch["metrics"].append([mse_loss.half().cpu().numpy()])

            metrics = {
                "mse_loss": mean_mse_loss,
                "mae_loss": mean_mae_loss,
            }
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Evaluating " + ending)

            total_samples += mse_loss.numel()
            epoch_mse_loss += (
                batch_mse_loss.half().cpu().item()
            )  # add total loss of batch
            epoch_mae_loss += batch_mae_loss.half().cpu().item()

        epoch_mse_loss = (
            epoch_mse_loss / total_samples
        )  # average loss per element for whole epoch
        epoch_mae_loss = epoch_mae_loss / total_samples
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["mse_loss"] = epoch_mse_loss
        self.epoch_metrics["mae_loss"] = epoch_mae_loss

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics


def validate(
    val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch
):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    logger.info("Evaluating on validation set ...")
    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = val_evaluator.evaluate(
            epoch, keep_all=True, stage="val"
        )
    eval_runtime = time.time() - eval_start_time
    logger.info(
        "Validation runtime: {} hours, {} minutes, {} seconds\n".format(
            *utils.readable_time(eval_runtime)
        )
    )

    global val_times
    val_times["total_time"] += eval_runtime
    val_times["count"] += 1
    avg_val_time = val_times["total_time"] / val_times["count"]
    avg_val_batch_time = avg_val_time / len(val_evaluator.dataloader)
    avg_val_sample_time = avg_val_time / len(val_evaluator.dataloader.dataset)
    logger.info(
        "Avg val. time: {} hours, {} minutes, {} seconds".format(
            *utils.readable_time(avg_val_time)
        )
    )
    logger.info("Avg batch val. time: {} seconds".format(avg_val_batch_time))
    logger.info("Avg sample val. time: {} seconds".format(avg_val_sample_time))

    print()
    print_str = "Epoch {} Validation Summary: ".format(epoch)
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar("{}/val".format(k), v, epoch)
        print_str += "{}: {:8f} | ".format(k, v)
    logger.info(print_str)

    if config.key_metric in NEG_METRICS:
        condition = aggr_metrics[config.key_metric] < best_value
    else:
        condition = aggr_metrics[config.key_metric] > best_value
    if condition:
        best_value = aggr_metrics[config.key_metric]
        if not config.no_savemodel:
            utils.save_model(
                os.path.join(config.save_dir, "model_best.pth"),
                epoch,
                val_evaluator.model,
            )
        best_metrics = aggr_metrics.copy()

        pred_filepath = os.path.join(config.pred_dir, "best_predictions")
        for key in per_batch.keys():
            per_batch[key] = np.array(
                per_batch[key], dtype=object
            )  # resolve issue of inconsistent array sizes
        np.savez(pred_filepath, **per_batch)

    return aggr_metrics, best_metrics, best_value


def test(test_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    logger.info("Testing on test set ...")
    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = test_evaluator.evaluate(
            epoch_num=None, keep_all=True, stage="test"
        )
        del aggr_metrics["epoch"]
    aggr_metrics = {f"test_{key}": value for key, value in aggr_metrics.items()}
    eval_runtime = time.time() - eval_start_time
    logger.info(
        "Testing runtime: {} hours, {} minutes, {} seconds\n".format(
            *utils.readable_time(eval_runtime)
        )
    )

    global val_times
    val_times["total_time"] += eval_runtime
    val_times["count"] += 1
    avg_val_time = val_times["total_time"] / val_times["count"]
    avg_val_batch_time = avg_val_time / len(test_evaluator.dataloader)
    avg_val_sample_time = avg_val_time / len(test_evaluator.dataloader.dataset)
    logger.info(
        "Avg val. time: {} hours, {} minutes, {} seconds".format(
            *utils.readable_time(avg_val_time)
        )
    )
    logger.info("Avg batch test. time: {} seconds".format(avg_val_batch_time))
    logger.info("Avg sample test. time: {} seconds".format(avg_val_sample_time))

    print()
    print_str = "Testing Summary: "
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar("{}/test".format(k), v, epoch)
        print_str += "{}: {:8f} | ".format(k, v)
    logger.info(print_str)
    key_metric = "test_" + config.key_metric

    if config.key_metric in NEG_METRICS:
        condition = aggr_metrics[key_metric] < best_value
    else:
        condition = aggr_metrics[key_metric] > best_value
    if condition:
        best_value = aggr_metrics[key_metric]
        if not config.no_savemodel:
            utils.save_model(
                os.path.join(config.save_dir, "model_best.pth"),
                epoch,
                test_evaluator.model,
            )
        best_metrics = aggr_metrics.copy()

        pred_filepath = os.path.join(config.pred_dir, "best_predictions")
        for key in per_batch.keys():
            per_batch[key] = np.array(
                per_batch[key], dtype=object
            )  # resolve issue of inconsistent array sizes
        np.savez(pred_filepath, **per_batch)

    return aggr_metrics, best_metrics, best_value

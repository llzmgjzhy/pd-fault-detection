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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from utils.utils import matthews_correlation

logger = logging.getLogger("__main__")

NEG_METRICS = {"loss", "mse"}

val_times = {"total_time": 0, "count": 0}


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
        return Anomaly_Detection_Runner

    if task == "classification":
        return ClassificationRunner

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


class Anomaly_Detection_Runner(BaseRunner):
    def __init__(self, *args, **kwargs):

        super(Anomaly_Detection_Runner, self).__init__(*args, **kwargs)

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        for i, batch in enumerate(self.dataloader):
            self.optimizer.zero_grad()

            X, targets = batch
            X = X.float().to(device=self.device)
            outputs = self.model(X)

            loss = self.loss_module(X, outputs)
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

    def evaluate(self, epoch_num=None, keep_all=True):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {
            "target_masks": [],
            "targets": [],
            "outputs": [],
            "metrics": [],
        }
        for i, batch in enumerate(self.dataloader):

            X, targets = batch
            X = X.float().to(self.device)
            outputs = self.model(X)

            loss = self.loss_module(X, outputs)
            batch_loss = loss.sum()
            mean_loss = loss.mean()  # mean loss (over samples)
            # (batch_size,) loss for each sample in the batch

            per_batch["targets"].append(targets.half().cpu().numpy())
            per_batch["outputs"].append(outputs.half().cpu().numpy())
            per_batch["metrics"].append([loss.half().cpu().numpy()])

            metrics = {
                "loss": mean_loss,
            }
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Evaluating " + ending)

            total_samples += loss.numel()
            epoch_loss += batch_loss.half().cpu().item()

        epoch_loss = (
            epoch_loss / total_samples
        )  # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics

    def test(self, epoch_num=None, train_loader=None):

        self.model = self.model.eval()
        attens_energy = []

        # (1) statistic on the train set
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                X, _ = batch
                X = X.float().to(self.device)
                outputs = self.model(X)

                loss = self.loss_module(X, outputs)

                # cal score
                score = torch.mean(loss, dim=-1).detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) statistic on the test set
        attens_energy = []
        test_labels = []
        for i, batch in enumerate(self.dataloader):
            X, targets = batch
            X = X.float().to(self.device)
            outputs = self.model(X)

            loss = self.loss_module(X, outputs)

            # cal score
            score = torch.mean(loss, dim=-1).detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(targets)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.config.anomaly_ratio)
        logger.info("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        # fault detection
        pred = np.array(pred)
        gt = np.array(gt)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average="binary"
        )

        mcc = matthews_correlation(gt, pred)

        self.epoch_metrics["accuracy"] = accuracy
        self.epoch_metrics["precision"] = precision
        self.epoch_metrics["recall"] = recall
        self.epoch_metrics["f1"] = f_score
        self.epoch_metrics["mcc"] = mcc

        return self.epoch_metrics


class ClassificationRunner(BaseRunner):
    def __init__(self, *args, **kwargs):

        super(ClassificationRunner, self).__init__(*args, **kwargs)

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        for i, batch in enumerate(self.dataloader):
            self.optimizer.zero_grad()

            X, targets = batch
            X = X.float().to(device=self.device)
            targets = targets.float().to(device=self.device)
            outputs = self.model(X)

            loss = self.loss_module(outputs, targets)
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

    def evaluate(self, epoch_num=None, keep_all=False):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch

        per_batch = {
            "target_masks": [],
            "targets": [],
            "outputs": [],
            "metrics": [],
        }
        for i, batch in enumerate(self.dataloader):

            X, targets = batch
            X = X.float().to(device=self.device)
            targets = targets.float().to(device=self.device)
            outputs = self.model(X)

            loss = self.loss_module(outputs, targets)
            batch_loss = loss.sum()
            mean_loss = loss.mean()  # mean loss (over samples)
            # (batch_size,) loss for each sample in the batch

            per_batch["targets"].append(targets.half().cpu().numpy())
            per_batch["outputs"].append(outputs.half().cpu().numpy())
            per_batch["metrics"].append([loss.half().cpu().numpy()])

            metrics = {
                "loss": mean_loss,
            }
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else "Epoch {} ".format(epoch_num)
                self.print_callback(i, metrics, prefix="Evaluating " + ending)

            total_samples += loss.numel()
            epoch_loss += batch_loss.half().cpu().item()

        epoch_loss = (
            epoch_loss / total_samples
        )  # average loss per element for whole epoch
        self.epoch_metrics["epoch"] = epoch_num
        self.epoch_metrics["loss"] = epoch_loss

        pred = torch.from_numpy(np.concatenate(per_batch["outputs"], axis=0))
        test_labels = np.concatenate(per_batch["targets"], axis=0).reshape(-1)

        pred = np.argmax(pred, axis=1)
        # pred = (pred > 0.5).cpu().numpy().astype(int)
        gt = np.array(test_labels).astype(int)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average="binary"
        )
        mcc = matthews_correlation(gt, pred).item()

        self.epoch_metrics["accuracy"] = accuracy
        self.epoch_metrics["precision"] = precision
        self.epoch_metrics["recall"] = recall
        self.epoch_metrics["f1"] = f_score
        self.epoch_metrics["mcc"] = mcc

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics


def validate(
    val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch, fold_i=0
):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    logger.info("Evaluating on validation set ...")
    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics = val_evaluator.evaluate(epoch, keep_all=False)
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
        tensorboard_writer.add_scalar(f"{k}/val_fold_{fold_i}", v, epoch)
        print_str += "{}: {:8f} | ".format(k, v)
    logger.info(print_str)

    if config.key_metric in NEG_METRICS:
        condition = aggr_metrics[config.key_metric] < best_value
    else:
        condition = aggr_metrics[config.key_metric] > best_value
    if condition:
        best_value = aggr_metrics[config.key_metric]
        utils.save_model(
            os.path.join(config.save_dir, "model_best.pth"),
            epoch,
            val_evaluator.model,
        )
        best_metrics = aggr_metrics.copy()

    return aggr_metrics, best_metrics, best_value


def test(test_evaluator):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    logger.info("Testing on test set ...")
    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics = test_evaluator.evaluate(keep_all=False)
        del aggr_metrics["epoch"]
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
        print_str += "{}: {:8f} | ".format(k, v)
    logger.info(print_str)

    return aggr_metrics

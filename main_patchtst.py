"""
train the patchtst model on the vsb dataset.

if computation memory is enough,you can set the test is true,and the model will predict the test data.

example script:
python .\train_patchtst.py --splits-num=5 --epochs=30 --test --divide-train"""

import logging
import sys
import pandas as pd
import numpy as np
import torch
from torch.optim import lr_scheduler
import random
import os
import time
from tqdm import tqdm
import copy

from dataset.patchtst.dataset_factory import dataloader_provider
from models import model_factory
from models.optimizer import get_optimizer
from models.loss import get_loss_module
from options import Options
from running import setup, pipeline_factory, NEG_METRICS, test, validate
from utils import utils
from utils.tools import EarlyStopping
from torch.utils.tensorboard import SummaryWriter


def main(config):
    total_epoch_time = 0
    total_start_time = time.time()

    # add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config.output_dir, "output.log"))
    logger.addHandler(file_handler)

    logger.info(f"Running:\n{' '.join(sys.argv)}\n")

    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build data indices
    config.data_path = os.path.join(config.root_path, config.data_path)
    meta_data_train = pd.read_csv(
        os.path.join(config.root_path, config.meta_path, "metadata_train.csv")
    )
    positive_ids = set(
        meta_data_train.loc[meta_data_train["target"] == 1, "id_measurement"]
    )
    signals_ids = meta_data_train["id_measurement"].unique()
    labels = np.array([int(id in positive_ids) for id in signals_ids], dtype=np.int64)

    # train : val : test , 3:1:1
    splits = utils.stratified_train_val_test_split(
        signals_ids, labels, num_folds=config.split_num, seed=config.seed
    )

    for fold_i, (train_idx, val_idx, test_idx) in enumerate(splits):
        fold_start_time = time.time()

        # build data
        train_loader, val_loader, test_loader = dataloader_provider(
            config, train_idx, val_idx, test_idx, signals_ids, labels
        )

        # load model
        model_class = model_factory[config.model_name]
        model = model_class(config).to(device)

        early_stopping = EarlyStopping(patience=config.patience)

        # initialize the optimizer
        optim_class = get_optimizer(config.optimizer)
        optimizer = optim_class(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        # loss criterion
        loss_module = get_loss_module(config)

        # initialize the scheduler
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=1e-8
        )

        # initialize runner, responsible for training, validation and testing
        runner_class = pipeline_factory(config)

        trainer = runner_class(
            model,
            train_loader,
            device,
            loss_module,
            config,
            optimizer=optimizer,
            print_interval=config.print_interval,
            console=config.console,
        )
        val_evaluator = runner_class(
            model,
            val_loader,
            device,
            loss_module,
            config,
            print_interval=config.print_interval,
            console=config.console,
        )
        test_evaluator = runner_class(
            model,
            test_loader,
            device,
            loss_module,
            config,
            print_interval=config.print_interval,
            console=config.console,
        )

        tensorboard_writer = SummaryWriter(config.tensorboard_dir)

        best_value = 1e16 if config.key_metric in NEG_METRICS else -1e16
        metrics = []
        best_metrics = {}

        # Evaluate on validation before training
        aggr_metrics_val, best_metrics, best_value = validate(
            val_evaluator,
            tensorboard_writer,
            config,
            best_metrics,
            best_value,
            epoch=0,
            fold_i=fold_i,
        )
        metrics_names, metrics_values = zip(*aggr_metrics_val.items())
        metrics.append(list(metrics_values))

        logger.info("Starting training...")

        # training

        start_epoch = 0
        for epoch in tqdm(
            range(start_epoch + 1, config.epochs + 1),
            desc="Training Epoch",
            leave=False,
        ):
            epoch_start_time = time.time()
            aggr_metrics_train = trainer.train_epoch(epoch)
            epoch_runtime = time.time() - epoch_start_time
            print_str = f"Epoch {epoch} Training Summary: "
            for k, v in aggr_metrics_train.items():
                tensorboard_writer.add_scalar(f"{k}/train_fold_{fold_i}", v, epoch)
                print_str += f"{k}: {v:8f} | "

            logger.info(print_str)
            logger.info(
                "Epoch runtime: {} hours, {} minutes, {} seconds\n".format(
                    *utils.readable_time(epoch_runtime)
                )
            )
            total_epoch_time += epoch_runtime
            avg_epoch_time = total_epoch_time / (epoch - start_epoch)
            avg_batch_time = avg_epoch_time / len(train_loader)
            logger.info(
                "Avg epoch train. time: {} hours, {} minutes, {} seconds".format(
                    *utils.readable_time(avg_epoch_time)
                )
            )
            logger.info("Avg batch train. time: {} seconds".format(avg_batch_time))

            if (epoch == config.epochs) or (epoch % config.val_interval == 0):
                aggr_metrics_val, best_metrics, best_value = validate(
                    val_evaluator,
                    tensorboard_writer,
                    config,
                    best_metrics,
                    best_value,
                    epoch,
                    fold_i=fold_i,
                )
                metrics_names, metrics_values = zip(*aggr_metrics_val.items())
                metrics.append(list(metrics_values))
                early_stopping(best_value)
                if early_stopping.early_stop:
                    logger.info("Early stopping")
                    break

            scheduler.step()
            print("lr = {:.10f}".format(optimizer.param_groups[0]["lr"]))

        # test
        model.load_state_dict(
            torch.load(
                os.path.join(config.save_dir, "model_best.pth"), weights_only=True
            )["state_dict"]
        )
        aggr_metrics_test = test(test_evaluator)

        header = metrics_names
        metrics_filepath = os.path.join(
            config.output_dir, "metrics_" + config.experiment_name + ".xlsx"
        )

        book = utils.export_performance_metrics(
            metrics_filepath,
            metrics,
            header,
            sheet_name=f"metrics_fold_{fold_i}",
        )

        # Export record metrics to a file accumulating records from all experiments
        # add fold info
        del aggr_metrics_val["epoch"]  # remove epoch from final metrics
        comment = config.comment + f"_fold_{fold_i}"
        utils.register_test_record(
            config.records_file,
            config.initial_timestamp,
            config.experiment_name,
            best_metrics,
            aggr_metrics_val,
            aggr_metrics_test,
            comment=comment,
        )

        logger.info(
            f"Fold {fold_i} best {config.key_metric} was {best_value}. Other metrics: {best_metrics}"
        )

        fold_runtime = time.time() - fold_start_time
        logger.info(
            "Fold {} total runtime: {} hours, {} minutes, {} seconds\n".format(
                fold_i, *utils.readable_time(fold_runtime)
            )
        )

    total_runtime = time.time() - total_start_time
    logger.info(
        "Total runtime: {} hours, {} minutes, {} seconds\n".format(
            *utils.readable_time(total_runtime)
        )
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    logger.info("Loading packages ...")

    args = Options().parse()  # `argparse` object
    origin_comment = args.comment
    for ii in range(args.itr):
        args_itr = copy.deepcopy(args)  # prevent itr forloop to change output_dir
        args_itr.seed = args_itr.seed + ii  # change seed for each iteration
        config = setup(args_itr)  # save experiment files itr times
        config.comment = origin_comment + f" itr{ii}"
        main(config)

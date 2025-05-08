import torch
import os
from sklearn.model_selection import StratifiedKFold
import numpy as np
from openpyxl import Workbook, load_workbook

import logging
import sys
import builtins

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def matthews_correlation(y_true, y_pred):
    """Calculates the Matthews correlation coefficient measure for quality of binary classification problems."""
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    y_true = torch.tensor(y_true, dtype=torch.float32)

    y_pred_pos = torch.round(torch.clamp(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = torch.round(torch.clamp(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = torch.sum(y_pos * y_pred_pos)
    tn = torch.sum(y_neg * y_pred_neg)

    fp = torch.sum(y_neg * y_pred_pos)
    fn = torch.sum(y_pos * y_pred_neg)

    numerator = tp * tn - fp * fn
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + torch.finfo(torch.float32).eps)


def create_dirs(dirs):
    """
    Input:
        dirs: a list of directories to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def stratified_train_val_test_split(X, y, num_folds=5, seed=42):
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    # get the indices of the folds
    splits = np.zeros_like(y, dtype=int)
    for fold_id, (_, val_idx) in enumerate(skf.split(X, y)):
        splits[val_idx] = fold_id

    fold_splits = []
    for val_fold in range(num_folds):
        test_fold = (val_fold + 1) % num_folds
        train_folds = [f for f in range(num_folds) if f not in [val_fold, test_fold]]

        train_idx = np.where(np.isin(splits, train_folds))[0]
        val_idx = np.where(splits == val_fold)[0]
        test_idx = np.where(splits == test_fold)[0]

        fold_splits.append((train_idx, val_idx, test_idx))

    return fold_splits


class Printer(object):
    """Class for printing output by refreshing the same line in the console, e.g. for indicating progress of a process"""

    def __init__(self, console=True):

        if console:
            self.print = self.dyn_print
        else:
            self.print = builtins.print

    @staticmethod
    def dyn_print(data):
        """Print things to stdout on one line, refreshing it dynamically"""
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()


def readable_time(time_difference):
    """Convert a float measuring time difference in seconds into a tuple of (hours, minutes, seconds)"""

    hours = time_difference // 3600
    minutes = (time_difference // 60) % 60
    seconds = time_difference % 60

    return hours, minutes, seconds


def export_performance_metrics(
    filepath, metrics_table, header, book=None, sheet_name="metrics"
):
    """Exports performance metrics on the validation set for all epochs to an excel file"""

    if book is None:
        book = Workbook()  # new excel work book
        del book["Sheet"]  # remove default sheet

    book = write_table_to_sheet([header] + metrics_table, book, sheet_name=sheet_name)

    book.save(filepath)
    logger.info("Exported per epoch performance metrics in '{}'".format(filepath))

    return book


def write_table_to_sheet(table, work_book, sheet_name=None):
    """Writes a table implemented as a list of lists to an excel sheet in the given work book object"""

    sheet = work_book.create_sheet(sheet_name, index=0)

    for row_ind, row_list in enumerate(table):
        write_row(sheet, row_ind, row_list)

    return work_book


def write_row(sheet, row_ind, data_list):
    """Write a list to row_ind row of an excel sheet"""

    # row = sheet.row(row_ind)
    for col_ind, col_value in enumerate(data_list, start=1):
        sheet.cell(
            row=row_ind + 1, column=col_ind, value=col_value
        )  # row and col starts from 1 in openpyxl
    return

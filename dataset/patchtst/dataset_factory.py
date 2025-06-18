from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset.patchtst.dataset import TimeSeriesTrainDataset
import numpy as np


def _create_loader(
    dataset_class, indices, signals_ids, labels, config, is_train=True, sampler=None
):
    dataset = dataset_class(signals_ids[indices], labels[indices], config.data_path)
    is_shuffle = is_train
    if sampler is not None:
        labels = labels[indices]
        class_sample_counts = np.bincount(labels)
        class_weights = 1.0 / class_sample_counts  # shape: [num_classes]
        weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        # shuffle=is_shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        sampler=sampler,
    )


def dataloader_provider(config, train_idx, val_idx, test_idx, signals_ids, labels):
    train_loader = _create_loader(
        TimeSeriesTrainDataset, train_idx, signals_ids, labels, config, sampler=True
    )
    val_loader = _create_loader(
        TimeSeriesTrainDataset, val_idx, signals_ids, labels, config
    )
    test_loader = _create_loader(
        TimeSeriesTrainDataset, test_idx, signals_ids, labels, config, is_train=False
    )

    return train_loader, val_loader, test_loader

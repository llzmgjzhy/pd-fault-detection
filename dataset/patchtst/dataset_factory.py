from torch.utils.data import DataLoader
from dataset.patchtst.dataset import TimeSeriesTrainDataset


def _create_loader(dataset_class, indices, signals_ids, labels, config):
    dataset = dataset_class(signals_ids[indices], labels[indices], config.data_path)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )


def dataloader_provider(config, train_idx, val_idx, test_idx, signals_ids, labels):
    train_loader = _create_loader(
        TimeSeriesTrainDataset, train_idx, signals_ids, labels, config
    )
    val_loader = _create_loader(
        TimeSeriesTrainDataset, val_idx, signals_ids, labels, config
    )
    test_loader = _create_loader(
        TimeSeriesTrainDataset, test_idx, signals_ids, labels, config
    )

    return train_loader, val_loader, test_loader

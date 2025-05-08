from torch.utils.data import Dataset
import numpy as np
import os


class TimeSeriesTrainDataset(Dataset):
    def __init__(self, signal_ids, labels, data_path):
        self.signal_ids = signal_ids
        self.labels = labels
        self.img_path = data_path

    def __getitem__(self, index):
        # check img path exists
        if not os.path.exists(self.img_path):
            print(f"Warning: Path {self.img_path} does not exist.")
            return None
        signal = np.load(
            os.path.join(self.img_path, f"signals_{self.signal_ids[index]}.npy")
        ).astype(np.float32)
        signal = np.transpose(signal, (1, 0))

        label = self.labels[index]
        return (signal, label)

    def __len__(self):
        return len(self.signal_ids)


class TimeSeriesTestDataset(Dataset):
    def __init__(self, signal_ids, data_path):
        self.signal_ids = signal_ids
        self.img_path = data_path

    def __getitem__(self, index):
        # check img path exists
        if not os.path.exists(self.img_path):
            print(f"Warning: Path {self.img_path} does not exist.")
            return None
        signal = np.load(
            os.path.join(self.img_path, f"signals_{self.signal_ids[index]}.npy")
        ).astype(np.float32)
        signal = np.transpose(signal, (1, 0))

        return signal

    def __len__(self):
        return len(self.signal_ids)

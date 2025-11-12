"""
Data loader for the knots dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

class KnotsDataset(Dataset):
    def __init__(self, pt_path):
        self.data = torch.load(pt_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def knots_collate_fn(batch):
    """
    batch: list of (X_in, X_tar)
    - X_in: [N_in, 2]
    - X_tar: [N_tar, 2]  # no confidence yet!

    Returns:
      X_in_padded: (B, max_N_in, 2)
      mask_in: (B, max_N_in)
      X_tar_padded_with_conf: (B, max_N_tar, 3)
    """
    X_in_list, X_tar_list = zip(*batch)

    # Input padding + mask
    X_in_padded = pad_sequence(X_in_list, batch_first=True, padding_value=0.0)
    mask_in = torch.zeros(X_in_padded.shape[:2], dtype=torch.bool)
    for i, x in enumerate(X_in_list):
        mask_in[i, :x.size(0)] = True
        if x.shape[0] == 0:
            raise ValueError(f"Empty input at sample {i}")

    # Target padding + confidence
    X_tar_padded = pad_sequence(X_tar_list, batch_first=True, padding_value=0.0)

    B, max_N_tar, _ = X_tar_padded.shape

    # Add confidence dimension:
    X_tar_with_conf = torch.zeros(B, max_N_tar, 3)
    X_tar_with_conf[:, :, :2] = X_tar_padded

    for i, x in enumerate(X_tar_list):
        L = x.size(0)
        X_tar_with_conf[i, :L, 2] = 1.0  # confidence = 1 for real points        

    return X_in_padded, mask_in, X_tar_with_conf

def get_dataloaders(batch_size = 32, num_workers = 2, random_state = 42,
                     train_size = 0.2, val_size = 0.5, path = "datasets/knots_dataset.pt"):
    """
    Returns train, val, test DataLoaders for the knots dataset.
    Args:
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of workers for DataLoader.
        random_state (int): Random seed for reproducibility.
        train_size (float): Proportion of data to use for training.
        val_size (float): Proportion of remaining data to use for validation.
        path (str): Path to the dataset file.
    Returns:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
    """
    dataset = KnotsDataset(path)
    
    # Create indices for splitting
    indices = list(range(len(dataset)))
    train_indices, valtest_indices = train_test_split(indices, train_size=train_size, random_state=random_state)
    val_indices, test_indices = train_test_split(valtest_indices, train_size=val_size, random_state=random_state)

    # Use Subset to create datasets for splits
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Make DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=knots_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=knots_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=knots_collate_fn)

    return train_loader, val_loader, test_loader

def knots_collate_fn_reverse(batch):
    """
    Reverse-direction collate: predict inputs from targets.
    batch: list of (X_in, X_tar)
      - X_in: [N_in, 2]
      - X_tar: [N_tar, 2]

    Returns:
      X_tar_padded: (B, max_N_tar, 2)
      mask_tar: (B, max_N_tar)
      X_in_padded_with_conf: (B, max_N_in, 3)
    """
    X_in_list, X_tar_list = zip(*batch)

    # Target padding + mask (now inputs)
    X_tar_padded = pad_sequence(X_tar_list, batch_first=True, padding_value=0.0)
    mask_tar = torch.zeros(X_tar_padded.shape[:2], dtype=torch.bool)
    for i, x in enumerate(X_tar_list):
        mask_tar[i, :x.size(0)] = True
        if x.shape[0] == 0:
            raise ValueError(f"Empty target at sample {i}")

    # Input padding + confidence (now outputs)
    X_in_padded = pad_sequence(X_in_list, batch_first=True, padding_value=0.0)
    B, max_N_in, _ = X_in_padded.shape
    X_in_with_conf = torch.zeros(B, max_N_in, 3)
    X_in_with_conf[:, :, :2] = X_in_padded
    for i, x in enumerate(X_in_list):
        L = x.size(0)
        X_in_with_conf[i, :L, 2] = 1.0  # confidence = 1 for real points

    return X_tar_padded, mask_tar, X_in_with_conf


def get_dataloaders_reverse(batch_size=32, num_workers=2, random_state=42,
                             train_size=0.2, val_size=0.5,
                             path="datasets/knots_dataset.pt"):
    """
    Same as get_dataloaders, but reversed: use targets as inputs
    and predict inputs (with confidence channel).
    """
    dataset = KnotsDataset(path)

    indices = list(range(len(dataset)))
    train_indices, valtest_indices = train_test_split(indices, train_size=train_size, random_state=random_state)
    val_indices, test_indices = train_test_split(valtest_indices, train_size=val_size, random_state=random_state)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=knots_collate_fn_reverse)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=knots_collate_fn_reverse)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, collate_fn=knots_collate_fn_reverse)

    return train_loader, val_loader, test_loader

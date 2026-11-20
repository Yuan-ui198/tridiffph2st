import importlib
import inspect

import pytorch_lightning as pl
from torch.utils.data import DataLoader


def _dataloader_kwargs(loader_config):
    kwargs = {
        "batch_size": loader_config.batch_size,
        "num_workers": loader_config.num_workers,
        "pin_memory": loader_config.pin_memory,
        "shuffle": loader_config.shuffle,
    }
    if loader_config.num_workers > 0:
        kwargs["persistent_workers"] = getattr(loader_config, "persistent_workers", True)
        kwargs["prefetch_factor"] = getattr(loader_config, "prefetch_factor", 2)
    return kwargs


class DataInterface(pl.LightningDataModule):
    def __init__(self, dataset_name=None, config=None, **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        self.config = config
        self.data_config = kwargs["data_config"]
        self.load_data_module()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = self.instancialize(phase="train")
            self.val_dataset = self.instancialize(phase="test")
        if stage == "test":
            self.test_dataset = self.instancialize(phase="test")
        if stage == "predict":
            self.test_dataset = self.instancialize(phase="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            **_dataloader_kwargs(self.data_config.train_dataloader),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            **_dataloader_kwargs(self.data_config.test_dataloader),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            **_dataloader_kwargs(self.data_config.test_dataloader),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            **_dataloader_kwargs(self.data_config.test_dataloader),
        )

    def load_data_module(self):
        for module_name in ("base_dataset", "new_dataset"):
            module = importlib.import_module(module_name)
            if hasattr(module, self.dataset_name):
                self.data_module = getattr(module, self.dataset_name)
                return
        raise ValueError(f"Invalid dataset class name: {self.dataset_name}")

    def instancialize(self, **other_args):
        class_args = inspect.getfullargspec(self.data_module.__init__).args[1:]
        config_keys = self.data_config.keys()
        args = {}
        for arg in class_args:
            if arg in config_keys:
                args[arg] = self.data_config[arg]
        args.update(other_args)

        # H3 builds the spatial microenvironment from global UNI features and
        # coordinates; never load legacy neighbor UNI tensors for this branch.
        spatial_branch = str(
            getattr(getattr(self.config, "MODEL", None), "spatial_branch_type", "")
        ).lower()
        if spatial_branch in {"h3_many_body", "h3", "hierarchical_many_body", "h3_microenvironment"}:
            args["use_neighbor_emb"] = False
        return self.data_module(**args)

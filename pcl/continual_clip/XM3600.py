import os
from typing import Tuple, Union, Optional

from torchvision import transforms
import numpy as np
from continuum.tasks import TaskType

from continuum.datasets import ImageFolderDataset, _ContinuumDataset
from continuum.download import download, unzip


class XM3600(_ContinuumDataset):
    """Subset of ImageNet1000 made of only 100 classes.

    You must download the ImageNet1000 dataset then provide the images subset.
    If in doubt, use the option at initialization `download=True` and it will
    auto-download for you the subset ids used in:
        * Small Task Incremental Learning
          Douillard et al. 2020
    """

    def __init__(
            self, *args, data_subset: Union[Tuple[np.array, np.array], str, None] = None, **kwargs
    ):
        self.data_subset = data_subset
        super().__init__(*args, **kwargs)

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        return [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]


    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        data = self._parse_subset(self.data_subset, train=self.train)  # type: ignore
        return (*data, None)

    def _parse_subset(
            self,
            subset: Union[Tuple[np.array, np.array], str, None],
            train: bool = True
    ) -> Tuple[np.array, np.array]:
        if isinstance(subset, str):
            x, y, caption = [], [], []

            with open(subset, "r") as f:
                for line in f:
                    split_line = line.split(" ", 1)
                    path = split_line[0].strip().split('#')[0]+'.jpg'
                    lang_id = split_line[0].strip().split('#')[2]
                    cap = split_line[1].strip()
                    x.append(os.path.join(self.data_path, path))
                    y.append(int(lang_id))
                    caption.append(cap)
            x = np.array(x)
            y = np.array(y)
            caption = np.array(caption)
            return x, y, caption
        return subset  # type: ignore

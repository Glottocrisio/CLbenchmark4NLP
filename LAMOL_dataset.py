#    Create the following two directories in wherever you want. (you can name the directories arbitrarily):
#    data directory: Where the dataset will be load by the model.
#    model directory: The place for the model to dump its outputs.
#Download the dataset: Download here and decompress it. After decompression, move all the files in the decompressed directory into data directory.
#Make a copy of env.example and save it as env. In env, set the value of DATA_DIR as data directory and set the value of MODEL_ROOT_DIR as model directory.

import os
import shutil
from pathlib import Path
from typing import Union

from torchvision.datasets.folder import default_loader
from zipfile import ZipFile

from torchvision.transforms import ToTensor

from avalanche.benchmarks.datasets import (
    SimpleDownloadableDataset,
    default_dataset_location,
)

# from avalanche.benchmarks.datasets.CLbenchmark4NLP.
from LAMOL_data import LAMOL


class LAMOL_data(SimpleDownloadableDataset):
    """LAMOL_data Pytorch Dataset"""

    def __init__(
        self,
        root: Union[str, Path] = None,
        *,
        train=True,
        transform=None,
        target_transform=None,
        loader=default_loader,
        download=True,
    ):
        """
        Creates an instance of the LAMOL_data dataset.
        :param root: The directory where the dataset can be found or downloaded.
            Defaults to None, which means that the default location for
            'LAMOL_data' will be used.
        :param train: If True, the training set will be returned. If False,
            the test set will be returned.
        :param transform: The transformations to apply to the X values.
        :param target_transform: The transformations to apply to the Y values.
        :param loader: The image loader to use.
        :param download: If True, the dataset will be downloaded if needed.
        """

        if root is None:
            root = default_dataset_location("LAMOL_data")

        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        super(LAMOL_data, self).__init__(root, download=download, verbose=True)
        self._load_dataset()

    def _download_dataset(self) -> None:
        data2download = LAMOL_data.__all__

        for name in data2download:
            if self.verbose:
                print("Downloading " + name[1] + "...")
            file = self._download_file(name[1], name[0])
            if name[1].endswith(".zip"):
                if self.verbose:
                    print(f"Extracting {name[0]}...")
                self._extract_archive(file)
                if self.verbose:
                    print("Extraction completed!")

    def _load_metadata(self) -> bool:
        if not self._check_integrity():
            return False

        # any scenario and factor is good here since we want just to load the
        # train images and targets with no particular order
        scen = "domain"
        factor = [_ for _ in range(4)]
        ntask = 9

        print("Loading paths...")
        with open(str(self.root / "Paths.pkl"), "rb") as f:
            self.train_test_paths = pkl.load(f)

        print("Loading labels...")
        with open(str(self.root / "Labels.pkl"), "rb") as f:
            self.all_targets = pkl.load(f)
            self.train_test_targets = []
            for fact in factor:
                for i in range(ntask + 1):
                    self.train_test_targets += self.all_targets[scen][fact][i]

        print("Loading LUP...")
        with open(str(self.root / "LUP.pkl"), "rb") as f:
            self.LUP = pkl.load(f)

        self.idx_list = []
        if self.train:
            for fact in factor:
                for i in range(ntask):
                    self.idx_list += self.LUP[scen][fact][i]
        else:
            for fact in factor:
                self.idx_list += self.LUP[scen][fact][-1]

        self.paths = []
        self.targets = []

        for idx in self.idx_list:
            self.paths.append(self.train_test_paths[idx])
            self.targets.append(self.train_test_targets[idx])

        return True

    def _download_error_message(self) -> str:
        base_url = LAMOL_data.base_gdrive_url
        all_urls = [
            base_url + name_url[1] for name_url in LAMOL_data._all_
        ]

        base_msg = (
            "[LAMOL_data] Direct download may no longer be supported!\n"
            "You should download data manually using the following links:\n"
        )

        for url in all_urls:
            base_msg += url
            base_msg += "\n"

        base_msg += "and place these files in " + str(self.root)

        return base_msg

    def _check_integrity(self):
        """Checks if the data is already available and intact"""

        for name in LAMOL_data._all_:
            filepath = self.root / name
            if not filepath.is_file():
                if self.verbose:
                    print(
                        "[LAMOL_data] Error checking integrity of:",
                        str(filepath),
                    )
                return False
        return True
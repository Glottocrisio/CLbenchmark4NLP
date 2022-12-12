#import gensim
#import pandas as pd
#import numpy as np
#import scipy
#import math
#import os
#import tensorflow as tf
#import matplotlib.pyplot as plt
#import seaborn as sns
#import re

#from gensim.models.keyedvectors import KeyedVectors

#import nltk
#from nltk.tokenize import RegexpTokenizer
#nltk.download('punkt')
#from sklearn.decomposition import TruncatedSVD, randomized_svd
#from sklearn.metrics.pairwise import cosine_similarity

import os
import shutil
from pathlib import Path
from typing import Union

from torchvision.datasets.folder import default_loader
from zipfile import ZipFile

from torchvision.transforms import ToTensor
from avalanche.benchmarks.utils import DatasetFolder

from avalanche.benchmarks.datasets import (
    DownloadableDataset,
    default_dataset_location,
)

# from avalanche.benchmarks.datasets.CLbenchmark4NLP.
from STS_data import (
    STS, STS_Companion 
)


#FROM AVALANCHE, SNIPPET FEATURING IN ALL DATASET SCRIPTS

class STS_data(DownloadableDataset):
    """STS_data Pytorch Dataset"""

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
        Creates an instance of the STS_data dataset.
        :param root: The directory where the dataset can be found or downloaded.
            Defaults to None, which means that the default location for
            'STS_data' will be used.
        :param train: If True, the training set will be returned. If False,
            the test set will be returned.
        :param transform: The transformations to apply to the X values.
        :param target_transform: The transformations to apply to the Y values.
        :param loader: The image loader to use.
        :param download: If True, the dataset will be downloaded if needed.
        """

        if root is None:
            root = default_dataset_location("STS_data")

        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        super(STS_data, self).__init__(root, download=download, verbose=True)
        self._load_dataset()

    def _download_dataset(self) -> None:
        data2download = STS_data.__all__

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
        base_url = STS_data.base_gdrive_url
        all_urls = [
            base_url + name_url[1] for name_url in STS_data._all_
        ]

        base_msg = (
            "[STS_data] Direct download may no longer be supported!\n"
            "You should download data manually using the following links:\n"
        )

        for url in all_urls:
            base_msg += url
            base_msg += "\n"

        base_msg += "and place these files in " + str(self.root)

        return base_msg

    def _check_integrity(self):
        """Checks if the data is already available and intact"""

        for name in STS_data._all_:
            filepath = self.root / name
            if not filepath.is_file():
                if self.verbose:
                    print(
                        "[STS_data] Error checking integrity of:",
                        str(filepath),
                    )
                return False
        return True



#THE FOLLOWING CODE IS COPIED FROM THE GITHUB REPOSITORY OF 
# "Continual Learning for Sentence Representations Using Conceptors"

def load_sts_dataset(filename):
    # For a STS dataset, loads the relevant information: the sentences and their human rated similarity score.
    sent_pairs = []
    with tf.gfile.GFile(filename, "r") as f:
        for line in f:
            ts = line.strip().split("\t")
            #sent_pairs.append((ts[5], ts[6], float(ts[4])))
            if len(ts) == 7 or len(ts) == 9:
                sent_pairs.append((re.sub("[^0-9]", "", ts[2]) + '-' + ts[1] , ts[5], ts[6], float(ts[4])))
            elif len(ts) == 6 or len(ts) == 8:
                sent_pairs.append((re.sub("[^0-9]", "", ts[1]) + '-' + ts[0] , ts[4], ts[5], float(ts[3])))
            else:
                print('data format is wrong!!!')
    return pd.DataFrame(sent_pairs, columns=["year-task", "sent_1", "sent_2", "sim"])


def load_all_sts_dataset():
    # Loads all of the STS datasets 
    stsbenchmarkDir = resourceFile + 'stsbenchmark/'
    stscompanionDir = resourceFile + 'stscompanion/'
    sts_train = load_sts_dataset(os.path.join(stsbenchmarkDir, "sts-train.csv"))    
    sts_dev = load_sts_dataset(os.path.join(stsbenchmarkDir, "sts-dev.csv"))
    sts_test = load_sts_dataset(os.path.join(stsbenchmarkDir, "sts-test.csv"))
    sts_other = load_sts_dataset(os.path.join(stscompanionDir, "sts-other.csv"))
    sts_mt = load_sts_dataset(os.path.join(stscompanionDir, "sts-mt.csv"))
    
    sts_all = pd.concat([sts_train, sts_dev, sts_test, sts_other, sts_mt ])
    
    return sts_all

sts_all = load_all_sts_dataset()

# show some sample sts data    
sts_all[:5] 

# For the Avalanche data loader adaptation:
################################################################################
# Copyright (c) 2022 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-12-2022                                                             #
# Author: Cosimo Palma                                                         #
#                                                                              #
# E-mail: cosimo.palma@phd.unipi.it                                            #
# Website: www.continualai.org                                                 #
################################################################################


import os
os.chdir('C:/Users/Palma/.avalanche/data')
import requests
from io import BytesIO
from pathlib import Path
from typing import Union
#import pickle as pkl
from torchvision.datasets.folder import default_loader
import xtarfile as  tarfile

from torchvision.transforms import ToTensor
from avalanche.benchmarks.utils import DatasetFolder

from avalanche.benchmarks.datasets import (
    DownloadableDataset,
    default_dataset_location,
)

# from avalanche.benchmarks.datasets.CLbenchmark4NLP.
import STS_data 


class STS_dataset(DownloadableDataset):
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

        super().__init__(root,  STS_data.__all__[0],
            STS_data.__all__[1],
            #download=True,
            #verbose=True,
        )
        #self._load_dataset()

    def _download_dataset(self) -> None:
        data2download = STS_data.__all__
   
        for name in data2download:
            url = getattr(STS_data, name)[1]
            path = getattr(STS_data, name)[0]
            lfilename = self.root / path
            response = requests.get(url, stream=True, auth=('user', 'pass'))
          
            isExist = os.path.exists(lfilename)
            if not isExist:
                if response.status_code == 200:
                    with open(lfilename, 'wb') as f:
                        f.write(response.raw.read())

                with tarfile.open(lfilename, mode="r:gz") as  tarobj: 
                        print("Extracting" + str(name) + "  dataset...")
                        tarobj.extractall("STS_data")
                        print(str(name) + " dataset extracted!")
        

    def _load_metadata(self) -> bool:
        if not self._check_integrity():
            return False

        # any scenario and factor is good here since we want just to load the
        # train images and targets with no particular order
    #    scen = "domain"
    #    factor = [_ for _ in range(4)]
    #    ntask = 9

    #    print("Loading paths...")
    #    with open(str(self.root / "Paths.pkl"), "rb") as f:
    #        self.train_test_paths = pkl.load(f)

    #    print("Loading labels...")
    #    with open(str(self.root / "Labels.pkl"), "rb") as f:
    #        self.all_targets = pkl.load(f)
    #        self.train_test_targets = []
    #        for fact in factor:
    #            for i in range(ntask + 1):
    #                self.train_test_targets += self.all_targets[scen][fact][i]

    #    print("Loading LUP...")
    #    with open(str(self.root / "LUP.pkl"), "rb") as f:
    #        self.LUP = pkl.load(f)

    #    self.idx_list = []
    #    if self.train:
    #        for fact in factor:
    #            for i in range(ntask):
    #                self.idx_list += self.LUP[scen][fact][i]
    #    else:
    #        for fact in factor:
    #            self.idx_list += self.LUP[scen][fact][-1]

    #    self.paths = []
    #    self.targets = []

    #    for idx in self.idx_list:
    #        self.paths.append(self.train_test_paths[idx])
    #        self.targets.append(self.train_test_targets[idx])

    #    return True

    def _download_error_message(self) -> str:
        
        base_msg = (
            "[STS_data] Direct download may no longer be supported!\n"
            "You should download data manually using the following links:\n"
            "http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz for STS Benchmark\n"
            "http://ixa2.si.ehu.es/stswiki/images/e/ee/Stscompanion.tar.gz STS Companion \n"
        )

        return base_msg


    def _check_integrity(self):
        """Checks if the data is already available and intact"""
        for name in STS_data.__all__:
            if name=="STS":
                name = "stsbenchmark"
                filepath = self.root / name
            else:
                filepath = self.root / name.tolower()
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

#def load_sts_dataset(filename):
#    # For a STS dataset, loads the relevant information: the sentences and their human rated similarity score.
#    sent_pairs = []
#    with tf.gfile.GFile(filename, "r") as f:
#        for line in f:
#            ts = line.strip().split("\t")
#            #sent_pairs.append((ts[5], ts[6], float(ts[4])))
#            if len(ts) == 7 or len(ts) == 9:
#                sent_pairs.append((re.sub("[^0-9]", "", ts[2]) + '-' + ts[1] , ts[5], ts[6], float(ts[4])))
#            elif len(ts) == 6 or len(ts) == 8:
#                sent_pairs.append((re.sub("[^0-9]", "", ts[1]) + '-' + ts[0] , ts[4], ts[5], float(ts[3])))
#            else:
#                print('data format is wrong!!!')
#    return pd.DataFrame(sent_pairs, columns=["year-task", "sent_1", "sent_2", "sim"])


#def load_all_sts_dataset():
#    # Loads all of the STS datasets 
#    stsbenchmarkDir = resourceFile + 'stsbenchmark/'
#    stscompanionDir = resourceFile + 'stscompanion/'
#    sts_train = load_sts_dataset(os.path.join(stsbenchmarkDir, "sts-train.csv"))    
#    sts_dev = load_sts_dataset(os.path.join(stsbenchmarkDir, "sts-dev.csv"))
#    sts_test = load_sts_dataset(os.path.join(stsbenchmarkDir, "sts-test.csv"))
#    sts_other = load_sts_dataset(os.path.join(stscompanionDir, "sts-other.csv"))
#    sts_mt = load_sts_dataset(os.path.join(stscompanionDir, "sts-mt.csv"))
    
#    sts_all = pd.concat([sts_train, sts_dev, sts_test, sts_other, sts_mt ])
    
#    return sts_all

#sts_all = load_all_sts_dataset()

## show some sample sts data    
#sts_all[:5] 


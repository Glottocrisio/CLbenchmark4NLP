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



#    Create the following two directories in wherever you want. (you can name the directories arbitrarily):
#    data directory: Where the dataset will be load by the model.
#    model directory: The place for the model to dump its outputs.
#Download the dataset: Download here and decompress it. After decompression, move all the files in the decompressed directory into data directory.
#Make a copy of env.example and save it as env. In env, set the value of DATA_DIR as data directory and set the value of MODEL_ROOT_DIR as model directory.

import os
os.chdir('C:/Users/Palma/.avalanche/data')
import requests
from io import BytesIO
from pathlib import Path
from typing import Union
#import pickle as pkl
from torchvision.datasets.folder import default_loader
import xtarfile as  tarfile
#import json

#from torchvision.transforms import ToTensor

from avalanche.benchmarks.datasets import (
   SimpleDownloadableDataset, DownloadableDataset,
    default_dataset_location,
)

# from avalanche.benchmarks.datasets.CLbenchmark4NLP.
import LAMOL_data  


class LAMOL_dataset(SimpleDownloadableDataset):
    """LAMOL_dataset Pytorch Dataset"""

    def __init__(
        self,
        *,
        root: Union[str, Path] = None,
        
        train=True,
        transform=None,
        target_transform=None,
        loader=default_loader,
        download = True,
        verbose = False,
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

        super().__init__(root,  LAMOL_data.LAMOL[0],
            LAMOL_data.LAMOL[1],
            #download=True,
            #verbose=True,
        )

        #self._load_dataset()

    def _download_dataset(self) -> None:
        url = LAMOL_data.LAMOL[1]
        path = LAMOL_data.LAMOL[0]
        lfilename = self.root / path
        response = requests.get(url, stream=True, auth=('user', 'pass'))
      
        isExist = os.path.exists(lfilename)
        if not isExist:
            if response.status_code == 200:
                with open(lfilename, 'wb') as f:
                    f.write(response.raw.read())

            with tarfile.open(lfilename, mode="r:gz") as  tarobj: 
                    print("Extracting LAMOL  dataset...")
                    tarobj.extractall("LAMOL_data")
                    print("LAMOL dataset extracted!")
                  
          
        #if self.verbose:
        #    print("LAMOL Extracting dataset...")

       
        #with tarfile.open(lfilename, mode="r:gz") as  tarobj: 
        #    for member in tarobj.getnames():
        #        filename = os.path.basename(member)
        #        # skip directories
        #        if not filename:
        #            continue

        #        # copy file (taken from zipfile's extract)
        #        source = tarobj.open(member)
        #        if "json" in filename:
        #            target = open(str(self.root / filename), "wb")
        #        else:
        #            dest_folder = os.path.join(
        #                *(member.split(os.path.sep)[1:-1])
        #            )
        #            dest_folder = self.root / dest_folder
        #            dest_folder.mkdir(exist_ok=True, parents=True)

        #            target = open(str(dest_folder / filename), "wb")
        #        with source, target:
        #            shutil.copyfileobj(source, target)

        # lfilename.unlink()


    def _load_metadata(self) -> bool:
        if not self._check_integrity():
           return False

    #    # any scenario and factor is good here since we want just to load the
    #    # train images and targets with no particular order
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
       
        url = LAMOL_data.LAMOL[1]
        base_msg = (
            "[LAMOL_data] Direct download may no longer be supported!\n"
            "You should download data manually using the following links:\n"
        )


        base_msg += str(url) + "and place these files in " + str(self.root)

        return base_msg

    def _check_integrity(self):
        """Checks if the data is already available and intact"""
        path = LAMOL_data.LAMOL[0]
       # for name in LAMOL_data.__all__:
        filepath = self.root / path
        if not filepath.is_file():
            if self.verbose:
                print(
                    "[LAMOL_data] Error checking integrity of:",
                    str(filepath),
                )
                print("false")
            return False
        print("true")
        return True

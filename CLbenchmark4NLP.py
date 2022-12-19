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


import LAMOL_dataset as Lds
import LAMOL_data as Ld

import STS_data as sts
import STS_dataset as stsd


s = stsd.STS_dataset()
s._download_dataset()
s._download_error_message()
s._check_integrity()

q = Lds.LAMOL_dataset()
q._download_dataset()
q._download_error_message()
q._check_integrity()
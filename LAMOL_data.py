
################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 09-12-2022                                                               #
# Author: Cosimo Palma                                                         #
# E-mail: cosimo.palma@phd.unipi.it                                            #
################################################################################

#LAnguage-MOdeling-for-Lifelong-Language-Learning

#Most research on lifelong learning (LLL) applies to images or games, but not language. We present LAMOL, a simple yet effective method for LLL based on language modeling. LAMOL replays pseudo-samples of previous tasks while requiring no extra memory or model capacity. Specifically, LAMOL is a language model that simultaneously learns to solve the task and generate training samples. When the model is trained for a new task, it generates pseudo-samples of previous tasks for training alongside data for the new task. The results show that LAMOL prevents catastrophic forgetting without any sign of intransigence and can perform up to five very different language tasks sequentially with only one model. Overall, LAMOL outperforms previous methods by a considerable margin and is only 2--3% worse than multitasking, which is usually considered the LLL upper bound.

#Dataset

#Task 	                        Dataset 
#Question Answering 	        SQuAD version 1.1
#Machine Translation 	        IWSLT
#Summarization 	                CNN/DM
#Natural Language Inference 	CNN/DM
#Sentiment Analysis 	        SST
#Semantic Role Labeling 	    QA‑SRL
#Zero-Shot Relation Extraction 	QA‑ZRE
#Goal-Oriented Dialogue 	    WOZ
#Semantic Parsing 	            WikiSQL
#Commonsense Reasoning 	        MWSC
#Text Classification 	        AGNews, Yelp, Amazon, DBPedia, Yahoo

#In order to unify the format of all the dataset, we first ran the code in https://github.com/salesforce/decaNLP to get the first 10 tranformed dataset, and then converted them into Squad-like format. For the last 5 dataset, we converted them directly. All converted dataset are available at 
#https://drive.google.com/u/0/uc?id=1rWcgnVcNpwxmBI3c5ovNx-E8XKOEL77S&export=download.
  
LAMOL = (#"LAMOL.tar.gz",
    "https://drive.google.com/u/0/uc?id=1rWcgnVcNpwxmBI3c5ovNx-E8XKOEL77S&export=download",  
    "97230c504ffd47808fbcb8bde4aad912",
)

__all__ = ["LAMOL"]






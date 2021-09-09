# Privacy-Preserving Support Vector Machine for Outsourcing

### Jiaqi Hu, Muyuan Li, Huimin Wang, Leyang Wen

Final Project for EECS 545; University of Michigan, Ann Arbor, 2020 Fall

## Introduction

This project is based on [this paper](https://ieeexplore.ieee.org/abstract/document/6682897), and is a reimplementation of the first client-server privacy-persevering (pp) protocol for SVM. With the recent emergence of cloud computing and data-driven machine learning methods, more and more people now aims to take advantage of these high-performance and flexible services. This leads to increasing demands of sharing private data to cloud computing providers or third-party-servers. Under this context, this project aims to fully exploit the homomorphic properties of certain encryption protocols to allow cloud servers to compute SVM without decrypting the sensitive client data.

Nowadays machine learning based classifiers have gained great popularity, as they can learn from a huge set of labeled data with superior speed and accuracy compared to many traditional methods. 

However, this also means that the performance of such classifiers is highly correlated to the size and quality of the training data. Many smaller organizations lack the human resource to collect huge amounts of valid data or the computational recourse to train and store large classifiers, making outsourcing classification tasks a sound choice. In particular, the client will provide a private set of testing data that they want to classify. The provider will take the dataset and process through their already trained classifier and provide the results. This framework would mitigate the technical requirements for the client to take advantage of the state-of-the-art machine learning algorithms. 

That being said, this framework unavoidably induces problems regarding privacy and data governance. For example, if a doctor wants to use sensitive patient information to diagnose disease via cloud computing, the doctor might not be legally capable of uploading this information to an untrusted-third-party-server. On the other hand, the cloud computing provider might not wise to disclose the detailed parameters of the classifier. To solve this problem, our project aims to establish a secured cloud computing channel for both the client and the provider. In other words, the objective is to build a cloud-based machine learning classifier while simultaneously making sure that a) the provider can not interpret the test data; b) the client can not interpret the classifier parameters.

For more detailed information, please check `EECS_545_Final_Report.pdf` under the main working directory.

## Usage
In the target directory, run
```
$ pip install -r requirements.txt
```
To see all the program flags, run with option -h
Running example :
```
$ ./main.py -p 2 -s 4 -t 0.3 -v
```

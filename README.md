# Ground-to-Aerial-Img-Matching: comparison between the SAN model and a GAN-based bodel 

## Abstract
We chose to address the problem of matching a query ground-image to the corresponding satellite image, using
image processing and ML methods. To solve this problem we will implement SAN, as proposed by Pro, Amerini et al.
We will use a pre-trained model to compute the segmentation of the satellite image to compare to the query ground
view. Then, the satellite view and its segmentation are processed and passed to 3 CNNs, along with the ground view.
The outputs of the networks are compared to estimate a similarity score.
The SAN approach is compared to the GAN-based approach proposed by Regmi et al. In that project, a synthetic satellite
image is generated from the ground-view using a GAN, in order to minimize the gap between the ground and aerial domains.
Furthermore, the features extracted from those images are combined to get a more roboust representation of the
ground-view, and used together to solve the ground-to-aerial matching problem. 
For training and testing we will use the CVUSA dataset, one of the main choices for this kind of tasks.

[[Paper SAN]](https://arxiv.org/pdf/2404.11302)
[[Paper GAN]](https://arxiv.org/pdf/1904.11045)

## Dataset
The CVUSA dataset is designed for a matching task between street and aerial views from various regions across the United States. This task aims to determine the localization of street-view images without relying on GPS coordinates. Ground images are sourced from Google Street View panoramas, while the corresponding aerial images, captured at zoom level 19, are obtained from Microsoft Bing Maps. The dataset includes 35,532 image pairs for training and 8,884 image pairs for testing, with recall serving as the primary evaluation metric.

In our project, due to limited resources, we used a subset of CVUSA. In particular, the one provided by the [repo
hosting the SAN project](https://pro1944191.github.io/SemanticAlignNet/). 

## How to run the code

### Pre-operations
First of all, we have to identify the (global) path that points to the CVUSA dataset to use for training and evaluation,
and put it in the `dataset_path` variable. Then, put the (global) paths to the CSVs containing the training and
evaluation triplets in `trainCSV` and `valCSV` respectively. The former will be partitioned in training and validation
sets, while the former will be used for testing.

### SAN training/evaluation
Given `dataset_path`, `trainCSV` and `valCSV` as defined above:
#### Training
To train the network you have to call

```train(device, dataset_path, trainCSV)```

(defined in `san_model/main.py`) and the training process will start.

#### Evaluation
To evaluate the network you have to call

```evaluate(device, dataset_path, trainCSV)```

(defined in `san_model/main.py`) and the model will be evaluated according to the top1Recall metric.

### GAN-based training/evaluation
Given `dataset_path`, `trainCSV` and `valCSV` as defined above:
#### Training
To train the whole network (both the JointFeatureLearning and the FeatureFusion networks) you have to call

```train_feature_extractor(device, dataset_path, trainCSV)```

(defined in `gen_model/main.py`) and the training process will start.

#### Evaluation
To evaluate the whole network (both the JointFeatureLearning and the FeatureFusion networks) you have to call

```evaluate_feature_extractor(device, dataset_path, trainCSV)```

(defined in `gen_model/main.py`) and the model will be evaluated according to the top1Recall metric.


## Authors of this project
[Salvatore Michele Rago](https://github.com/salvatore373), [Mattia Maffongelli](https://github.com/mattiamaffo)
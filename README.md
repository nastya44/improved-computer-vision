# Gradient-free deep learning

<img align="right" width="450" height="300" src="./docs/thumbnail.png">

![Python](./docs/ico/python.svg)
![Pytorch](./docs/ico/pytorch.svg)

The project from the discipline "Applied Modeling" combines new trends in optimization theory with relevant deep learning problems.

## Objective

1. Examine the most widely used gradient optimization algorithms;
2. Develop and apply a gradient-free optimization algorithm for Backpropagation;
3. Evaluate algorithms performance on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) benchmark problem;
4. Train and assess the deep learning model for object segmentation, ["YOLO"](https://openaccess.thecvf.com/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf) (You Only Look Once).

## Relevance

 - The problem of image segmentation is widely recognized in the field of automation.
 - One of the most significant applications of this technology in Ukraine at present is in the area of military technology: specifically, automatic targeting for unmanned aerial vehicles (UAVs) or artillery systems.

## Gradient vs finite-difference

 - Gradient methods offer advantages compared to zero-order methods:
    - **Scalability** - gradient methods, especially when implemented with efficient matrix operations on specialized hardware such as GPUs, scale well depending on the size of the dataset and model parameters;
    - **Information Utilization** - gradients provide valuable information about the shape and direction of the objective function. They indicate the direction of the steepest ascent, and their magnitude gives you an idea of how steep the ascent is. This information is used to move more directly to the minimum. Methods without derivatives do not use this information.
 - Finite-difference methods maintain these advantages;
 - In contrast to finite-difference methods, the gradient encounters difficulties in navigating nonsmooth functions (such as the Rectified Linear Unit (ReLU)).

## Train data

 - The object segmentation problem for the project is "Car Object Detection";
 - The [dataset](https://www.kaggle.com/datasets/sshikamaru/car-object-detection) is available on the Kaggle platform under an open-source licence.


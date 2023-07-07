# WebAPI for image recognition with a custom CNN (PyTorch)

## Network

The architecture consists of seven basic residual blocks with additional dropout and pooling layers.
The model was trained on the CIFAR-100 dataset (62.4% top-1 and 87.9% top-5 accuracy on the testing subset).

## Interface

Upload an image to obtain a prediction with a detailed visualization.

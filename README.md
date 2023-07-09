# WebAPI for image recognition with a custom Residual CNN using PyTorch and Flask

## Network

The architecture consists of seven basic residual blocks with additional dropout and pooling layers.
The model was trained on the CIFAR-100 ([link](https://www.cs.toronto.edu/~kriz/cifar.html)) dataset (62.4% top-1 and 87.9% top-5 accuracy on the testing subset).

## Interface

Upload an image to obtain a classification with a detailed visualization of the results:
![program interface with an example of an inference output](blob/main/interface.jpg?raw=true)

To run the application, install dependencies with `pip install -r requirements.txt` and run Flask server:

`flask --app ./server.py run`

## Training and evaluation

Training script, detailed evaluation and examples of inference are in `notebooks` directory.

## References

Dataset: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009

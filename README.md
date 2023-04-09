Going through:
<https://medium.com/the-future-machine-learning-and-ai/how-to-build-a-deep-learning-model-in-less-than-1-hour-8cb310a88013>

# Installing pytorch and pytorchvision...

Getting error: ```AttributeError: partially initialized module 'torch' has no attribute 'ops' (most likely due to a circular import)```
- entering `torchvision==0.15.1` in requirements.in auto-installs `torchvision==0.15.1` in requirements.txt (via pip-compile)
- tried reverting to python 3.9 but still get same error.
- reading <https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c>
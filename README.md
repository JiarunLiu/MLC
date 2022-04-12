# MLC

PyTorch implementation of *Agreement or Disagreement in Noise-tolerant Mutual Learning?*

## Requirements:

+ python3.6
+ numpy
+ torch-1.4.0
+ torchvision-0.5.0

## Usage

`MLC.py` is used for training a model on dataset with noisy labels and validating it.

Here is an example:

```shell
python python MLC.py --dir experiment/ --dataset 'mnist' --noise_type sn --noise 0.2 --forget-rate 0.2
```

or you can train MLC with shel script:

```shell
sh script/mnist.sh
```


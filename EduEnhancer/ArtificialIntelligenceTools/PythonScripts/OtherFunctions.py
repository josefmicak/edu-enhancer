import torch
import sys


def device_name():
    """
    Returns the information on which device is used to perform computations (GPU/CPU)

    """
    if torch.cuda.is_available() is True:
        print("GPU")
    else:
        print("CPU")


if __name__ == '__main__':
    if sys.argv[1] == 'device_name':
        device_name()
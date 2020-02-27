import argparse

def get_args():
    """ Method to get all commandline arguments for training """
    parser = argparse.ArgumentParser(description="PyTorch Higgs Training")

    parser.add_argument('--epochs', default=20, type=int,
                        metavar='N', help='Total number of epochs to run')

    parser.add_argument('--batch_size', default=1024, type=int,
                        metavar='N', help='training batch size')

    parser.add_argument('--percent_unlabeled', type=float, default=1.0,
                        help='Number of labeled data to have')

    args = parser.parse_args()

    return args

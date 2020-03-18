import argparse


def get_args():
    """ Method to get all commandline arguments for training """
    parser = argparse.ArgumentParser(description="PyTorch Higgs Training")

    parser.add_argument('--epochs', default=20, type=int,
                        metavar='N', help='Total number of epochs to run')

    parser.add_argument('--batch_size', default=128, type=int,
                        metavar='N', help='training batch size')

    parser.add_argument('--percent_unlabeled', type=float, default=1.0,
                        help='Number of labeled data to have')

    parser.add_argument('--learning_rate', type=float,
                        default=0.01, help='Learning rate for neural nets')

    parser.add_argument('--weight_decay', type=float,
                        default=0.01, help='Weight decay for learning rate')

    parser.add_argument('--env', type=int, help='NUMBER OF DIFF ENVIRONMENTS')


    parser.add_argument('--val_iteration', type=int, default=1024, help='Number of labeled data')
    args = parser.parse_args()

    return args

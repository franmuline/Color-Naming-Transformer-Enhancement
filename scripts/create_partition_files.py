import os
import argparse


def fivek_partition(name, path, train, val, test):
    """
    Function to create the directory and files necessary for the specified partition of the FiveK dataset
    Args:
        name: Name of the partition
        path: Path to the dataset
        train: Fraction of the dataset to be used for training
        val: Fraction of the dataset to be used for validation
        test: Fraction of the dataset to be used for testing

    Returns:
        Creates a directory in the specified path with the name of the partition and three files:
        - train.txt: Contains the names of the images to be used for training
        - val.txt: Contains the names of the images to be used for validation
        - test.txt: Contains the names of the images to be used for testing
    """
    TOTAL_SAMPLES = 5000
    train_samples = int(TOTAL_SAMPLES * train)
    val_samples = int(TOTAL_SAMPLES * val)
    test_samples = int(TOTAL_SAMPLES * test)

    # Create the directory
    os.makedirs(path + "/" + name, exist_ok=True)

    # The names of the images from the FiveK dataset are an 'a' followed by four digits, e.g., a0001, a0002, etc.
    # They go from 0001 to 5000

    # Create the train file
    with open(path + "/" + name + "/train.txt", "w") as f:
        for i in range(1, train_samples + 1):
            f.write("a" + str(i).zfill(4) + "\n")

    # Create the val file
    with open(path + "/" + name + "/val.txt", "w") as f:
        for i in range(train_samples + 1, train_samples + val_samples + 1):
            f.write("a" + str(i).zfill(4) + "\n")

    # Create the test file
    with open(path + "/" + name + "/test.txt", "w") as f:
        for i in range(train_samples + val_samples + 1, TOTAL_SAMPLES + 1):
            f.write("a" + str(i).zfill(4) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset name: can be fivek", default="fivek")
    parser.add_argument("--path", type=str, help="Path to the dataset", default="../../Datasets/FiveK")
    parser.add_argument("--name", type=str, help="Name that we will give to the partition", default="Ours")
    parser.add_argument("--train", type=float, help="Train percentage", default=0.7)
    parser.add_argument("--val", type=float, help="Validation percentage", default=0.15)
    parser.add_argument("--test", type=float, help="Test percentage", default=0.15)

    args = parser.parse_args()

    dataset = args.dataset
    if dataset == "fivek":
        fivek_partition(args.name, args.path, args.train, args.val, args.test)
    else:
        raise ValueError("Dataset not supported")

    # Check if the sum of the fractions is equal to 1
    if args.train + args.val + args.test != 1:
        raise ValueError("The sum of the fractions must be equal to 1")

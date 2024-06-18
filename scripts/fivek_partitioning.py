import os
import shutil
import argparse

cvc_input_path = "../../Datasets/FiveK-UEGAN"
cvc_output_path = "../../Datasets/FiveK-UEGAN_partitioned"

input_path = cvc_input_path
output_path = cvc_output_path

extra_dir = "/input"


def copy_files(in_p, out_p, file):
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            # Find the file that starts with the line
            for entry in os.scandir(in_p):
                if entry.is_file() and entry.name.startswith(line):
                    line = entry.name
                    break
            # Copy the image from the input folder to the output folder
            try:
                shutil.copy(in_p + "/" + line, out_p + "/" + line)
            except FileNotFoundError:
                print("File not found: ", line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Path to the data folder",
                        default=input_path + extra_dir)
    parser.add_argument("--train", type=str, help="Path to the train file. Each line corresponds"
                                                                 "to the name of an image in the data folder",
                        default=input_path + "/train.txt")
    parser.add_argument("--val", type=str, help="Path to the val file. Each line corresponds"
                                                               "to the name of an image in the data folder",
                        default=None)
    parser.add_argument("--test", type=str, help="Path to the test file. Each line corresponds"
                                                                "to the name of an image in the data folder",
                        default=input_path + "/test.txt")
    parser.add_argument("--output_path", type=str, help="Path to the output folder",
                        default=output_path + extra_dir)
    args = parser.parse_args()

    in_p = args.input_path
    out_p = args.output_path

    train_f = args.train
    val_f = args.val
    test_f = args.test

    os.makedirs(out_p, exist_ok=True)
    os.makedirs(out_p + "/train", exist_ok=True)
    os.makedirs(out_p + "/val", exist_ok=True)
    os.makedirs(out_p + "/test", exist_ok=True)

    # Copy the files
    copy_files(in_p, out_p + "/train", train_f)
    copy_files(in_p, out_p + "/test", test_f)
    if val_f is not None:
        copy_files(in_p, out_p + "/val", val_f)




import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dir', default='rawdata/17307130181', type=str)
parser.add_argument('--old', default='_', type=str)
parser.add_argument('--new', default='-', type=str)

args = parser.parse_args()

if __name__ == "__main__":
    if args.dir is None or os.path.isfile(args.dir):
        exit()

    files = os.listdir(args.dir)
    for file in files:
        new_name = file.replace(args.old, args.new)

        file_path = os.path.join(args.dir, file)
        new_path = os.path.join(args.dir, new_name)

        os.rename(file_path, new_path)

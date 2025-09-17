from argparse import ArgumentParser

from tests import tests
from globals import config


def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        type=str,
    )

    return parser.parse_args()


def get_filename(args):
    filename = args.filename

    if filename.endswith(".py"):
        return filename.replace(".py", "")

    return filename


if __name__ == "__main__":
    args = get_args()

    filename = get_filename(args)

    print(filename)

    kwargs = {
        "config": config,
    }

    tests[filename](**kwargs)

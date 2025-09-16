from argparse import ArgumentParser

from scripts import scripts
from globals import config


def getArgs():
    parser = ArgumentParser()

    parser.add_argument(
        "-f",
        "--fileName",
        required=True,
        type=str,
    )

    return parser.parse_args()


def get_filename(args):
    fileName = args.fileName

    if fileName.endswith(".py"):
        return fileName.replace(".py", "")

    return fileName


if __name__ == "__main__":
    args = getArgs()

    fileName = get_filename(args)

    kwargs = {
        "config": config,
    }

    scripts[fileName](**kwargs)

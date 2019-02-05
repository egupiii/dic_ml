# Import libraries and modules
import argparse
import numpy as np



# Set command-line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                 description="argparse sample")

parser.add_argument("--num_iters", default=10, type=int, help="number of iterations")

parser.add_argument("--alpha", default=0.1, type=float, help="initial alpha")

parser.add_argument("--display", action="store_true", help="display of calculation process")

parser.add_argument("--text", default="Hello, World!", type=str, help="text sample")


def main(args):
    print(args.text)
    x = args.alpha
    for i in range(args.num_iters):
        x *= 2
        if args.display:
            print(x)
    print("RESULT : {}".format(x))


if __name__ == "__main__":
    # Be run here at first when running the py file

    # Import command-line arguments
    args = parser.parse_args()
    main(args)
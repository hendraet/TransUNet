import os                                                                                                                                                                                                                                                                         
if os.environ.get('REMOTE_PYCHARM_DEBUG_SESSION', False):                                                                                
    import pydevd_pycharm                                           
    pydevd_pycharm.settrace('localhost', port=int(os.environ.get('REMOTE_PYCHARM_DEBUG_PORT', '12022')),
                            stdoutToServer=True, stderrToServer=True)

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
from scipy.ndimage import gaussian_filter1d


def main(args: argparse.Namespace):
    with open(args.log_filename) as log_file:
        lines = log_file.readlines()

    losses = []
    losses_ce = []
    for line in lines:
        if not line.startswith("iteration"):
            continue
        split_line = line.split()
        losses.append(float(split_line[5][:-1]))
        losses_ce.append(float(split_line[7][:-1]))

    step = 1
    num_loss_points = len(losses)
    selected_losses = numpy.asarray(losses)[0::step]
    selected_losses_ce = numpy.asarray(losses_ce)[0::step]
    x = numpy.arange(0, num_loss_points, 1)[0::step]

    loss_smooth = gaussian_filter1d(selected_losses, sigma=16)
    loss_smooth_ce = gaussian_filter1d(selected_losses_ce, sigma=16)

    px = 1 / plt.rcParams['figure.dpi']
    size = 2000
    plt.subplots(figsize=(size * px, size * px))
    plt.xticks(numpy.arange(0, num_loss_points, step=2000), rotation=90)
    plt.plot(x, selected_losses, "-c")
    plt.plot(x, selected_losses_ce, "-y")
    plt.plot(x, loss_smooth, "-b")
    plt.plot(x, loss_smooth_ce, "-r")

    plt.savefig("logs/losses.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_filename", type=Path, help="Path to log file that should be plotted")
    parsed_args = parser.parse_args()
    main(parsed_args)

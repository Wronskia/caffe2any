import pandas as pd
import argparse

# This scripts executes with : python3 inputPROTO.py --file=nv3dfi_deploy.prototxt_bfs.csv
# nv3dfi_deploy.prototxt_bfs is provided by python caffe2any.py examples/nv3dfi_deploy.prototxt

# This script only works with the BFS input of caffe2any
# TODO : adding an if condition to support FC layers betweeen conv/pooling layers

# First we read the csv file given in the command line (BFS output of caffe2any)
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None, help='Extra config parameters')
    return parser.parse_args()


# get methods which parse the BFS file
def get_pad(line):
    return int(line[1][' Node Details'].split("/")[1].split(" ")[1].split("=")[1])


def get_stride(line):
    return int(line[1][' Node Details'].split("/")[1].split(" ")[0].split("=")[1])


def get_k(line):
    return int(line[1][' Node Details'].split("/")[0].split("=")[1].split("x")[1])


def main():
    args = get_arguments()
    output = pd.read_csv(args.file, sep=',')
    layers = []
    # Append all conv and pool layers in a list
    for line in output.iloc[::-1].iterrows():
        if 'conv' in line[1]['Node'] or 'pool' in line[1]['Node']:
            layers.append(line[1]['Node'])

    # Reads the dataframe from the last layer to the first to do the backward computations
    # The idea is to consider the output dimensions of the last conv/pool layer and to compute
    # what should be this input for this output and then take this results to do the same recursively
    # with all conv/pool layers
    # the formula used for conv is :
    '''
    Accepts a volume of size W1×H1×D1
    Requires four hyperparameters:
    Number of filters K,
    their spatial extent F,
    the stride S,
    the amount of zero padding P.
    Produces a volume of size W2×H2×D2 where:
    W2=(W1−F+2P)/S+1 H2=(H1−F+2P)/S+1
    '''
    # and the one for pooling is :
    '''
    Accepts a volume of size W1×H1×D1
    Requires two hyperparameters:
    their spatial extent F,
    the stride S,
    Produces a volume of size W2×H2×D2 where:
    W2=(W1−F)/S+1
    H2=(H1−F)/S+1
    D2=D1
    '''
    for line in output.iloc[::-1].iterrows():
        if 'Conv' in line[1][' Type']:
            k_conv = get_k(line)
            s_conv = get_stride(line)
            pad_conv = get_pad(line)
            if layers[0] == line[1]['Node']:
                OFMy = (int(line[1][' OFMy']) - 1) * s_conv + k_conv - 2 * pad_conv
                OFMx = (line[1][' OFMx'] - 1) * s_conv + k_conv - 2 * pad_conv
                print(str(OFMx) + "----" + str(OFMx))
            else:
                OFMx = (OFMx - 1) * s_conv + k_conv - 2 * pad_conv
                OFMy = (OFMy - 1) * s_conv + k_conv - 2 * pad_conv
                print(str(OFMx) + "----" + str(OFMx))
        if 'Pool' in line[1][' Type']:
            k_pool = get_k(line)
            s_pool = get_stride(line)
            pad_pool = get_pad(line)
            if layers[0] == line[1]['Node']:
                OFMy = (line[1][' OFMy'] - 1) * s_pool + k_pool
                OFMx = (line[1][' OFMx'] - 1) * s_pool + k_pool
                print(str(OFMx) + "----" + str(OFMx))
            else:
                OFMx = (OFMx - 1) * s_pool + k_pool
                OFMy = (OFMy - 1) * s_pool + k_pool
                print(str(OFMx) + "----" + str(OFMx))

    print(str(OFMx) + "------" + str(OFMy))


if __name__ == "__main__":
    main()

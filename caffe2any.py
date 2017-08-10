#!/usr/bin/python

"""
Start by downloading the Caffe proto file from
https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto

Then use protoc to compile it to caffe_pb2
protoc caffe.proto

Finally use to parse it

summary
 -l [deconv|conv|]
"""

# See - http://stackoverflow.com/questions/2970858/why-doesnt-print-work-in-a-lambda
#from __future__ import print_function
import sys
import argparse
from collections import deque, Counter
import caffe_pb2 as caffe
from google.protobuf import text_format
from printers.png import PngPrinter
from printers import csv, console
from caffe_parser import parse_caffe_net
from transformers import inventory
import topology
import copy

DEBUG = False
# options
options = {
    # Merges Convolution and ReLU nodes. This makes for a more compact and readable graph.
    'merge_conv_relu': True,
    # Merges Convolution, ReLU, and Pooling nodes.
    'merge_conv_relu_pooling': False,
    # Merges InnerProduct and ReLU nodes. This makes for a more compact and readable graph.
    'merge_ip_relu': True,
    # For Test/Inference networks, Dropout nodes are not interesting and can be removed for readability
    'remove_dropout': True,
    # For Test/Inference networks, Batch Normalization nodes can sometimes be removed.  This is
    # explained by TensorFlow documentation:
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md#fold_batch_norms
    # TODO: change the implementation so that BN is removed only if we recognize the CONV + BN pattern
    'fold_batchnorm': True,
    # Fold the scale layer into the convolution weights
    # TODO: change the implementation so that BN is removed only if we recognize the CONV + Scale pattern
    'fold_scale': True,
}


def debug(str):
    if DEBUG:
        print (str)

def sum_blob_mem(tplgy, node, blobs, sum):
    if node.type == "Input" or node.role == "Modifier":
        return
    out_edges = tplgy.find_outgoing_edges(node)
    for out_edge in out_edges:
        if out_edge.blob not in blobs:
            shape = out_edge.blob.shape
            sum[0] += out_edge.blob.size()
            blobs.append(out_edge.blob)

from transformers.update_blobs_sizes import update_blobs_sizes
from transformers import fold_transforms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--printer', help='output printer (csv, console, png)', default='console')
    parser.add_argument('-d', '--display', type=str, help='display inventory,unique,output,bfs,mem')
    parser.add_argument('infile', help='input prototxt file')
    args = parser.parse_args()

    net = caffe.NetParameter()

    # Read a Caffe prototxt file
    try:
        f = open(sys.argv[1], "rb")
        text_format.Parse(f.read(), net)
        f.close()
    except IOError:
        exit("Could not open file " + sys.argv[1])

    tplgy = parse_caffe_net(net)
    # calculate BLOBs sizes
    tplgy.traverse(lambda node: update_blobs_sizes(tplgy, node))

    # Handle optional processing on the topology
    if options['remove_dropout']:
        tplgy.remove_node_by_type('Dropout')
    if options['fold_batchnorm']:
        fold_transforms.fold_pair(tplgy, 'Convolution', 'BatchNorm')
    if options['fold_scale']:
        fold_transforms.fold_pair(tplgy, 'Convolution', 'Scale')

    if options['merge_conv_relu']:
        tplgy.merge_nodes('Convolution', 'ReLU', 'Convolution_ReLU')

    if options['merge_ip_relu']:
        tplgy.merge_nodes('InnerProduct', 'ReLU', 'InnerProduct_ReLU')

    # This is a temp bug-bypass
    if args.printer == 'png':
        if options['merge_conv_relu_pooling']:
            tplgy.merge_nodes('Convolution_ReLU', 'Pooling', 'Convolution_ReLU_Pooling')

    # tplgy.dump_edges()
    if args.printer == 'console':
        printer = console.ConsolePrinter()
    elif args.printer == 'png':
        printer = PngPrinter(args, net)
    else:
        printer = csv.CsvPrinter(args.infile + '.csv')

    if args.display != None:
        for disp_opt in args.display.split(','):
            if disp_opt == 'inventory':
                printer.print_inventory(tplgy)
            elif disp_opt == 'unique':
                unique_nodes = {}
                tplgy.traverse(lambda node: add_unique(node, unique_nodes))
                printer.print_unique_all(unique_nodes)
            elif disp_opt == 'output':
                print("outputs:")
                outputs = tplgy.find_output_blobs()
                for output in outputs:
                    print('\t' + output)
            elif disp_opt == 'bfs':
                printer.print_bfs(tplgy)
            elif disp_opt == 'mem':
                sum = [0]
                blobs = []
                #tplgy.traverse_blobs(lambda blob: count_memory(blob, sum))
                tplgy.traverse(lambda node: sum_blob_mem(tplgy, node, blobs, sum))
                print("Total BLOB memory: " + str(sum[0]))
            else:
                exit ("Error: invalid display option")


if __name__ == '__main__':
    main()

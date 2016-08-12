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
from __future__ import print_function
import sys
import argparse
from collections import deque, Counter
import caffe_pb2
from google.protobuf import text_format
from printers import ConsolePrinter, CsvPrinter
import topology

DEBUG = False

def debug(str):
    if DEBUG: print (str)

def is_equal_conv(layer1, layer2):
	param1 = layer1.convolution_param
	kernel_size1 = param1.kernel_size
	stride1 = param1.stride
	pad1 = param1.pad
	
	param2 = layer2.convolution_param
	kernel_size2 = param2.kernel_size
	stride2 = param2.stride
	pad2 = param2.pad

	return (kernel_size1 == kernel_size2 and stride1 == stride2 and pad1==pad2)

def is_equal(layer1, layer2):
	assert layer1.type == layer2.type
	if layer1.type == "Pooling": 
		return layer1.pooling_param == layer2.pooling_param
	if layer1.type == "Convolution": 
		return is_equal_conv(layer1, layer2)
	if layer1.type == "LRN": 
		return layer1.lrn_param == layer2.lrn_param
	return True

def is_unique(layer, unique_list):
	unique = True
	for unique_layer in unique_list:
		if is_equal(unique_layer, layer):
			unique = False
	return unique	

def add_unique(layer, unique_layers):
	if unique_layers.get(layer.type)==None:
		unique_layers[layer.type] = []
	if is_unique(layer, unique_layers[layer.type]):
		unique_layers[layer.type].append(layer)

def update_blobs_size(tplgy, node):
#	print('updating node:' + node.name)
	in_edges = tplgy.find_incoming_edges(node)
	out_edges = tplgy.find_outgoing_edges(node)
	if node.type == 'Convolution':
		assert len(in_edges)==1 and len(out_edges)==1, node.name
		if in_edges[0].blob.shape != None:
			out_edges[0].blob.shape = node.transform(in_edges[0].blob.shape)
	elif node.type == 'ReLU':
		assert len(in_edges)==1, node.name
		if in_edges[0].blob.shape != None:
			for edge in out_edges:
				edge.blob.shape = in_edges[0].blob.shape
	elif node.type == 'Pooling':
		assert len(in_edges)==1 and len(out_edges)==1, node.name
		if in_edges[0].blob.shape != None:
			out_edges[0].blob.shape = node.transform(in_edges[0].blob.shape)
	elif node.type == 'ROIPooling':
		assert len(in_edges)==2 and len(out_edges)==1, node.name
		#print(in_edges[0].blob.shape)
		if in_edges[0].blob.shape != None:
			out_edges[0].blob.shape = in_edges[0].blob.shape
	elif node.type == 'InnerProduct':
		assert len(in_edges)==1 and len(out_edges)==1, node.name
		#print(in_edges[0].blob.shape)
		if in_edges[0].blob.shape != None:
			out_edges[0].blob.shape = in_edges[0].blob.shape
	elif node.type == 'Python':
		pass # Don't know how to handle this
	else:
		assert len(in_edges)==1 and len(out_edges)==1, node.name
		#print(in_edges[0].blob.shape)
		if in_edges[0].blob.shape != None:
			out_edges[0].blob.shape = in_edges[0].blob.shape

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('infile', help='input prototxt file')
	parser.add_argument('-f', '--format', help='output format (csv, console)', default='console')
	parser.add_argument('-d', '--display', type=str, help='display catalog, unique, output, bfs')
	args = parser.parse_args()

	net = caffe_pb2.NetParameter()
	
	# Read a Caffe prototxt file
	try:
		f = open(sys.argv[1], "rb")
		text_format.Parse(f.read(), net)
		f.close()
	except IOError:
		print ("Could not open file ", sys.argv[1])

	tplgy = topology.populate(net)
	# calculate BLOBs sizes
	tplgy.traverse(lambda node: update_blobs_size(tplgy, node))

	if args.format == 'console':
		printer = ConsolePrinter()
	else:
		printer = CsvPrinter(args.infile + '.csv') 

	if args.display != None:
		for disp_opt in args.display.split(','):
			if disp_opt == 'catalog':
				printer.print_catalog(tplgy)
			elif disp_opt == 'unique':
				unique_nodes = {}
				tplgy.traverse(lambda node: add_unique(node.layer, unique_nodes))
				printer.print_unique_all(unique_nodes)
			elif disp_opt == 'output':
				print("outputs:")
				outputs = tplgy.find_output_blobs()
				for output in outputs:
					print('\t' + output)
			elif disp_opt == 'bfs':
				printer.print_bfs(tplgy)
			else:
				exit ("Error: invalid display option")


if __name__ == '__main__':
	main()
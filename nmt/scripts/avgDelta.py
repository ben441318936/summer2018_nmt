"""
Takes two files "trace" and "label", and computes delta, the average deletion probability that transforms "label" into "trace".
"""


import os
import argparse

def parse_argument():

	parser = argparse.ArgumentParser()

	parser.add_argument("trace", help="name of the trace file", type=str)
	parser.add_argument("label", help="name of the label file", type=str)

	return parser.parse_args()

def calculateDelta(trace_file,label_file):

	__location__ = os.getcwd()

	trace_file = os.path.join(__location__, trace_file)
	label_file = os.path.join(__location__, label_file)

	with open(trace_file) as trace:
		with open(label_file) as label:

			num_seq = sum(1 for line in trace)
			trace.seek(0)

			delta = 0
			for (trace_line, label_line) in zip(trace,label):
				delta += (len(label_line) - len(trace_line)) / len(label_line)

	return delta / num_seq


def main():
	args = parse_argument()

	trace_file = args.trace
	label_file = args.label

	print(calculateDelta(trace_file,label_file))



if __name__ == "__main__":
	main()





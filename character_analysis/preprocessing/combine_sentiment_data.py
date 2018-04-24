import os, sys, random
import re, csv
import pandas as pd

pos_file = "../data/rt-polaritydata/rt-polarity.pos"
neg_file = "../data/rt-polaritydata/rt-polarity.neg"
out_file = "../data/rt-polarity.csv"

COL_TEXT = "text"
COL_LABEL = "is_char"


def combine_data():

	pos_lines = []
	with open(pos_file, "r", encoding='ISO-8859-1') as pos_f:
		for line in pos_f:
			pos_lines.append(line)

	neg_lines = []
	with open(neg_file, "r", encoding='ISO-8859-1') as neg_f:
		for line in neg_f:
			neg_lines.append(line)

	print("pos: ", len(pos_lines))
	print("neg: ", len(neg_lines))

	assert(len(pos_lines) == len(neg_lines))

	pos_label = ["1"] * len(pos_lines)
	neg_label = ["0"] * len(neg_lines)

	data = pos_lines + neg_lines 
	labels = pos_label + neg_label 

	df = pd.DataFrame({COL_TEXT: data, COL_LABEL: labels})

	df.to_csv(out_file, index=False)


if __name__ == "__main__":

	combine_data()

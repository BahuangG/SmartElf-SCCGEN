# coding:utf-8
from nlgeval import compute_metrics
import numpy as np
import csv
import os

filename = 'newdata/sml-1.csv'
tempfile = 'newdata/temp.csv'

# 将原始文件中的数据全部转换为小写，并写入一个临时文件中
with open(filename, 'r') as infile, open(tempfile, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for row in reader:
        lowercase_row = [word.lower() for word in row]
        writer.writerow(lowercase_row)

# 用临时文件覆盖原始文件
os.remove(filename)
os.rename(tempfile, filename)



metrics_dict = compute_metrics(hypothesis='newdata/sml-1.csv',
                                   references=['newdata/nl.csv'],no_skipthoughts=True, no_glove=True)

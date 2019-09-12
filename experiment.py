from util import *
import numpy as np
import matplotlib.pyplot as plt
import csv

gt_f = "plane_data/plane_data_gt.csv"

f = "plane_data/plane_data_measure_0.1.csv"
output = "output3.csv"

f = open(f,"r")
reader1 = csv.reader(f)
data = [i for i in reader1]
title = data[0]
data = np.array(data[1:])[:,1:].tolist()
data = [[np.array(i,dtype=np.float)] for i in data]

# gt_f = open(gt_f,"r")
# reader2 = csv.reader(f)
# gt = [i for i in reader2]
# gt = np.array(gt[1:])[:,1:].tolist()
# gt = [[np.array(i,dtype=np.float)] for i in gt]

dimension = len(data[0][0])

tracker = Tracker(dimension=dimension,Q=0.001,R=10,dtype=float)
lines,ret = tracker.track(data)
lines = np.array(lines[0],dtype=str)
ind = np.array([[i] for i in range(1,501)],dtype=str)
lines = np.hstack((ind,lines))
lines = np.vstack((title,lines))

print(lines.shape)

writer = csv.writer(open(output,"w"))
for i in lines:
    writer.writerow(i)




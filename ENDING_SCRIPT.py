#!/usr/bin/env python3

#### DAG Adjusted Sysem test ENDER

import csv
import subprocess
import time
import os
from os.path import exists

## next, we get the output file names and check the last line for success
valid_combos = []

submits = []
with open("SUBMITS.txt", "r") as f:
    submits = list(f.readlines())
    #print(submits)

for i in submits:
    check_tag = 0
    job_tag = i[7:-5] #Pulling from SUBMITS.txt
    #print(job_tag,"job tag\n")
    out_name = '_out_'+job_tag+'.out'
    err_name = '_err_'+job_tag+'.err'

    #print(out_name,"output name")

    if os.path.exists(out_name):
        #print("exists")
        with open(out_name, 'r') as f:
            last_line = f.readlines()[-1]
            #print(last_line)
            if "success" in last_line:
                check_tag+=1
            f.close()
        #if os.stat(err_name).st_size == 0:
            #check_tag +=1
    if check_tag:
        valid_combos.append(job_tag)

#print(valid_combos)


with open("valid_combos.csv", 'w') as f:
    f.write("Framework,Compute Capability,Cuda Library,Framework Version \n")
    for combo in valid_combos:
        combo = combo.replace("_", ",")
        f.write(combo+'\n')
    f.close()

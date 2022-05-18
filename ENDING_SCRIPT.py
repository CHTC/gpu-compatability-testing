#!/usr/bin/env python3

#### DAG Adjusted Sysem test ENDER

import csv
import subprocess
import time
import os
from os.path import exists

## next, we get the output file names and check the last line for success
valid_combos = []
invalid_combos = []

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
    if not os.path.exists(out_name):
        failure = "Failure to Match"

    if os.path.exists(out_name):
        with open(out_name, 'r') as f:
            last_line_f = f.readlines()[-1]
            #print(last_line)
            if "success" in last_line_f:
                check_tag+=1
            elif "failure" or "failed" in last_line_f:
                failure = "Environment Resolution Failure"
            elif "gpu_issue" in last_line_f:
                failure = "GPU/Framework Comm Failure"
       
    if check_tag:
        valid_combos.append(job_tag)
    else:
        invalid_combos.append((job_tag+"_"+failure))


with open("valid_combos.csv", 'w') as f:
    f.write("Framework, Compute Capability, Cuda Library, Framework Version \n")
    for combo in valid_combos:
        combo = combo.replace("_", ",")
        f.write(combo+'\n')
    f.close()

with open("invalid_combos.csv", 'w') as f:
    f.write("Framework, Compute Capability, Cuda Library, Framework Version, Failure Cause \n")
    for combo in invalid_combos:
        combo = combo.replace("_", ",")
        f.write(combo+'\n')
    f.close()


#!/usr/bin/env python3

#### DAG Adjusted Sysem test ENDER

import csv
import subprocess
import time
import os
from os.path import exists

## next, we get the output file names and check the last line for success


with open("tuple_test.txt", "r") as f:
    submit = f.readlines()
    #print(submits)

for i in submit:
    check_tag = 0
    job_tag = i[7:-4] #Pulling from SUBMITS.txt
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
        with open(job_tag+"_test_results.txt\n", 'w') as f:
            f.write("Test succeeded")
            f.close
    else:
        with open(job_tag+"_test_results.txt\n",'w') as f:
            f.write("Test failed: "+failure)
            f.close()
if os.path.isfile('cleanup_tuple.txt'):
    with open('cleanup_tuple.txt','r') as f:
        for i in f:
            command = 'rm -f '+i[0:-1]
            subprocess.run(command, shell=True)

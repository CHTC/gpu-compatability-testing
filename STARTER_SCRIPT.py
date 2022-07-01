#!/usr/bin/env python3
#### DAG Adjusted Sysem test STARTER

import csv
import subprocess
import time
import os
from os.path import exists
import sys


##########################
#Set up CL arguments
##########################

if sys.argv[4].lower() in ['true']:
    cleanup=1
else:
    cleanup=0
if cleanup:
        cleanup_files = []

tf_params = subprocess.run("conda search tensorflow-gpu -c conda-forge | grep -E -o ' [0-9]+.[0-9]+.[0-9]+ ' | cut -d. -f 1-2 | awk '{$1=$1;print}' | uniq", check=True, capture_output=True,shell=True)
tf_params = list(tf_params.stdout.decode('utf-8').splitlines())
tf_params = tf_params[-1*int(float(sys.argv[1])):]
tf_params = list(filter(lambda i: i[:1] != "1", tf_params))
 
pt_params = subprocess.run("conda search pytorch -c pytorch | grep -E -o ' [0-9]+.[0-9]+.[0-9]+ ' | cut -d. -f 1-2 | awk '{$1=$1;print}' | uniq", check=True, capture_output=True,shell=True)
pt_params = list(pt_params.stdout.decode('utf-8').splitlines())
pt_params = pt_params[-1*int(float(sys.argv[2])):]

cuda_lib_params = subprocess.run("conda search cudatoolkit -c conda-forge | grep -E -o ' [0-9]+.[0-9]+.[0-9]+ ' | cut -d. -f 1-2 | awk '{$1=$1;print}' | uniq", check=True, capture_output=True,shell=True)
cuda_lib_params = list(cuda_lib_params.stdout.decode('utf-8').splitlines())
cuda_lib_params = cuda_lib_params[-1*int(float(sys.argv[3])):]

## Get valid system values for compute capability

result = subprocess.run(["condor_status","-compact","-constraint","TotalGpus > 0","-af","CUDACapability"], stdout=subprocess.PIPE)

compute_capability_params = list(set(result.stdout.decode('utf-8').split("\n")))

#remove	bad values
bad_inds = []
for ind,val in enumerate(compute_capability_params):
        if val=="undefined" or len(val)==0:
                bad_inds.append(ind)
bad_inds.reverse()
for i in bad_inds:
        del compute_capability_params[i]
compute_capability_params.sort() ## Sorted list of available compute capabilities

                        
submits = [] #Create list to start tracking all the jobs to be submitted

##########################
#Tensorflow tests
##########################
for compute_capability in compute_capability_params:
    for cuda_lib_version in cuda_lib_params:
        #tf_version = "2.6"
        for tf_version in tf_params:
            trial = [compute_capability,cuda_lib_version,tf_version]
            job_tag = "tf_"+trial[0]+"_"+trial[1]+"_"+trial[2]
            latest_minor_cuda_lib = trial[1]
            if latest_minor_cuda_lib[-1] == "0":
                cuda_lib_range = trial[1]
            else:
                cuda_lib_range = trial[1][0:-1]+"0"
                
            #Set up environment .yml for TF    
            env_yml = """channels:
- conda-forge
- defaults
dependencies:
- tensorflow-gpu={}
- cudatoolkit={}""".format(tf_version,cuda_lib_version)
        
            env_name = "environment_{}.yml".format(str(job_tag).replace(".",""))
            with open(env_name, 'w') as f:
                f.write(env_yml)
                f.close()
            if cleanup:
                cleanup_files.append(env_name)

            #Set up test script for TF jobs
            test_script = """import tensorflow as tf
import os

num_GPUs = len(tf.config.list_physical_devices('GPU'))
cuda_dev = os.environ['CUDA_VISIBLE_DEVICES']
if num_GPUs>0 and len(cuda_dev)>0:
    print("success")
else:
    print("gpu_issue")"""

            script_name = "script_{}.py".format(job_tag)
            with open(script_name, 'w') as f:
                f.write(test_script)
                f.close()
            if cleanup:
                cleanup_files.append(script_name)

            #Set up script to test for GPU on execute node
            run_file = """#!/bin/bash
set -e

# installation steps for Miniconda
export HOME=$PWD
export PATH
sh Miniconda3-latest-Linux-x86_64.sh -b -p $PWD/miniconda3
export PATH=$PWD/miniconda3/bin:$PATH

# install environment
conda env create -f {} -n {}
source activate base 
conda activate {}

# modify this line to run your desired Python script
python3 {}

""".format(env_name,env_name,env_name,script_name)
            run_name = "run_{}.sh".format(job_tag)
            with open(run_name,'w') as f:
                f.write(run_file)
                f.close()
            if cleanup:
                cleanup_files.append(run_name)

            #Set up TF submit files    
            submit_file = """universe = vanilla

log = _job_{}.log
#
executable = {}

output = _out_{}.out
error = _err_{}.err
#
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {},{},{}

require_gpus = (Capability == {}) && ({} <= DriverVersion)
periodic_remove = (time() - QDate) > (24 * 3600)
request_cpus = 1
request_gpus = 1
request_memory = 8GB
request_disk = 40GB

+WantGPULab = true
+GPUJobLength = "short"

queue 1""".format(job_tag,run_name,job_tag,job_tag,"http://proxy.chtc.wisc.edu/SQUID/gpu-examples/Miniconda3-latest-Linux-x86_64.sh",env_name,script_name,trial[0],trial[1])
        
            submit_name = 'submit_{}.sub'.format(job_tag)
            with open(submit_name, 'w') as f:
                f.write(submit_file)
                f.close()
            if cleanup:
                cleanup_files.append(submit_name)
            submits.append(submit_name)






##########################
#PyTorch tests
##########################
for compute_capability in compute_capability_params:
    for cuda_lib_version in cuda_lib_params:
        for pt_version in pt_params:
            trial = [compute_capability,cuda_lib_version,pt_version]
            job_tag = "pt_"+trial[0]+"_"+trial[1]+"_"+trial[2]
            latest_minor_cuda_lib = trial[1]
            if latest_minor_cuda_lib[-1] == "0":
                cuda_lib_range = trial[1]
            else:
                cuda_lib_range = trial[1][0:-1]+"0"
                
            #Set up conda env .yml for PT jobs    
            env_yml = """channels:
- pytorch
- defaults
dependencies:
- pytorch={}
- torchvision
- cudatoolkit={}""".format(pt_version,cuda_lib_version)
        
            env_name = "environment_{}.yml".format(str(job_tag).replace(".",""))
            with open(env_name, 'w') as f:
                f.write(env_yml)
                f.close()
            if cleanup:
                cleanup_files.append(env_name)

            #Set up PT test script to test for GPU on execute node
            test_script = """import torch
cuda_available = torch.cuda.is_available()
num_GPUs = torch.cuda.device_count()
current_device = torch.cuda.current_device()
current_dev_name = torch.cuda.get_device_name(current_device)

if cuda_available and num_GPUs>0 and len(current_dev_name)>0:
    print("success")
else:
    print("gpu_issue")"""

            script_name = "script_{}.py".format(job_tag) ##Here I'm using the fact that the pt_versions and tf_versions don't overlap
            with open(script_name, 'w') as f:
                f.write(test_script)
                f.close()
            if cleanup:
                cleanup_files.append(script_name)    

            #Set up PT run file
            run_file = """#!/bin/bash
set -e

# installation steps for Miniconda
export HOME=$PWD
export PATH
sh Miniconda3-latest-Linux-x86_64.sh -b -p $PWD/miniconda3
export PATH=$PWD/miniconda3/bin:$PATH

# install environment
conda env create -f {} -n {}
source activate base 
conda activate {}

# modify this line to run your desired Python script
python3 {}

""".format(env_name,env_name,env_name,script_name)
            run_name = "run_{}.sh".format(job_tag)
            with open(run_name,'w') as f:
                f.write(run_file)
                f.close()
            if cleanup:
                cleanup_files.append(run_name)    
                
            #Set up PT submit files
            submit_file = """universe = vanilla

log = _job_{}.log
#
executable = {}

output = _out_{}.out
error = _err_{}.err
#
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_input_files = {},{},{}

require_gpus = (Capability == {}) && ({} <= DriverVersion)
periodic_remove = (time() - QDate) > (24 * 3600)
request_cpus = 1
request_gpus = 1
request_memory = 8GB
request_disk = 40GB

+WantGPULab = true
+GPUJobLength = "short"

queue 1""".format(job_tag,run_name,job_tag,job_tag,"http://proxy.chtc.wisc.edu/SQUID/gpu-examples/Miniconda3-latest-Linux-x86_64.sh",env_name,script_name,trial[0],trial[1])
        
            submit_name = 'submit_{}.sub'.format(job_tag)
            with open(submit_name, 'w') as f:
                f.write(submit_file)
                f.close()
            if cleanup:
                cleanup_files.append(submit_name)

            submits.append(submit_name)


#Add files to cleanup if option is selected
if cleanup:
    cleanup_files.append('A.dag*')
    cleanup_files.append('MY_DAG.*')
    cleanup_files.append('_*')
    cleanup_files.append('SUBMITS.txt')
    cleanup_files.append("cleanup.txt")


#If cleanup is selected, write these files to be deleted by ENDING_SCRIPT.py
if cleanup:
    with open('cleanup.txt','w') as myfile:
        for i in cleanup_files:
            myfile.write(i+'\n')
        myfile.close()
        
#Set up DAG file for handling job submissions        
with open("A.dag","w") as myfile:
    for i,n in enumerate(submits):
        myfile.write("JOB "+str(i)+" "+str(n)+"\n")

#File that ENDING_SCRIPT.py uses to track/aggregate job info (necessary because jobs that time out won't give an output file)
with open("SUBMITS.txt","w") as myfile:
    for i in submits:
        myfile.write(i+"\n")
  

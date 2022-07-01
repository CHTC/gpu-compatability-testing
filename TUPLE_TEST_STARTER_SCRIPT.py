#!/usr/bin/env python3

import csv
import subprocess
import time
import os
from os.path import exists
import sys


##########################
#Set up CL arguments
##########################
framework = str(sys.argv[1])
framework_version = str(sys.argv[2])
compute_cap = str(sys.argv[3])
cuda_lib_version = str(sys.argv[4])
if sys.argv[5].lower() in ['true']:
    cleanup=1
else:
    cleanup=0
if cleanup:
    cleanup_files = []

#Set job tag to handle file naming
job_tag = "{}_{}_{}_{}".format(framework, framework_version, compute_cap, cuda_lib_version)

##########################
#Query CHTC for available Compute Capabilities
##########################
result = subprocess.run(["condor_status","-compact","-constraint","TotalGpus > 0","-af","CUDACapability"], stdout=subprocess.PIPE)

compute_capability_params = list(set(result.stdout.decode('utf-8').split("\n")))

#remove	bad and repeat values 
bad_inds = []
for ind,val in enumerate(compute_capability_params):
    if val=="undefined" or len(val)==0:
        bad_inds.append(ind)
bad_inds.reverse()
for i in bad_inds:
    del compute_capability_params[i]
compute_capability_params.sort() ## Sorted list of available compute capabilities

##########################
#Check that the requested compute capability is available in the system  
##########################                      
if compute_cap not in compute_capability_params:
    print("The selected compute capability is not currently available on CHTC")
    #No point submitting a doomed job, so exit
    exit()



##########################
#Set up files specific to each framework
##########################


##########################
#TensorFlow
##########################
if framework == "tf":
    #Environment .yml file
    env_yml = """channels:
- conda-forge
- defaults
dependencies:
- tensorflow-gpu={}
- cudatoolkit={}""".format(framework_version,cuda_lib_version)
    env_name = "environment_{}.yml".format(str(job_tag).replace("_",""))

    #Script used to test if tf is working properly on execute node
    test_script = """import tensorflow as tf
import os

num_GPUs = len(tf.config.list_physical_devices('GPU'))
cuda_dev = os.environ['CUDA_VISIBLE_DEVICES']
if num_GPUs>0 and len(cuda_dev)>0:
    print("success")
else:
    print("gpu_issue")"""

    script_name = "TF_test.py"


##########################
#PyTorch
##########################
if framework == "pt":

    #PyTorch environment .yml file
    env_yml = """channels:
- pytorch
- defaults
dependencies:
- pytorch={}
- torchvision
- cudatoolkit={}""".format(framework_version,cuda_lib_version)
    env_name = "environment_{}.yml".format(str(job_tag).replace("_",""))

    #Script used to test if pt is working properly on execute node
    test_script = """import torch
cuda_available = torch.cuda.is_available()
num_GPUs = torch.cuda.device_count()
current_device = torch.cuda.current_device()
current_dev_name = torch.cuda.get_device_name(current_device)

if cuda_available and num_GPUs>0 and len(current_dev_name)>0:
    print("success")
else:
    print("gpu_issue")"""

    script_name = "PT_test.py" 

##########################
#Communal Files and Writing Specific Files
##########################

##########################
#Environment file
##########################
env_name = "test_env.yml"
with open(env_name, 'w') as f:
    f.write(env_yml)
    f.close()
if cleanup:
    cleanup_files.append(env_name)

##########################
#Write the Script File
##########################
with open(script_name, 'w') as f:
    f.write(test_script)
    f.close()
if cleanup:
    cleanup_files.append(script_name)
    
##########################
#Run File
##########################
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

#If cleanup attribute is true, start tracking which files to delete
if cleanup:
    cleanup_files.append(run_name)

##########################
#Submit File
##########################
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

queue 1""".format(job_tag,run_name,job_tag,job_tag,"http://proxy.chtc.wisc.edu/SQUID/gpu-examples/Miniconda3-latest-Linux-x86_64.sh",env_name,script_name,compute_cap,cuda_lib_version)
        
submit_name = 'submit_{}.sub'.format(job_tag)

#Write the submit file
with open(submit_name, 'w') as f:
    f.write(submit_file)
    f.close()
if cleanup:
    cleanup_files.append(submit_name)

#Additional files to cleanup
if cleanup:
    cleanup_files.append("B.dag*")
    cleanup_files.append("tuple_test.txt")
    cleanup_files.append("_*")
    cleanup_files.append("TUPLE_TEST_DAG*")
    cleanup_files.append("cleanup_tuple.txt")

##########################
#Building Dag
##########################
with open("B.dag","w") as myfile:
    myfile.write("JOB 1 "+submit_name+"\n")
    f.close()

#File for the post processing script to grab the job name to look for
with open("tuple_test.txt","w") as myfile:
    myfile.write(submit_name)
    f.close()

#Write all the files to clean
if cleanup:
    with open('cleanup_tuple.txt','w') as myfile:
        for i in cleanup_files:
            myfile.write(i+'\n')
        myfile.close()

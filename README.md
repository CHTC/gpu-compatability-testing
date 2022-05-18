This directory contains a testing suite designed to gain insight about the compatibilities of various GPU 
parameters in CHTC, such as DL framework versions, CUDA runtime versions, and CUDA compute capability versions. It 
contains a ```STARTER_SCRIPT.py``` that generates job input files based on version pairs pulled from Conda, 
```MY_DAG.dag``` that submits and manages each test job, and ```ENDING_SCRIPT.py``` that interprets the results of 
each job. It outputs two files, ```valid_combos.csv``` and ```invalid_combos.csv``` that list valid version pairs 
and invalid version pairs (along with a reason for failure), respectively. Two additional files, 
```Miniconda3-latest-Linux-x86_64.sh``` and ```A.sub``` are used to support running the jobs.

To run the test, decide how many of the most recent parameters should be tested (eg, do you want to test the 10 
latest versions of PyTorch/TensorFlow, and the latest 5 CUDA runtime libraries) and edit the arguments passed in 
```MY_DAG.dag``` after the line containing ```SCRIPT PRE A STARTER_SCRIPT.py```. These arguments correspond to 
testing the previous XX most recent versions of TensorFlow, PyTorch, and CUDA runtime libraries, in that order. Note that the CUDA Compute capability is not a configurable setting, because ```STARTER_SCRIPT.py``` queries the CHTC system and only tests those compute capabilities that exist within the system. To  submit the test job, run ```dagman_submit MY_DAG.dag```. After 24 hours, ```ENDING_SCRIPT.py``` will interpret the 
results and return the valid and invalid combinations. 


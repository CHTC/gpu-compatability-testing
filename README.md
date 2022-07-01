# GPU Compatibility Testing

This directory contains a testing suite designed to gain insight about the compatibilities of various GPU 
parameters in CHTC, such as DL framework versions, CUDA runtime versions, and CUDA compute capability versions. 

There are two tools: one that tests some portion of the entire version compatibility space, and one that tests only a single tuple of versions. Both tools are interfaced with through ```wrapper.py```

## Single Tuple Test

To test a single tuple of versions, run on the submit node:

```python3 wrapper.py -c -t [framework] [framework version] [compute capability] [CUDA library version]```

where the ```-c``` flag specifies that the wrapper script should clean up files as they are no longer needed (recommended) and the ```-t``` flag specifies that the test is to be run against a single tuple. The framework argument takes either ```tf``` or ```pt```, and the version arguments take the corresponding versions to test. Upon completion, the results of the test will be output in a file called ```[framework]_[framework version]_[compute capability]_[CUDA lib version]_test_results.txt```. The contents of this file will explain the results of the test, indicating either ```Test succeeded``` or ```Test failed: [reason for failure]```.

For example, to test TensorFlow 2.6 against Compute Capability 7.5 with Cuda Library Version 11.2, you would run:

```python3 wrapper.py -c -t tf 2.6 7.5 11.2```

When this command is run, it will spawn a single parent DAG process through DAGman that handles the DAG submissions. This parent process will then spawn a second DAG handler job that generates submit files and submits the job. This should be visible through ```condor_q``` as two running jobs. Shortly thereafter, the actual test job will enter the queue, hopefully match to an execute node (although if there is no existing match, it will sit idle until timeout), and run. When the test job is finished and has exited the queue, the DAG handler process will run a post script to collect results/clean up extraneous files and also exit the queue. Finally, the parent process will also terminate. In total, three jobs will have run, and the entire process usually takes ~15-20 minutes.

## Version Space Test

To test some subspace of the entire version space, run on the submit node:

```python3 wrapper.py -c -ws [# of most recent TensorFlow versions] [# of most recent PyTorch versions] [# of most recent CUDA library versions]```

where the ```-c``` flag specifies that the wrapper script should clean up files as they are no longer needed (recommended) and the ```-ws``` flag specifies that the test is to run against the version space as opposed to a single tuple. Upon completion, valid combos will be stored in a file called ```valid_combos.csv``` and invalid combos will be stored in a file called ```invalid_combos.csv```.

Compute capabilities are not passed to the wrapper, because the version space test will run against all available compute capabilities available on the system.

For example, to run a version space test that tests the 6 most recent versions of TensorFlow, the 3 most recent versions of PyTorch and 8 of the most recent CUDA library versions, you would run:

```python3 wrapper.py -c -ws 6 3 8```

Much like the single tuple test, running this script will first spawn a parent DAG process to handle the other DAG processes. This parent process will then spawn a second DAG handler job that generates submit files and submits all the jobs. When all job files have been generated, each test job will enter the queue (this number can get quite large, on the order of hundreds, depending on the parameters passed to the wrapper script). After 24 hours, any jobs that remain in the queue will be killed on the assumption that there is no existing match for them. The DAG handler process will then run a post script to aggregate results and clean up extraneous files. Upon completion, the handler process will terminate, and finally the parent DAG process will also terminate.

## File Descriptions
```wrapper.py``` -- A wrapper script: allows the user to easily run and configure the version space and tuple tests with user-defined inputs.

```STARTER_SCRIPT.py``` -- Used by version space test: a script that generates job input files based on version pairs pulled from Conda.

```ENDING_SCRIPT.py``` -- Used by version space test: a script that handles post-processing of the test and deletes extraneous files.

```A.sub``` -- Used by version space test: a submit file that allows DAGman to launch a slim handler job that runs ```STARTER_SCRIPT.py``` and ```ENDING_SCRIPT.py```.

```TUPLE_TEST_STARTER_SCRIPT.py``` -- Used by the tuple test: a script that generates the job input files based on the arguments/versions passed to it.

```TUPLE_TEST_ENDING_SCRIPT.py``` -- Used by the tuple test: a script that handles post-processing of the tuple test and deletes extraneous files.

```Miniconda3-latest-Linux-x86_64.sh``` -- Used by all jobs: a portable Miniconda installation file used to build conda envs in all jobs. This file is not located in the repo, but rather on CHTC's SQUID server, where it is globally accessible. 

```LICENSE``` -- The license governing use of this software.


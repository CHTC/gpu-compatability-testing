# GPU Compatibility Testing
This directory contains a testing suite designed to gain insight about the compatibilities of various GPU 
parameters in CHTC, such as DL framework versions, CUDA runtime versions, and CUDA compute capability versions. 

There are two tools: one that tests some portion of the entire version compatibility space, and one that tests only a single tuple of versions. Both tools are interfaced with through ```wrapper.py```

## Single Tuple Test

To test a single tuple of versions, run on the submit node:

```python3 wrapper.py -c -t [framework] [framework version] [compute capability] [CUDA library version]```

where the ```-c``` flag specifies that the wrapper script should clean up files as they are no longer needed (recommended) and the ```-t``` flag specifies that the test is to be run against a single tuple. The framework argument takes either ```tf``` or ```pt```, and the version arguments take the corresponding versions to test. Upon completion, the results of the test will be output in a file called ```test_results.txt```.

For example, to test TensorFlow 2.6 against Compute Capability 7.5 with Cuda Library Version 11.2, you would run:

```python3 wrapper.py -c -t tf 2.6 7.5 11.2```

## Version Space Test

To test some subspace of the entire version space, run on the submit node:

```python3 wrapper.py -c -ws [# of most recent TensorFlow versions] [# of most recent PyTorch versions] [# of most recent CUDA library versions]```

where the ```-c``` flag specifies that the wrapper script should clean up files as they are no longer needed (recommended) and the ```-ws``` flag specifies that the test is to run against the version space as opposed to a single tuple. Upon completion, valid combos will be stored in a file called ```valid_combos.csv``` and invalid combos will be stored in a file called ```invalid_combos.csv```.

For example, to run a version space test that tests the 6 most recent versions of TensorFlow, the 3 most recent versions of PyTorch and 8 of the most recent CUDA library versions, you would run:

```python3 wrapper.py -c -ws 6 3 8```



## File Descriptions
```wrapper.py``` -- A wrapper script: allows the user to easily run and configure the version space and tuple tests with user-defined inputs.

```STARTER_SCRIPT.py``` -- Used by version space test: a script that generates job input files based on version pairs pulled from Conda.

```ENDING_SCRIPT.py``` -- Used by version space test: a script that handles post-processing of the test and deletes extraneous files.

```A.sub``` -- Used by version space test: a submit file that allows DAGman to launch a slim handler job that runs ```STARTER_SCRIPT.py``` and ```ENDING_SCRIPT.py```.

```TUPLE_TEST_STARTER_SCRIPT.py``` -- Used by the tuple test: a script that generates the job input files based on the arguments/versions passed to it.

```TUPLE_TEST_ENDING_SCRIPT.py``` -- Used by the tuple test: a script that handles post-processing of the tuple test and deletes extraneous files.

```Miniconda3-latest-Linux-x86_64.sh``` -- Used by all jobs: a portable Miniconda installation file used to build conda envs in all jobs.




import argparse
import subprocess
parser = argparse.ArgumentParser(description='Select job run-type.')

parser.add_argument(
    '--whole-space', 
    '-ws',
    type=int,
    nargs=3, 
    help='pick values to determine how many recent versions of framework, compute capability, and cuda library to search')
parser.add_argument(
    '--tuple', 
    '-t',
    nargs=4,
    help='Pick a framework (pt or tf), framework version, compute capability, and cuda library version')
parser.add_argument(
    '--cleanup',
    '-c',
    action='store_true',
    help='Cleanup all leftover files'
    )
args = parser.parse_args()

if args.whole_space:
    MY_DAG = '''SUBDAG EXTERNAL A A.dag
    #Edit the next line
    SCRIPT PRE A STARTER_SCRIPT.py {} {} {} {}
    SCRIPT POST A ENDING_SCRIPT.py'''.format(args.whole_space[0],args.whole_space[1],args.whole_space[2],args.cleanup)
    
    dag_file_name = "MY_DAG.dag"
    with open(dag_file_name, 'w') as f:
        f.write(MY_DAG)
        f.close()
    subprocess.run(['condor_submit_dag','-f',dag_file_name])



elif args.tuple:
    TUPLE_TEST_DAG = '''SUBDAG EXTERNAL A B.dag
#Edit the next line, selecting "tf" or "pt" for framework, and then framework version, compute capability, and cuda lib version
SCRIPT PRE A TUPLE_TEST_STARTER_SCRIPT.py {} {} {} {} {}
SCRIPT POST A TUPLE_TEST_ENDING_SCRIPT.py'''.format(str(args.tuple[0]),args.tuple[1],args.tuple[2],args.tuple[3],args.cleanup)
    dag_file_name = "TUPLE_TEST_DAG.dag"
    with open(dag_file_name, 'w') as f:
        f.write(TUPLE_TEST_DAG)
        f.close()
    subprocess.run(['condor_submit_dag','-f',dag_file_name])
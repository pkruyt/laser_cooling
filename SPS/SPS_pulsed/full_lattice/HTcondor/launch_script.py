import os,json,shutil,stat,sys
import numpy as np

python_script_source_path = '/afs/cern.ch/user/p/pkruyt/public/IBS/IBS_kicks.py'
python_script_name = os.path.basename(python_script_source_path)

sequence_file_path = '/afs/cern.ch/user/p/pkruyt/public/IBS/sps.json'

# initiate settings for output
settings = {}
settings['output_directory_afs'] = '/afs/cern.ch/user/p/pkruyt/public/IBS/output_test'
settings['output_directory_eos'] = '/eos/user/p/pkruyt/HTcondor/output'
turnbyturn_file_name = 'lead.npz'
turnbyturn_path_eos = os.path.join(settings['output_directory_eos'], turnbyturn_file_name)

job_file_name = os.path.join(settings['output_directory_afs'], 'IBS_kicks_gpu.job')


bash_script_path = os.path.join(settings['output_directory_afs'],'IBS_kicks_gpu.sh')
bash_script_name = os.path.basename(bash_script_path)
bash_script = open(bash_script_path,'w')
bash_script.write(
f"""#!/bin/bash\n
echo 'sourcing environment'
source /afs/cern.ch/user/p/pkruyt/miniforge3/bin/activate
echo 'Running job'
python {python_script_name} 1> out.txt 2> err.txt
echo 'Done'
xrdcp -f {turnbyturn_file_name} {turnbyturn_path_eos}
xrdcp -f out.txt {os.path.join(settings['output_directory_eos'],"out_gpu.txt")}
xrdcp -f err.txt {os.path.join(settings['output_directory_eos'],"err_gpu.txt")}
xrdcp -f abort.txt {os.path.join(settings['output_directory_eos'],"abort_gpu.txt")}
""")

bash_script.close()
os.chmod(bash_script_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IXOTH)

job_file = open(job_file_name,'w')

job_file.write(
f'''executable        = {bash_script_path}
transfer_input_files  = {python_script_source_path},{sequence_file_path}
output                = {os.path.join(settings['output_directory_afs'],"IBS_kicks_gpu.out")}
error                 = {os.path.join(settings['output_directory_afs'],"IBS_kicks_gpu.err")}
log                   = {os.path.join(settings['output_directory_afs'],"IBS_kicks_gpu.log")}
request_GPUs = 1
+MaxRuntime           = 1209500
queue''')

job_file.close()

#shutil.copytree(settings['output_directory_afs'],settings['output_directory_eos'])

#os.system(f'condor_submit -spool {job_file_name}')
os.system(f'condor_submit {job_file_name}')
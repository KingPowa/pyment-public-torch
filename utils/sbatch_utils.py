import os
import subprocess

def query_lr():
    env = os.environ
    if "OPT_LR" in env:
        return float(env["OPT_LR"])
    else: return None

class SbatchScheduler:

    def __init__(self, 
                 lr: float,
                 script_loc: str):
        self.lr = lr
        self.script_loc = script_loc

    def run(self):
        command = ['sbatch', *self.__format_arguments()]
        command_str = ' '.join(command)
        print(f"Command issued: {command_str}")
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            env=self.__set_env())
    
        # Wait for the process to finish and capture stdout and stderr
        stdout, stderr = process.communicate()

        if stderr:
            raise ValueError(f"Error in submitting the jobs: {stderr.decode()}")
        
        # Capture the job ID from the stdout
        job_id = None
        if stdout:
            # Slurm typically returns output like: "Submitted batch job <job_id>"
            output_str = stdout.decode().strip()
            if "Submitted batch job" in output_str:
                job_id = output_str.split()[-1]  # The last word is the job ID

        return job_id

    def __set_env(self):
        curr_env = os.environ
        curr_env["OPT_LR"] = str(self.lr)
        return curr_env
    
    def __format_arguments(self):

        return [self.script_loc]
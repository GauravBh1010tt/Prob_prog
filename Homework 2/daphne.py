import json
import subprocess

def daphne(args, cwd='/Users/gaurav/Desktop/CS532-HW2/daphne'):
    proc = subprocess.run(['lein','run','-f','json'] + args,
                          capture_output=True, cwd=cwd)
    if(proc.returncode != 0):
        raise Exception(proc.stdout.decode() + proc.stderr.decode())
    return json.loads(proc.stdout)


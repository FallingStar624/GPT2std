import subprocess
import sys
import re
import argparse

class Dummy(object):
    def __init__(self, user):
        self.user = user
        self.mem = 0
        self.gpu = 0
        
    def add(self, mem, gpu):
        self.mem += mem
        self.gpu += gpu
        
    def __repr__(self):
        return f"Mem : {self.mem}, GPU: {self.gpu}"

    def __str__(self):
        return f"{self.user:15s} | Mem : {self.mem:11d} | GPU: {self.gpu}"
    
def find_num_gpu(string):
    pattern = "gpu:3090:(\d)"
    result = re.findall(pattern, string)
    if len(result) == 0:
        return 0
    return int(result[0].split(":")[-1])
        
def info(string):
    sub_string = string.split()
    node_name = ''
    num_gpu = ''
    memory = ''
    for s in sub_string:
        if s.startswith('Nodes'):
            node_name = s.split('=')[-1]
        if s.startswith('Mem'):
            memory = int(s.split('=')[-1])
        if s.startswith('GRES'):
            rgpu = re.search('gpu[a-zA-Z0-9\_\:]*:\d', s.split('=')[-1])
            if rgpu is None:
                num_gpu = 0
            else:
                num_gpu = int(rgpu.group()[-1])
    return node_name, memory, num_gpu
    
def summary_node(name, mem, alloc_mem, gpu, alloc_gpu):
    if name:
        print(f"{name:15s} | Mem {alloc_mem:6d}/{mem:6d} | GPU {alloc_gpu}/{gpu}")


def resource_per_node():
    cmd_reculsive = ["scontrol", "show", "nodes", "-d"]

    bytes = subprocess.check_output(cmd_reculsive)
    outputs = bytes.decode('utf-8')

    curr_node = ''
    node_mem = 0
    alloc_mem = 0
    node_gpu = 0
    alloc_gpu = 0
    
    total_gpu = 0
    total_alloc_gpu = 0

    for out in outputs.split('\n'):
        name = re.search('NodeName=[a-z]+\d{2}', out)
        if name: 
            summary_node(curr_node, node_mem, alloc_mem, node_gpu, alloc_gpu)
            total_gpu += node_gpu
            total_alloc_gpu += alloc_gpu
            curr_node = name.group().split('=')[-1]
            node_gpu = 0
            alloc_gpu = 0
        gpu = re.search('Gres=gpu:[a-zA-Z0-9\_]+:\d', out)
        if gpu: node_gpu = int(gpu.group()[-1])
        alloc_gpus = re.search('GresUsed=gpu:[a-zA-Z0-9\_]+:\d', out)
        if alloc_gpus: alloc_gpu = int(alloc_gpus.group()[-1])
        mems = re.search('RealMemory=\d+ AllocMem=\d+', out)
        if mems:
            node_mem = int(re.search('RealMemory=\d+', out).group().split('=')[-1])
            alloc_mem = int(re.search('AllocMem=\d+', out).group().split('=')[-1])

    summary_node(curr_node, node_mem, alloc_mem, node_gpu, alloc_gpu)        
    total_gpu += node_gpu
    total_alloc_gpu += alloc_gpu
    print(f"Total GPU Usage : {total_alloc_gpu}/{total_gpu}")
    print()

def resource_per_user():
    cmd_reculsive = ["scontrol", "show", "jobs", "-d"]

    bytes = subprocess.check_output(cmd_reculsive)
    outputs = bytes.decode('utf-8')

    users = dict()
    user_name = ''
    is_running = False
    for out in outputs.split('\n'):
        user = re.search('UserId=[a-zA-Z0-9\_]+', out)
        if user: 
            is_running = False
            user_name = user.group().split('=')[-1]
            if not user_name in users:
                users[user_name] = Dummy(user_name)
        job_state = re.search('JobState=[a-zA-Z]+', out)
        if job_state:
            is_running = job_state.group().split('=')[-1] == 'RUNNING'
        if is_running:
            resources = re.search('Nodes=[a-z]\d{2}\sCPU_IDs=[0-9\,\-]+\sMem=\d+\sGRES=[a-zA-Z0-9\_\:]*', out)
            if resources:
                _, mem, gpu = info(resources.group())
                users[user_name].add(mem, gpu)
           
    total_gpu = 0
    for user, value in users.items():
        if value.mem > 0:
            print(value)
        total_gpu += value.gpu
    print(f"Total GPU Usage : {total_gpu}")
    print()
        
parser = argparse.ArgumentParser(description='Summary Resource Usage')
parser.add_argument('-n', '--node', action='store_true', help='resources per node')
parser.add_argument('-u', '--user', action='store_true', help='resources per user')

def main():
    args = parser.parse_args()
    print()
    if args.node:
        resource_per_node()
    if args.user:
        resource_per_user()

if __name__ == '__main__':
    main()
    

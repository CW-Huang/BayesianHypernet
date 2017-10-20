import os
import itertools
import numpy as np
import subprocess



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--launch', type=int, default=1, help="set to 0 for a dry_run")
parser.add_argument('--hours_per_job', type=int, default=1, help="expected run time, in hours")
#parser.add_argument('--exp_script', type=str, default='$HOME/memgen/dk_mlp.py')
locals().update(parser.parse_args().__dict__)


# TODO: save a log of the launched jobs
"""
This is a template for launching a set of experiments, using line_search, and/or grid_search.

You need to specify:
    exp_script
    grid/line search
    test or not
    

Optionally, specify:
    duration, memory, Theano/TensorFlow


-------------------------------------------

-------------------------------------------
WHAT IS LINE SEARCH???
line_search searches over one hyper-parameter at a time, leaving all the others fixed to default values.
For instance:
    dropout_p in [0,.25,.5,.75]
    l2 in [.1, .01, .001, .0001, 0.]
would give 9 experiments, not 20

We can combine line_search with grid_search; this amounts to considering ALL of the line_searches as ONE DIMENSION for grid_search

-------------------------------------------

job_str: complete bash command


"""

# TODO: move these elsewhere?
def grid_search(args_vals):
    """ arg_vals: a list of lists, each one of format (argument, list of possible values) """
    lists = []
    for arg_vals in args_vals:
        arg, vals = arg_vals
        ll = []
        for val in vals:
            ll.append(" --" + arg + "=" + str(val))
        lists.append(ll)
    return ["".join(item) for item in itertools.product(*lists)]

def line_search(args_vals):
    search = []
    for arg_vals in args_vals:
        arg, vals = arg_vals
        for val in vals:
            search.append(" --" + arg + "=" + str(val))
    return search

def combine_grid_line_search(grid, line):
    return ["".join(item) for item in itertools.product(grid, line)]

def test():
    test_args_vals = []
    test_args_vals.append(['lr', [.1,.01]])
    test_args_vals.append(['num_hids', [100,200,500]])
    gs, ls, gls = grid_search(test_args_vals), line_search(test_args_vals), 0#, grid_search([line_search(test_args, test_values)
    print gs
    print ls
    print len(combine_grid_line_search(gs, ls))

#test()




# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------


job_prefix = ""

# TODO: tensorflow...
# Check which cluster we're using
if subprocess.check_output("hostname").startswith("hades"):
    #launch_str = "smart-dispatch --walltime=48:00:00 --queue=@hades launch THEANO_FLAGS=device=gpu,floatX=float32"
    job_prefix += "smart-dispatch --walltime=24:00:00 --queue=@hades launch THEANO_FLAGS=device=gpu,floatX=float32 python "
elif subprocess.check_output("hostname").startswith("helios"):
    job_prefix += "jobdispatch --gpu --queue=gpu_1 --duree=12:00H --env=THEANO_FLAGS=device=gpu,floatX=float32 --project=jvb-000-ag python "
else: # TODO: SLURM
    print "running at MILA, assuming job takes about", hours_per_job, "hours_per_job"
    job_prefix += 'sbatch --gres=gpu -C"gpu6gb|gpu12gb" --mem=4000 -t 0-' + str(hours_per_job)


# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------




exp_script = ' $HOME/projects/BayesianHypernets/BayesianHypernet/dk_anomaly_detection.py '
job_prefix += exp_script


grid = [] 
#grid += [["acq", ['bald', 'max_ent', 'var_ratio', 'random']]]
grid += [["coupling", ['4', '6', '8', '12', '16', '32']]]
grid += [["model", ['hnet']]]
grid += [["lr0", ['.001', '.0001', '.0003']]]
grid += [["nonlinearity", ['rectify', 'gelu']]]
grid += [["save", ['1']]]

# TODO: savepath should also contain exp_script? 
#   (actually, we should make a log of everything in a text file or something...)
#   we could copy the launcher to the save_dir (but need to check for overwrites...)
launcher_name = os.path.basename(__file__)
#https://stackoverflow.com/questions/12842997/how-to-copy-a-file-using-python
#print os.path.abspath(__file__)
#import shutil



job_strs = []
for settings in grid_search(grid):
    job_str = job_prefix + settings
    job_str += " --save_dir=" + os.environ["SAVE_PATH"] + "/" + launcher_name
    print job_str
    job_strs.append(job_str)



if launch:
    for job_str in job_strs:
        os.system(job_str)




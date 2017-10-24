import os
import itertools
import numpy as np
import subprocess



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--launch', type=int, default=1, help="set to 0 for a dry_run")
parser.add_argument('--hours_per_job', type=int, default=1, help="expected run time, in hours")
# TODO
#parser.add_argument('--n_splits', type=int, default=20, help="number of train/valid splits, for cross-validation")
#parser.add_argument('--n_hopt', type=int, default=100, help="number of random hparam settings to try")

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
    args_vals = args_vals[::-1] # reverse the order; least important grid dimensions go last!
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


# TODO: harder than it seemed at first :P 
def line_search_with_defaults(args_vals):
    """ Here, we assume that the first entry in every list of vals is the default argument for that arg"""
    search = []
    defaults = [" --" + arg_vals[0]+ "=" + str(arg_vals[1][0]) for arg_vals in args_vals]
    print defaults
    for arg_vals in args_vals:
        arg, vals = arg_vals
        for val in vals[1:]:
            search.append(" --" + arg + "=" + str(val))
    search = [list_outer_product()] # TODO
    return search

def list_outer_product(list1, list2):
    return ["".join(item) for item in itertools.product(list1, list2)]

def test():
    test_args_vals = []
    test_args_vals.append(['lr', [.1,.01]])
    test_args_vals.append(['num_hids', [100,200,500]])
    gs, ls, gls = grid_search(test_args_vals), line_search(test_args_vals), 0#, grid_search([line_search(test_args, test_values)
    print gs
    print ls
    print len(list_outer_product(gs, ls))

test_args_vals = []
test_args_vals.append(['lr', [.1,.01]])
test_args_vals.append(['num_hids', [100,200,500]])
#test()
#print list_outer_product(["model=1", "m=2"], ['a', 'b', 'c'])
#print line_search(test_args_vals)
#print line_search_with_defaults(test_args_vals)
#assert False

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
    #job_prefix += 'sbatch --gres=gpu -C"gpu6gb|gpu12gb" --mem=4000 -t 0-' + str(hours_per_job)
    job_prefix += 'sbatch --mem=4000 -t 0-' + str(hours_per_job)


# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------




exp_script = ' $HOME/BayesianHypernetCW/experiment_MLP_WN_Regression_Task.py '
job_prefix += exp_script


# TODO: for later use
"""
# RANDOM SEARCH: (maybe we should start with a grid, though...)
args = []
for seed in [np.random.randint(2**32 - 1)) for _ in range(n_hopt)]:
    arg += " --seed="+str(seed)
    for dataset in ['boston', 'concrete', 'energy', 'kin8nm', 'naval', 'power', 'protein', 'wine', 'yacht', 'year']: # TODO: year
        arg += " --dataset="+str(dataset)
        for model in ['BHN', 'MCDropout', 'Backprop']:
            arg += " --model="+str(model)
            # all methods
            arg += " --lr=" + str(np.exp(np.random.uniform(-1.5,-4.5)))
            arg += ' --n_hiddens=' + str(np.random.choice([1,2,4])) # n_layers
            arg += " --n_units=" + str(np.random.choice(range(50,1000)))
            arg += " --lbda=" + str( np.exp(np.random.normal(-1, 1.5))) # precision: lower = more variance!
            # MCdropout
            if model == 'MCDropout':
                arg += " --drop_prob=" + str(np.random.uniform(.005, .05))
            # BHNs
            if model == "BHN":
                coupling = np.random.choice([2,4,8])
                flow = np.random.choice(['IAF', 'RealNVP'])
                if flow == 'RealNVP':
                    coupling *= 2
                arg += " --flow="+flow
                arg += " --coupling="+str(coupling)
            args += arg
"""

# settings for MCDropout
MCD_grid = []
MCD_grid += [["drop_prob", [.001, .002, .005, .01, .02, .05, .1, .2]]]
MCDs = [" --model=MCDropout " + item for item in  grid_search(MCD_grid)]
# settings for BHNs
BHN_grid = []
BHN_grid += [["flow", ["IAF", "RealNVP"]]]
BHN_grid += [["coupling", [2,4,8,16]]]
BHNs = [" --model=BHN " + item for item in  grid_search(BHN_grid)]
models = BHNs + MCDs + [" --model=Backprop "]
print models
models = [model + ' --dataset=airfoil --cross_validate=20' for model in models]


grid = [] 
grid += [["lr0", ['.1', '.01', '.001', '.0001', '.00001']]]
grid += [['n_hiddens', [1,2,4]]]
grid += [["n_units", [50,100,200,500,1000,2000]]]
grid += [["lbda", [.1,.2,.5,1,2,5,10]]]

all_args = list_outer_product(line_search(grid), models)


# TODO: savepath should also contain exp_script? 
#   (actually, we should make a log of everything in a text file or something...)
#   we could copy the launcher to the save_dir (but need to check for overwrites...)
launcher_name = os.path.basename(__file__)
#https://stackoverflow.com/questions/12842997/how-to-copy-a-file-using-python
#print os.path.abspath(__file__)
#import shutil

job_strs = [job_prefix + args + " --save_dir=" + os.environ["SAVE_PATH"] + "/" + launcher_name for args in all_args]
for job_str in job_strs:
    print job_str

print "\n\n\n\t\t\t\t TOTAL NUMBER OF JOBS="+str(len(job_strs))+"\n\n\n"

if launch:
    for job_str in job_strs:
        os.system(job_str)




import os
import itertools
import numpy as np
import subprocess



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--launch', type=int, default=1, help="set to 0 for a dry_run")
#parser.add_argument('--hours_per_job', type=int, default=3, help="expected run time, in hours")
#parser.add_argument('--exp_script', type=str, default='$HOME/memgen/dk_mlp.py')
locals().update(parser.parse_args().__dict__)

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

# TODO
if subprocess.check_output("hostname").startswith("hades"):
    job_prefix += "smart-dispatch --walltime=24:00:00 --queue=@hades launch THEANO_FLAGS=device=gpu,floatX=float32 python "


# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------




# TODO
exp_script = ' $HOME/BayesianHypernetCW/regression.py '
job_prefix += exp_script


model_strs = []
model_strs += [" --model=MCD --drop_prob=.01"] # TODO: dataset-specific
model_strs += [" --model=BHN --flow=IAF --coupling=4"]
        #--drop_prob=" + str(p) for p in [.05, .01, .005]]


# TODO
grid = [] 
grid += [["lr0", ['.01']]]
grid += [["lbda", [1]]]
grid += [["split", [0]]]
grid += [["epochs", [100]]]
grid += [['dataset', ['airfoil', 'parkinsons'] + ['boston', 'concrete', 'energy', 'kin8nm', 'naval', 'power', 'protein', 'wine', 'yacht', 'year --epochs=10']]]

#
launcher_name = os.path.basename(__file__)

job_strs = []
for settings in grid_search(grid):
    job_str = job_prefix + settings
    # TODO
    job_str += " --save_dir=" + os.environ["SAVE_PATH"] + "/" + launcher_name
    for model_str in model_strs:
        _job_str = job_str + model_str
        print _job_str
        job_strs.append(_job_str)

print "njobs", len(job_strs)


if launch:
    for job_str in job_strs:
        os.system(job_str)




import os
import itertools
import numpy as np
import subprocess



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--launch', type=int, default=1, help="set to 0 for a dry_run")
parser.add_argument('--eval_only', type=int, default=0)
parser.add_argument('--train_on_valid', type=int, default=0)

#parser.add_argument('--hours_per_job', type=int, default=3, help="expected run time, in hours")
#parser.add_argument('--exp_script', type=str, default='$HOME/memgen/dk_mlp.py')
locals().update(parser.parse_args().__dict__)

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
if os.path.exists('/home/capybara/this_is_guillimin'):
    #launch_str = "smart-dispatch --walltime=48:00:00 --queue=@hades launch THEANO_FLAGS=device=gpu,floatX=float32"
    #job_prefix += "smart-dispatch --walltime=10:00:00 --queue=@guillimin launch THEANO_FLAGS=floatX=float32 python "
    job_prefix += "smart-dispatch --walltime=10:00:00 launch THEANO_FLAGS=floatX=float32 python "
elif subprocess.check_output("hostname").startswith("briaree"):
    job_prefix += "jobdispatch --duree=0:50:0 --mem=4G --env=THEANO_FLAGS=floatX=float32 python "
elif subprocess.check_output("hostname").startswith("hades"):
    job_prefix += "smart-dispatch --walltime=24:00:00 --queue=@hades launch THEANO_FLAGS=device=gpu,floatX=float32 python "
elif subprocess.check_output("hostname").startswith("helios"):
    job_prefix += "jobdispatch --gpu --queue=gpu_1 --duree=1:00H --env=THEANO_FLAGS=device=gpu,floatX=float32 --project=jvb-000-ag python "
else: # TODO: SLURM
    assert False
    print "running at MILA, assuming job takes about", hours_per_job, "hours_per_job"
    #job_prefix += 'sbatch --gres=gpu -C"gpu6gb|gpu12gb" --mem=4000 -t 0-' + str(hours_per_job)
    job_prefix += 'sbatch --gres=gpu --mem=4000 -t 0-' + str(hours_per_job)


# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------




exp_script = ' $HOME/BayesianHypernetCW/regression.py '
job_prefix += exp_script


model_strs = []
model_strs += [" --model=MCD --drop_prob=.01", "--model=BHN --flow=IAF --coupling=4"]


grid = [] 
grid += [["lr0", ['.01', '.001']]]
grid += [["lbda", 100.**np.arange(-3,2)]]
#grid += [["length_scale", ['1e-6', '1e-4', '1e-2', '1e-1', '1']]]
grid += [['dataset', ['boston', 'concrete', 'energy', 'wine', 'yacht']]]
grid += [['split', range(20)]]

#
launcher_name = os.path.basename(__file__)

job_strs = []
for settings in grid_search(grid):
    job_str = job_prefix + settings
    job_str += " --save_dir=" + os.environ["SAVE_PATH"] + "/" + launcher_name
    if eval_only:
        job_str += ' --eval_only=1 '
    if train_on_valid:
        job_str += ' --train_on_valid'
    for model_str in model_strs:
        _job_str = job_str + model_str
        print _job_str
        job_strs.append(_job_str)

print "njobs", len(job_strs)

if launch:
    for job_str in job_strs:
        os.system(job_str)




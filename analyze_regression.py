import time
t0 = time.time()
import numpy
np = numpy
#from pylab import *
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--EXP', type=str, default='dev_regression')
locals().update(parser.parse_args().__dict__)

#EXPS = ["dev_regression", "large", "protein", "small", "year"]

if EXP == "dev_regression":
    datasets = ['airfoil', 'parkinsons']
    epochs = 400
    n_units = 50
    splits = range(20)
elif EXP == "large":
    datasets = ['naval', 'kin8nm', 'power']
    epochs = 400
    n_units = 50
    splits = range(20)
elif EXP == "protein":
    datasets = ['protein']
    epochs = 200
    n_units = 100
    splits = range(5)
elif EXP == "small":
    datasets = ['boston', 'concrete', 'energy', 'wine', 'yacht']
    epochs = 400
    n_units = 50
    splits = range(20)
elif EXP == "year":
    datasets = ['airfoil', 'parkinsons']
    epochs = 100
    n_units = 100
    splits = range(1)


# These never change
taus = 10.**np.arange(-3,6)
lbdas = 100.**np.arange(-3,2)
lrs = ['.01', '.001']
models = ['BHN_flow=IAF_coupling=4', 'MCD_drop_prob=.01']

# find all "final" results
dd = os.path.join(os.environ['SAVE_PATH'], 'launch_' + EXP + '.py')
pp = [p for p in os.listdir(dd) if 'FINAL' in p]
print "done loading", time.time() - t0

# 
if os.environ['CLUSTER'] == 'MILA':
    script_name = 'slurm_script'
else:
    script_name = 'regression.py'

# 
def get_fname(d, lbda, lr, m, s):#, tau):
    #return os.path.join(dd, script_name + '___dataset=' + d + '_epochs=' + str(epochs) + '_lbda=' + str(lbda) + '_lr0=' + str(lr) + '_n_units=' + str(n_units) + '_split=' + str(s) + '_model=' + m)# + '_tau=' + str(tau))
    return script_name + '___dataset=' + d + '_epochs=' + str(epochs) + '_lbda=' + str(lbda) + '_lr0=' + str(lr) + '_n_units=' + str(n_units) + '_split=' + str(s) + '_model=' + m



bests = {} # for all splits
mean_best_RMSEs = {} # averaged across splits
std_best_RMSEs = {} # averaged across splits
mean_best_LLs = {} # averaged across splits
std_best_LLs = {} # averaged across splits

# Here, we find the best hparams for each (dataset, model) INDEPENDENTLY for each split (to avoid hyperoptimization on test set)
#   We average the test performances for each (dataset, model), using the hyperparameters that performed best on the valid set OF THAT SPLIT
#   This gives us a fair estimate of the performance of the combined (model, hyperoptimization) algorithm.
#       In our current case, hopt is just grid_search.
for d in datasets:
    for m in models:
        bests[(d,m)] = {}
        va_best_RMSEs = [] 
        te_best_RMSEs = []
        va_best_LLs = [] 
        te_best_LLs = []
        for s in splits:
            # find the best results for this model on this dataset, FOR EACH SPLIT
            # we do this *independently* for RMSE and LL 
            va_best_RMSE_ss = ''
            va_best_RMSE = np.inf
            te_best_RMSE = np.inf
            va_best_LL_ss = ''
            va_best_LL = -np.inf
            te_best_LL = -np.inf
            for lbda in lbdas:
                for lr in lrs:
                    ss = get_fname(d, lbda, lr, m, s)
                    print '\t\t' + ss
                    pp_ = [p for p in pp if ss in p]

                    # RMSE:
                    rr = [ p for p in pp_ if p.startswith(ss + '_FINAL_va_RMSE=') ]
                    assert len(rr) in [0,1]
                    if len(rr) == 1:
                        RMSE = np.round(float(rr[0].split('_RMSE=')[1]), 3)
                        if RMSE < va_best_RMSE:
                            va_best_RMSE_ss = ss
                            va_best_RMSE = RMSE
                            # use the hyperparameters with the best validation performance on the test set:
                            te_best_RMSE = np.round(float([ p for p in pp_ if p.startswith(ss + '_FINAL_te_RMSE=') ][0].split('_RMSE=')[1]), 3)
                        print rr[0]
                    else:
                        print rr

                    # LL:
                    for tau in taus:
                        rr = [ p for p in pp_ if p.startswith(ss + '_tau=' + str(tau) + '_FINAL_va_LL=') ]
                        assert len(rr) in [0,1]
                        if len(rr) == 1:
                            LL = np.round(float(rr[0].split('_LL=')[1]), 3)
                            if LL > va_best_LL:
                                va_best_LL_ss = ss
                                va_best_LL = LL
                                # use the hyperparameters with the best validation performance on the test set:
                                te_best_LL = np.round(float([ p for p in pp_ if p.startswith(ss + "_tau=" + str(tau) + '_FINAL_te_LL=') ][0].split('_LL=')[1]), 3)


            va_best_RMSEs.append(va_best_RMSE)
            te_best_RMSEs.append(te_best_RMSE)
            va_best_LLs.append(va_best_LL)
            te_best_LLs.append(te_best_LL)
            # record best (TODO: RMSE)
            #bests[(d,m)][s] = (va_best, te_best, va_best_ss)
            
        mean_best_RMSEs[(d,m)] = (np.mean(va_best_RMSEs), np.mean(te_best_RMSEs))
        std_best_RMSEs[(d,m)] = (np.std(va_best_RMSEs), np.std(te_best_RMSEs))
        mean_best_LLs[(d,m)] = (np.mean(va_best_LLs), np.mean(te_best_LLs))
        std_best_LLs[(d,m)] = (np.std(va_best_LLs), np.std(te_best_LLs))

        print "dataset, model, time elapsed = ", d, m, time.time() - t0
                        

# TODO: print these out earlier / above?
# THIS IS WHAT WE NEED FOR THE TABLE!
print "mean LL (valid, test) for each method and dataset"
for k,v in mean_best_LLs.items():
    print "\t", k, v
print "std LL (valid, test) for each method and dataset"
for k,v in std_best_LLs.items():
    print "\t", k, v

print "mean RMSE (valid, test) for each method and dataset"
for k,v in mean_best_RMSEs.items():
    print "\t", k, v
print "std RMSE (valid, test) for each method and dataset"
for k,v in std_best_RMSEs.items():
    print "\t", k, v

print "Done, total time =", time.time() - t0



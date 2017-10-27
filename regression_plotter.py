from pylab import *
import os

pp = os.path.listdir('.')

taus = 10**np.arange(-6,3)
lbdas = 10**np.arange(-3,2)

splits = range(20)

lrs = [.01, .001]
models = ['BHN', 'MCD'] #TODO
dats = ['airfoil', 'parkinsons']


"""
algorithm:
    for each (model, dataset):
        for each split
"""

vap = [p for p in os.listdir('.') if 'FINAL_va_LL' in p]
var = {p: float(p.split('LL=')[1]) for p in vap}
tep = [p for p in os.listdir('.') if 'FINAL_te_LL' in p]
ter = {p: float(p.split('LL=')[1]) for p in tep}


bests = {} # for all splits
avg_bests = {} # averaged across splits

# Here, we find the best hparams for each (dataset, model) INDEPENDENTLY for each split (to avoid hyperoptimization on test set)
#   We average the test performances for each (dataset, model), using the hyperparameters that performed best on the valid set OF THAT SPLIT
#   This gives us a fair estimate of the performance of the combined (model, hyperoptimization) algorithm.
#       In our current case, hopt is just grid_search.
for dat in dats:
    for model in models:
        bests[(dat,model)] = {}
        va_bests = [] 
        te_bests = []
        for split in splits:
            # find the best results for this model on this dataset, FOR EACH SPLIT
            va_best = -np.inf
            va_best_ss = ''
            for tau in taus:
                for lbda in lbda:
                    for lr in lrs:
                        ss = '/home/capybara/BayesianHypernetCW/launchers/launch_dev_regression.py/regression.py___dataset=' + dat + '_split=' + str(split)+ '_lr0=' + str(lr) + '_epochs=400_lbda=' + str(lbda) + '_model=' + model + '_tau=' + str(tau) + '_FINAL_va_LL='
                        res = float([p for p in vap if p startswith ss][0].split('va_LL=')[1])
                        if res > va_best:
                            va_best = res
                            va_best_ss = ss
                            te_best = float([p for p in tep if p startswith ss][0].split('te_LL=')[1])
            va_bests.append(va_best)
            te_bests.append(te_best)
            # save best
            bests[(dat,model)][split] = (va_best, te_best, va_best_ss)
            
        avg_bests[(dat,model)] = (mean(va_bests), mean(te_bests))

print avg_bests








        # TODO: plotting (I'll want to load results above and average across splits... *sigh*

    #figure(); suptitle('RMSE ' + dat)
    #figure(); suptitle('LL ' + dat)




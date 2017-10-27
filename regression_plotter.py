from pylab import *
import os


taus = 10.**np.arange(-6,3)
lbdas = 100.**np.arange(-3,2)

splits = range(20)

lrs = ['.01', '.001']
models = ['BHN', 'MCD'] #TODO
models = ['BHN_flow=IAF_coupling=4', 'MCD_drop_prob=.01'] #TODO
dats = ['airfoil', 'parkinsons']


"""
algorithm:
    for each (model, dataset):
        for each split
"""
dd = '/home/capybara/BayesianHypernetCW/launchers/'
dd = '/data/lisatmp4/kruegerd/CLUSTER_RESULTS/mp2/OLD/'
dd += 'launch_dev_regression.py/'

vap = [p for p in os.listdir(dd) if 'FINAL_va_LL' in p]
var = {p: float(p.split('LL=')[1]) for p in vap}
tep = [p for p in os.listdir(dd) if 'FINAL_te_LL' in p]
ter = {p: float(p.split('LL=')[1]) for p in tep}

#pp = [p for p in os.listdir(dd) if 'FINAL_va_LL' in p]


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
            te_best = -np.inf
            va_best_ss = ''
            for tau in taus:
                for lbda in lbdas:
                    for lr in lrs:
                        #ss = '/home/capybara/BayesianHypernetCW/launchers/launch_dev_regression.py/'
                        ss = 'regression.py___dataset=' + dat + '_split=' + str(split)+ '_lr0=' + str(lr) + '_epochs=400_lbda=' + str(lbda) + '_model=' + model + '_tau=' + str(tau) + '_FINAL_va_LL='
                        strs = [p for p in vap if p.startswith(ss)]
                        if len(strs) > 0:
                            print strs
                            va_LL = float(strs[0].split('FINAL_va_LL=')[1])
                            if va_LL > va_best:
                                va_best = va_LL
                                va_best_ss = ss
                                te_best = float([p for p in tep if p.startswith(ss)][0].split('te_LL=')[1])
                        else:
                            print  ss
                        #res = float([p for p in vap if p.startswith(ss)][0].split('va_LL=')[1])
            va_bests.append(va_best)
            te_bests.append(te_best)
            # save best
            bests[(dat,model)][split] = (va_best, te_best, va_best_ss)
            
        avg_bests[(dat,model)] = (mean(va_bests), mean(te_bests))
        """
                        ss = 'regression.py___dataset=' + dat + '_split=' + str(split)+ '_lr0=' + str(lr) + '_epochs=400_lbda=' + str(lbda) + '_model=' + model + '_tau=' + str(tau) + '_FINAL_va_LL='
                        #/data/lisatmp4/kruegerd/CLUSTER_RESULTS/mp2/OLD/launch_dev_regression.py/regression.py___dataset=airfoil_split=0_lr0=.01_epochs=400_lbda=0.01_model=MCD_drop_prob=.01_tau=1000.0_FINAL_va_LL=-5468571.749
                        #/data/lisatmp4/kruegerd/CLUSTER_RESULTS/mp2/OLD/launch_dev_regression.py/regression.py___dataset=airfoil_split=0_lr0=.01_epochs=400_lbda=0.01_model=MCD_drop_prob=.01_FINAL_va_RMSE=124.214
                        strs = [p for p in pp if p.startswith(ss)]
                        if len(strs) > 0:
                            print strs
                        else:
                            print  ss
                        va_LL = float(strs[0].split('FINAL_va_LL=')[1])
                        if va_LL > va_best:
                            va_best_ss = ss
                            va_best = va_LL
                            te_best = float(strs[0].split('FINAL_te_LL=')[1])
                            #print '\t\t\t' + ss
                        #import ipdb; ipdb.set_trace()#os.
                        """
                        

print avg_bests



# TODO: plotting (I'll want to load results above and average across splits... *sigh*
if 0:
    figure(); suptitle('RMSE ' + dat)
    figure(); suptitle('LL ' + dat)




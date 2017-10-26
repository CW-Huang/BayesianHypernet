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


for dat in dats:
    for model in models:
        for split in splits:
            # find the best results for this model on this dataset, FOR EACH SPLIT
            va_best = -np.inf
            va_best_ss = ''
            for tau in taus:
                for lbda in lbda:
                    for lr in lrs:
                        ss = '/home/capybara/BayesianHypernetCW/launchers/launch_dev_regression.py/regression.py___dataset=' + dat + '_split=' + str(split)+ '_lr0=' + str(lr) + '_epochs=400_lbda=' + str(lbda) + '_model=' + model + '_tau=' + str(tau) + '_FINAL_va_LL='
                        os.
                        
                k
            best = launch_dev_regression.py/regression.py___dataset=parkinsons_split=18_lr0=.01_epochs=400_lbda=100.0_model=BHN_flow=IAF_coupling=4_tau=0.001_FINAL_va_LL=












        # TODO: plotting

    figure(); suptitle('RMSE ' + dat)


    




    figure(); suptitle('LL ' + dat)


    ss = '/home/capybara/BayesianHypernetCW/launchers/launch_dev_regression.py/regression.py___dataset=' + dat + '_split=' + str(split)+ '_lr0=' + str(lr) + '_epochs=400_lbda=' + str(lbda) + '_model=' + model


from capy_utils import *
from pylab import *

# TODO: need to clean this up, 


if 0: # adjust ylims
    for mm, metric in enumerate(['_va_LL', '_va_RMSE']):
        if mm == 0:
            ylims = (-.1,.16)
        else:
            ylims = (.4,.6)

        figure(mm+1)
        
        for ii, save_str in enumerate(save_strs):
            for row, tau in enumerate(taus):
                for col, l in enumerate(ls):
                    subplot(len(taus), len(ls), len(ls)*row + col + 1)
                    ylim(ylims)


# 
save_strs = [os.environ['SAVE_PATH'] + d for d in [
        '/slurm_script___grid_search=1/airfoil_regression_mBHNp0.005c4lr00.001seed1791095845reinit1flowIAF',
        '/slurm_script___grid_search=1_model=MCDropout/airfoil_regression_mMCDropoutp0.01c4lr00.001seed1791095845reinit1flowIAF']]

#
ls = 100.**np.arange(-4,2)[::-1]#ls = [.1, .01,  .001] # length scale should be smaller!
taus = 3.**np.arange(3,8)[::-1]#taus = [.01, .1, 1, 10., 100.] # tau should be larger!
ls = [str(l) for l in ls]
taus = [str(tau) for tau in taus]
trials = [str(t) for t in range(10)]


#
if 1: # plot va_LL and va_RMSE
    for mm, metric in enumerate(['_va_LL', '_va_RMSE']):
        if mm == 0:
            ylims = (-.5,1)
        else:
            ylims = (.3,.9)
        # OVERRIDE!
        if mm == 0:
            ylims = (-.3,.7)
        else:
            ylims = (.4,.5)
        
        figure()
        suptitle(metric + "blue is BHN, orange is dropout")
        resultz = []
        for ii, save_str in enumerate(save_strs):
            #
            all_save_strs = [save_str + 'trial' + trial for trial in trials]
            all_save_strs = [ss + 'tau' + tau for tau in taus for ss in all_save_strs]
            all_save_strs = [ss + 'l' + l for l in ls for ss in all_save_strs]
            all_save_strs = [ss + metric for ss in all_save_strs]
            #
            results = {ss: np.loadtxt(ss) for ss in all_save_strs}
            resultz.append(results)
            for row, tau in enumerate(taus):
                for col, l in enumerate(ls):
                    subplot(len(taus), len(ls), len(ls)*row + col + 1)
                    ylim(ylims)
                    # 10 trials
                    lcs = [item[1] for item in results.items() if ('tau'+tau in item[0] and 'l'+l in item[0])]
                    # average over trials
                    #err_plot(np.array(lcs))
                    plot(np.mean(lcs, axis=0))




if 1: # examine BHN test set performance
    BHN_va = np.zeros((len(taus), len(ls)))
    BHN_te = np.zeros((len(taus), len(ls)))
    #
    save_str = save_strs[0] 
    all_save_strs = [save_str + 'trial' + trial for trial in trials]
    all_save_strs = [ss + 'tau' + tau for tau in taus for ss in all_save_strs]
    all_save_strs = [ss + 'l' + l for l in ls for ss in all_save_strs]
    va_results = {ss: np.loadtxt(ss + '_va_LL') for ss in all_save_strs}
    #
    save_str = save_strs[0]
    all_save_strs = [save_str + 'trial' + trial for trial in trials]
    all_save_strs = [ss + 'tau' + tau for tau in taus for ss in all_save_strs]
    all_save_strs = [ss + 'l' + l for l in ls for ss in all_save_strs]
    te_results = {ss: np.loadtxt(ss + '_te_LL') for ss in all_save_strs}

    for row, tau in enumerate(taus):
        for col, l in enumerate(ls):
            # 10 trials
            lcs = [item[1] for item in va_results.items() if ('tau'+tau in item[0] and 'l'+l in item[0])]
            lc = np.array(lcs).mean(0)
            best_epoch = np.argmax(lc)
            BHN_va[row,col] = lc[best_epoch]
            lcs = [item[1] for item in te_results.items() if ('tau'+tau in item[0] and 'l'+l in item[0])]
            lc = np.array(lcs).mean(0)
            BHN_te[row,col] = lc[best_epoch]


    print "BHN valid/test LLs"
    print np.round(BHN_va, 2)
    print np.round(BHN_te,2)
    print np.max(BHN_te)
    figure(99); title('sorted test performance (across hparams)')
    plot(sorted(BHN_te.flatten()), label='BHN')
    legend()

if 1: # examine MCD test set performance
    MCD_va = np.zeros((len(taus), len(ls)))
    MCD_te = np.zeros((len(taus), len(ls)))
    #
    save_str = save_strs[1] 
    all_save_strs = [save_str + 'trial' + trial for trial in trials]
    all_save_strs = [ss + 'tau' + tau for tau in taus for ss in all_save_strs]
    all_save_strs = [ss + 'l' + l for l in ls for ss in all_save_strs]
    va_results = {ss: np.loadtxt(ss + '_va_LL') for ss in all_save_strs}
    #
    save_str = save_strs[1]
    all_save_strs = [save_str + 'trial' + trial for trial in trials]
    all_save_strs = [ss + 'tau' + tau for tau in taus for ss in all_save_strs]
    all_save_strs = [ss + 'l' + l for l in ls for ss in all_save_strs]
    te_results = {ss: np.loadtxt(ss + '_te_LL') for ss in all_save_strs}

    for row, tau in enumerate(taus):
        for col, l in enumerate(ls):
            # 10 trials
            lcs = [item[1] for item in va_results.items() if ('tau'+tau in item[0] and 'l'+l in item[0])]
            lc = np.array(lcs).mean(0)
            best_epoch = np.argmax(lc)
            MCD_va[row,col] = lc[best_epoch]
            lcs = [item[1] for item in te_results.items() if ('tau'+tau in item[0] and 'l'+l in item[0])]
            lc = np.array(lcs).mean(0)
            MCD_te[row,col] = lc[best_epoch]


    print "MCD valid/test LLs"
    print np.round(MCD_va, 2)
    print np.round(MCD_te,2)
    print np.max(MCD_te)
    figure(99); title('sorted test performance (across hparams)')
    plot(sorted(MCD_te.flatten()), label='MCD')
    legend()







show()




















































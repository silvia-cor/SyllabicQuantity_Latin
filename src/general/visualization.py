import matplotlib.pyplot as plt
import numpy as np
import pickle

#function to make a scatter plot of macro/micro F1 for kfold and loo experiments
def f1_scatterplot(exp_path):
    try:
        with open(exp_path, 'rb') as handle:
            df = pickle.load(handle)
    except:
        print('Cannot find experiment file :(')

    tots_features = []
    macro_f1s = []
    micro_f1s = []
    for method in df:
        if method == 'True': pass
        else:
            tots_features.append(np.round(np.mean(df[method]['tot_features'])))
            macro_f1s.append(df[method]['macroF1'])
            micro_f1s.append(df[method]['microF1'])

    macro_color = 'tab:blue'
    micro_color = 'tab:orange'
    plt.scatter(tots_features, macro_f1s, label='Macro-F1')
    plt.scatter(tots_features, micro_f1s, label='Micro-F1')
    plt.plot(np.unique(tots_features), np.poly1d(np.polyfit(tots_features, macro_f1s, 1))(np.unique(tots_features)), color=macro_color)
    plt.plot(np.unique(tots_features), np.poly1d(np.polyfit(tots_features, micro_f1s, 1))(np.unique(tots_features)), color=micro_color)
    plt.xlabel('#features tot')
    plt.ylabel('F1')
    plt.legend(loc="upper right")
    plt.show()


def val_performance_visual(models, n_epochs):
    colors = ['tab:green', 'tab:red', 'tab:purple']
    for i, model in enumerate(models):
        plt.plot(models.get(model), color=colors[i])
    plt.legend(list(models.keys()))
    plt.xlabel('Epochs')
    plt.ylabel('Macro-F1')
    plt.xticks(np.arange(0, n_epochs+1, step=n_epochs/10))
    plt.savefig('../output/NN_val_F1.png')
    plt.show()


#f1_scatter('../output/kfold_results.csv')


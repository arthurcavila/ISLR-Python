import statsmodels.api as sm
import pandas as pd
import numpy as np
from itertools import combinations

def all_subsets(model, y, X, **kwargs):
    """
    model: any statsmodels regressor that takes numpy array arguments as inputs 
           and that has bic() and aic() methods
    y: endogenous variable for the model (numpy array)
    X: exogenous variables for the model (numpy array)
    **kwargs: any keyword arguments to be passed to the model
    
    Returns: DataFrame 
             number of variables, AIC, BIC, and the identification of all features present in the model
             for all possible combination of features
    """
    # count features different than constant
    nfeatures = X.shape[1]
    idx_fixer = 0
    if all(X[:,0] == 1):
        nfeatures -= 1
        idx_fixer += 1
    
    all_subsets ={}
    modelid = 0
    
    for i in range(1,nfeatures+1):
        for j in combinations(np.arange(idx_fixer, X.shape[1]),i):
            results = model(y,sm.add_constant(X[:,j]), **kwargs).fit()
            features_bool = [x in j for x in np.arange(idx_fixer, X.shape[1])]
            all_subsets[modelid] = [i, results.aic, results.bic] + features_bool
            nn = ['N_var', 'AIC', 'BIC']+['Feature'+str(i) for i in range(idx_fixer, X.shape[1])]
            modelid += 1
    return(pd.DataFrame(all_subsets, index = nn).transpose())

def report_best_all(model, y, X, n, **kwargs):
    '''
    model: any statsmodels regressor that takes numpy array arguments as inputs 
           and that has bic() and aic() methods
    y: endogenous variable for the model (numpy array)
    X: exogenous variables for the model (numpy array)
    n: number of best specifications to be printed using both 'AIC' and 'BIC' criteria
    **kwargs: any keyword arguments to be passed to the model
    
    Returns: DataFrame 
             number of variables, AIC, BIC, and the identification of all features present in the model
             for all possible combination of features
    '''
    all_models = all_subsets(model, y, X, **kwargs)
    print(all_models.sort_values('AIC').iloc[0:(n+1),:])
    print(all_models.sort_values('BIC').iloc[0:(n+1),:])
    return(all_models)

def add1(model, y, X, active_features=None, crit='AIC', **kwargs):
    '''
    model: any statsmodels regressor that takes numpy array arguments as inputs 
           and that has bic() and aic() methods
    y: endogenous variable for the model (numpy array)
    X: exogenous variables for the model, both active and candidates to be added (numpy array)
    active_features: numeric list indicating the current active features of X
    crit: 'AIC' or 'BIC', will sort the output.
    **kwargs: any keyword arguments to be passed to the model
    
    Returns: DataFrame 
             number of variables, AIC, BIC, and the identification of all features present in the model
             and all possible combinations of one additional feature from X sorted by crit
    '''
    if active_features == None:
        active_features = [0]
    
    newX = X[:,active_features]
    nfeatures = newX.shape[1]
    if X.shape[1] == nfeatures:
        print('No features to add.')
        return None
    idx_fixer = 0
    if all(newX[:,0] == 1):
        nfeatures -= 1
    #    idx_fixer += 1
    if all(X[:,0] == 1):
        idx_fixer += 1
    candidates = [i for i in range(X.shape[1]) if (i not in active_features) & (i >= idx_fixer)]
    
    all_subsets ={}
    modelid = 0
    results = model(y,sm.add_constant(newX), **kwargs).fit()
    features_bool = [x in active_features for x in np.arange(idx_fixer, X.shape[1])]
    all_subsets[modelid] = [nfeatures, results.aic, results.bic] + features_bool
    nn = ['N_var', 'AIC', 'BIC']+['Feature'+str(i+1-idx_fixer) for i in range(idx_fixer, X.shape[1])]
    
    for i in candidates:
        j = np.sort(np.append(active_features, i))
        modelid += 1
        results = model(y,sm.add_constant(X[:,j]), **kwargs).fit()
        features_bool = [x in j for x in np.arange(idx_fixer, X.shape[1])]
        all_subsets[modelid] = [nfeatures+1, results.aic, results.bic] + features_bool
        nn = ['N_var', 'AIC', 'BIC']+['Feature'+str(i+1-idx_fixer) for i in range(idx_fixer, X.shape[1])]
    
    summary = pd.DataFrame(all_subsets, index = nn).transpose().sort_values(crit)
    return summary

def drop1(model, y, X, active_features=None, crit='AIC', **kwargs):
    '''
    model: any statsmodels regressor that takes numpy array arguments as inputs 
           and that has bic() and aic() methods
    y: endogenous variable for the model (numpy array)
    X: exogenous variables for the model, both active and candidates to be added (numpy array)
    active_features: numeric list indicating the current active features of X
    crit: 'AIC' or 'BIC', will sort the output.
    **kwargs: any keyword arguments to be passed to the model
    
    Returns: DataFrame 
             number of variables, AIC, BIC, and the identification of all active features present in the model
             and all possible combinations of one dropped feature sorted by crit
    '''
    if active_features == None:
        active_features = [i for i in range(X.shape[1])]
    
    newX = X[:,active_features]
    nfeatures = newX.shape[1]
    idx_fixer = 0
    
    if all(newX[:,0] == 1):
        nfeatures -= 1
    #    idx_fixer += 1
    if all(X[:,0] == 1):
        idx_fixer += 1
    if X.shape[1] <= idx_fixer:
        print('Not enough features to drop.')
        return None
    
    candidates = [i for i in range(X.shape[1]) if (i in active_features) & (i >= idx_fixer)]
    
    all_subsets ={}
    modelid = 0
    results = model(y,sm.add_constant(newX), **kwargs).fit()
    features_bool = [x in active_features for x in np.arange(idx_fixer, X.shape[1])]
    all_subsets[modelid] = [nfeatures, results.aic, results.bic] + features_bool
    nn = ['N_var', 'AIC', 'BIC']+['Feature'+str(i+1-idx_fixer) for i in range(idx_fixer, X.shape[1])]
    
    for j in combinations(candidates,nfeatures-1):
        modelid += 1
        results = model(y,sm.add_constant(X[:,j]), **kwargs).fit()
        features_bool = [x in j for x in np.arange(idx_fixer, X.shape[1])]
        all_subsets[modelid] = [nfeatures-1, results.aic, results.bic] + features_bool
        nn = ['N_var', 'AIC', 'BIC']+['Feature'+str(i+1-idx_fixer) for i in range(idx_fixer, X.shape[1])]
    summary = pd.DataFrame(all_subsets, index = nn).transpose().sort_values(crit)
    return(summary)

def step_back(model, y, X, active_features=None, crit='AIC', **kwargs):
    '''
    model: any statsmodels regressor that takes numpy array arguments as inputs 
           and that has bic() and aic() methods
    y: endogenous variable for the model (numpy array)
    X: exogenous variables for the model, both active and candidates to be added (numpy array)
    active_features: numeric list indicating the current active features of X
    crit: 'AIC' or 'BIC', will sort the output.
    **kwargs: any keyword arguments to be passed to the model
    
    Returns: DataFrame 
             number of variables, AIC, BIC, and the identification of all features present in the model
             and the best deletion of feature iterated until only one feature is left based on crit
    '''
    if active_features == None:
        active_features = [i for i in range(X.shape[1])]
    
    newX = X[:,active_features]
    nfeatures = newX.shape[1]
    idx_fixer = 0
    
    if all(newX[:,0] == 1):
        nfeatures -= 1
    if all(X[:,0] == 1):
        idx_fixer += 1
        
    candidates = [i for i in range(X.shape[1]) if (i in active_features) & (i >= idx_fixer)]

    all_subsets ={}
    modelid = 0
    results = model(y,sm.add_constant(newX), **kwargs).fit()
    features_bool = [x in active_features for x in np.arange(idx_fixer, X.shape[1])]
    all_subsets[modelid] = [nfeatures, results.aic, results.bic] + features_bool
    nn = ['N_var', 'AIC', 'BIC']+['Feature'+str(i+1-idx_fixer) for i in range(idx_fixer, X.shape[1])]    
    
    summary = pd.DataFrame(all_subsets, index = nn).transpose().sort_values(crit)
    
    while nfeatures > 1:
        new_summary = drop1(model,y,X,active_features,crit,**kwargs)
        
        if (new_summary.iloc[0,0] == nfeatures):
            if (new_summary.shape[0]==1):
                break
            best = new_summary.iloc[1,:]
        else:
            best = new_summary.iloc[0,:]
        
        summary = summary.append(best)
        active_features = [i for i in candidates if best[i+3-idx_fixer]]
        nfeatures -=1
        
    return summary.sort_values(crit)

def step_forward(model, y, X, active_features=None, crit='AIC', nbest = 1, **kwargs):
    '''
    model: any statsmodels regressor that takes numpy array arguments as inputs 
           and that has bic() and aic() methods
    y: endogenous variable for the model (numpy array)
    X: exogenous variables for the model, both active and candidates to be added (numpy array)
    active_features: numeric list indicating the current active features of X
    crit: 'AIC' or 'BIC', will sort the output.
    **kwargs: any keyword arguments to be passed to the model
    
    Returns: DataFrame 
             number of variables, AIC, BIC, and the identification of all features present in the model
             and the best addition of one feature iterated until only all features are present based on crit
    '''
    if active_features == None:
        active_features = [0]
    
    newX = X[:,active_features]
    nfeatures = newX.shape[1]
    idx_fixer = 0
    
    if all(newX[:,0] == 1):
        nfeatures -= 1
    if all(X[:,0] == 1):
        idx_fixer += 1
        
    candidates = [i for i in range(X.shape[1]) if (i not in active_features) & (i >= idx_fixer)]

    all_subsets ={}
    modelid = 0
    results = model(y,sm.add_constant(newX), **kwargs).fit()
    features_bool = [x in active_features for x in np.arange(idx_fixer, X.shape[1])]
    all_subsets[modelid] = [nfeatures, results.aic, results.bic] + features_bool
    nn = ['N_var', 'AIC', 'BIC']+['Feature'+str(i+1-idx_fixer) for i in range(idx_fixer, X.shape[1])]    
    
    summary = pd.DataFrame(all_subsets, index = nn).transpose().sort_values(crit)
    
    while (nfeatures <= len(candidates)-idx_fixer):
        new_summary = add1(model,y,X,active_features,crit,**kwargs)
        if (new_summary.iloc[0,0] == nfeatures):
            if (new_summary.shape[0]==1):
                break
            best = new_summary.iloc[1,:]
        else:
            best = new_summary.iloc[0,:]
        summary = summary.append(best)
        j = [i for i in candidates if best[i+3-idx_fixer]]
        active_features = [x for x in np.unique(np.append(active_features, j))]
        nfeatures +=1
        
    return summary.sort_values(crit)

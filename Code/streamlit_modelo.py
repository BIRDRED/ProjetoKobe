import streamlit as st
import pandas
import numpy as np
from sklearn import model_selection, tree, ensemble, metrics, feature_selection
import joblib

fname = '../Data/Raw/kobe_dataset.csv'
savefile = '../Data/model_kobe.pkl'

############################################ LEITURA DOS DADOS
print('=> Leitura dos dados')
df_kobe = pandas.read_csv(fname,sep=',')
kobe_target_col = 'shot_made_flag'
kobe_label_map = df_kobe[['loc_x', 'loc_y','lat','lon','minutes_remaining','period','playoffs','shot_distance','shot_made_flag','shot_type']].dropna()
df_kobe = kobe_label_map
print(df_kobe.head())
print(df_kobe['shot_type'].value_counts())
print(df_kobe['shot_type'].unique())
############################################ TREINO/TESTE E VALIDACAO
results = {}
print (df_kobe.columns)
for kobe_type in df_kobe['shot_type'].unique():
    print('=> Training for kobe:', kobe_type)
    print('\tSeparacao treino/teste')
    kobe = df_kobe.loc[df_kobe['shot_type'] == kobe_type].copy()
    Y = kobe[kobe_target_col]
    X = kobe.drop([kobe_target_col,'shot_type','lat','lon'], axis=1)
    ml_feature = list(X.columns)
    # train/test
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(X, Y, test_size=0.2,train_size=0.8)
    cvfold = model_selection.StratifiedKFold(n_splits = 10, random_state = 0, shuffle=True)
    print('\t\tTreino:', xtrain.shape[0])
    print('\t\tTeste :', xtest.shape[0])

    ############################################ GRID-SEARCH VALIDACAO CRUZADA
    print('\tTreinamento e hiperparametros')
    param_grid = { 'ccp_alpha':[0.0], 'class_weight':[None], 'criterion':['gini'],
                       'max_depth':[None], 'max_features':[None], 'max_leaf_nodes':[None],
                      'min_impurity_decrease':[0.0], 'min_impurity_split':[None],
                      'min_samples_leaf':[1], 'min_samples_split':[2],
                      'min_weight_fraction_leaf':[0.0], 'presort':['deprecated'],
                      'random_state':[6651], 'splitter':['best']
                 }
    selector = feature_selection.RFE(tree.DecisionTreeClassifier(),
                                     n_features_to_select = 4)
    selector.fit(xtrain, ytrain)
    ml_feature = np.array(ml_feature)[selector.support_]
    
    model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(),
                                         param_grid = param_grid,
                                         scoring = 'f1',
                                         refit = True,
                                         cv = cvfold,
                                         return_train_score=True
                                        )
    model.fit(xtrain[ml_feature], ytrain)

    ############################################ AVALIACAO GRUPO DE TESTE
    print('\tAvaliação do modelo')
    print(xtrain[ml_feature])
    threshold = 0.5
    xtrain.loc[:, 'probabilidade'] = model.predict_proba(xtrain[ml_feature])[:,1]
    xtrain.loc[:, 'classificacao'] = (xtrain.loc[:, 'probabilidade'] > threshold).astype(int)
    xtrain.loc[:, 'categoria'] = 'treino'

    xtest.loc[:, 'probabilidade']  = model.predict_proba(xtest[ml_feature])[:,1]
    xtest.loc[:, 'classificacao'] = (xtest.loc[:, 'probabilidade'] > threshold).astype(int)
    xtest.loc[:, 'categoria'] = 'teste'

    kobe = pandas.concat((xtrain, xtest))
    kobe[kobe_target_col] = pandas.concat((ytrain, ytest))
    kobe['target_label'] =  ['Acertou' if t else 'Errou'
                            for t in kobe[kobe_target_col]]
    
    print('\t\tAcurácia treino:', metrics.accuracy_score(ytrain, xtrain['classificacao']))
    print('\t\tAcurácia teste :', metrics.accuracy_score(ytest, xtest['classificacao']))

    ############################################ RETREINAMENTO DADOS COMPLETOS
    print('\tRetreinamento com histórico completo')
    model = model.best_estimator_
    model = model.fit(X[ml_feature], Y)
    
    ############################################ DADOS PARA EXPORTACAO
    results[kobe_type] = {
        'model': model,
        'data': kobe, 
        'features': ml_feature,
        'target_col': kobe_target_col,
        'threshold': threshold
    }

############################################ EXPORTACAO RESULTADOS
print('=> Exportacao dos resultados')

joblib.dump(results, savefile, compress=9)
print('\tModelo salvo em', savefile)


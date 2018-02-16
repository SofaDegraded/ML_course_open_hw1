# подгружаем все нужные пакеты
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns


# настройка внешнего вида графиков в seaborn
sns.set_context(
    "notebook", 
    font_scale = 1.,       
    rc = { 
        "figure.figsize" : (15, 9), 
        "axes.titlesize" : 3
    }
)
train = pd.read_csv('mlbootcamp5_train.csv', sep=';',
                    index_col='id')
print('Размер датасета: ', train.shape)
#train.head()
#train_uniques = pd.melt(frame=train, value_vars=['gender','cholesterol', 
#                                                 'gluc', 'smoke', 'alco', 
#                                                 'active', 'cardio'])
#train_uniques = pd.DataFrame(train_uniques.groupby(['variable', 
#                                                    'value'])['value'].count()) \
#    .sort_index(level=[0, 1]) \
#    .rename(columns={'value': 'count'}) \
#    .reset_index()
    
#sns.factorplot(x='variable', y='count', hue='value', 
#               data=train_uniques, kind='bar', size=12)
#train_uniques = pd.melt(frame=train, value_vars=['gender','cholesterol', 
#                                                 'gluc', 'smoke', 'alco', 
#                                                 'active'], 
#                        id_vars=['cardio'])
#train_uniques = pd.DataFrame(train_uniques.groupby(['variable', 'value', 
#                                                    'cardio'])['value'].count()) \
#    .sort_index(level=[0, 1]) \
#    .rename(columns={'value': 'count'}) \
#    .reset_index()
    
#sns.factorplot(x='variable', y='count', hue='value', 
#               col='cardio', data=train_uniques, kind='bar', size=9)
#plt.show()
#for c in train.columns:
#    n = train[c].nunique()
#    print(c)
    
#    if n <= 3:
#        print(n, sorted(train[c].value_counts().to_dict().items()))
#    else:
#        print(n)
#    print(10 * '-')
#cols = ['gluc', 'cholesterol', 'weight', 'alco', 'smoke', 'gender', 'height']
#corr_p = train[cols].corr(method='pearson')
#corr = train.corr(method='pearson')
#corr_s = train.corr(method='spearman')
#print(corr)
#print(corr_s)
#mask = np.zeros_like(corr)
#mask[np.triu_indices_from(mask)] = True
#with sns.axes_style("white"):
#    ax = sns.heatmap(corr, mask=mask, vmax=.2, square=True)
#sns.heatmap(corr)

#sns.violinplot(x='gender', y='height', hue='gender', data=train)
#sns.kdeplot(train[train['gender'] == 1]['height'])
#sns.kdeplot(train[train['gender'] == 2]['height'])
#train = train[(train['ap_lo']>0) & (train['ap_hi']>0) & (train['ap_lo']<250) & (train['ap_hi']<250) & (train['ap_hi'] > train['ap_lo'])]
#train['ap_lo']=train['ap_lo'].apply(np.log1p)
#train['ap_hi']=train['ap_hi'].apply(np.log1p)
#g = sns.jointplot('ap_lo', 'ap_hi', data=train)
#"""Сетка"""
#g.ax_joint.grid(True) 

#"""Преобразуем логарифмические значения на шкалах в реальные"""
#g.ax_joint.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(round(int(np.exp(x))))))
#g.ax_joint.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: str(round(int(np.exp(x))))))
train['age_years'] = (train['age'] // 365.25).astype(int)
ax = sns.countplot(x="age_years", hue="cardio", data=train)
plt.show()
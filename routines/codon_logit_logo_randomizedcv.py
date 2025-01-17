import os
import sys
sys.path.append("/home/pdavis/sarissa/utils")
from Bio import SeqIO
import pandas as pd
import numpy as np
from scipy.stats import loguniform
from cv import GroupBasedCrossValidator
from feature_extraction import CodonFrequencies
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression # Replace this with your estimator
from bson import ObjectId
import argparse
import joblib
import yaml


parser = argparse.ArgumentParser(description="Run machine learning model training based on a YAML config file.")
parser.add_argument("yaml_file", help="Path to the YAML configuration file.")
args = parser.parse_args()

with open(args.yaml_file, 'r') as f:
        config = yaml.safe_load(f)

seqs = {s.id:s for s in SeqIO.parse(config['seqs'],'fasta')}
meta = pd.read_table(config['meta'])
meta = meta[meta['accession'].isin(set(seqs.keys()))].copy()
labels_dict = dict(meta[['accession','label']].values)
groups_dict = dict(meta[['accession','groups']].values)
out_prefix = str(ObjectId())

test_set = set(meta[meta['groups'].isin(config['test_groups'])]['accession'].values)

if config['resample']:
    print("Resampling {} times.".format(config['resample']))
    resample_index = np.zeros(config['resample'])
    model_groups = meta.groupby('groups')
    groups = meta['groups'].unique()
    pos_groups = [g for g in groups if '1_' in g]
    neg_groups = [g for g in groups if '0_' in g]
    for i in range(config['resample']):
        class_pick = np.random.randint(2)
        if class_pick == 1:
            selected = np.random.choice(pos_groups)
        else:
            selected = np.random.choice(neg_groups)
        resample_index[i] = model_groups.get_group(selected).sample(1).index[0]
    train_accessions = [a for a in meta.loc[resample_index]['accession'] if a not in test_set]
    test_accessions = [a for a in meta.loc[resample_index]['accession'] if a in test_set]
else:
    train_accessions = [a for a in meta['accession'] if a not in test_set]
    test_accessions = [a for a in meta['accession'] if a in test_set]

X = [str(seqs[a].seq) for a in train_accessions]
y = np.array([labels_dict[a] for a in train_accessions])
groups = np.array([groups_dict[a] for a in train_accessions])
X_test =  [str(seqs[a].seq) for a in test_accessions]
y_test = np.array([labels_dict[a] for a in test_accessions])

pipe_steps = []

pipe_steps.append(('codons',CodonFrequencies()))
pipe_steps.append(('std_scaler',StandardScaler()))
pipe_steps.append(('logit',LogisticRegression(penalty='l1',solver='liblinear',max_iter=1000,fit_intercept=False)))

pipeline = Pipeline(steps=pipe_steps,memory=config['cache_path'])

cv = GroupBasedCrossValidator(n_splits=config['n_splits'])
param_grid = {key:loguniform(config['CV_params'][key]['a'],config['CV_params'][key]['b']) for key in config['CV_params'].keys()}

grid_search = RandomizedSearchCV(pipeline,param_grid,n_iter=config['n_iter'],scoring='neg_brier_score',cv=cv,n_jobs=-1)
print("Fitting model.")
grid_search.fit(X,y,groups=groups)
print("Training complete.")
predict = grid_search.predict(X)
a_score = accuracy_score(y,predict)
p_score = precision_score(y,predict)
r_score = recall_score(y,predict)


misclass = meta[meta['accession'].isin(set(pd.Series(train_accessions)[y!=predict]))]

print("Trained model score (accuracy):",a_score)

test_predict = grid_search.predict(X_test)
a_test = accuracy_score(y_test,test_predict)
p_test = precision_score(y_test,test_predict)
r_test = recall_score(y_test,test_predict)

misclass_test = meta[meta['accession'].isin(set(pd.Series(test_accessions)[y_test!=test_predict]))]

print("Test set score (accuracy):",a_test)
cv_results = pd.DataFrame(grid_search.cv_results_).sort_values('mean_test_score')
print("Max mean CV score: {}".format(cv_results['mean_test_score'].max()))
print("CV results:")
print(cv_results)
feature_vector = grid_search.best_estimator_.named_steps['codons'].features
model_coef = pd.Series(dict(zip(feature_vector[np.where(grid_search.best_estimator_.named_steps['logit'].coef_[0] != 0)],grid_search.best_estimator_.named_steps['logit'].coef_[0][np.where(grid_search.best_estimator_.named_steps['logit'].coef_[0] != 0)])))
#Write output
os.chdir(config['output_dir'])
summary = {'test_groups': config['test_groups'], 'training_accuracy': a_score, 'training_recall': r_score, 'training_precision': p_score, 'test_accuracy':a_test,
        'test_recall':r_test,'test_precision':p_test,'model_type':config['model_type'],'feature_type':config['feature_type']}
print("Model predictors with coefficients greater than 0:")
print(model_coef)
model_coef.sort_values().to_csv(out_prefix+"_model_coefficients.tsv",sep='\t')
cv_results.to_csv(out_prefix+"_cv_results.tsv",sep='\t')
pd.Series(summary).to_csv(out_prefix+"_summary.tsv",sep='\t')
misclass.to_csv(out_prefix+"_training_misclassified.tsv",sep='\t',index=False)
misclass_test.to_csv(out_prefix+"_test_misclassfied.tsv",sep='\t',index=False)
#output serialized classifier object for persistence
joblib.dump(grid_search,out_prefix+"_CLF.joblib")


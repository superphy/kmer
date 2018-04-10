import yaml
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
from kmerprediction.complete_kmer_counter import get_counts, get_kmer_names
from kmerprediction.get_data import get_kmer_us_uk_split
from kmerprediction.utils import parse_metadata
from kmerprediction import constants

complete_path = ('/home/rylan/miniconda3/envs/kmer/lib/python3.6/site-packages/' +
                 'kmerprediction/kmer_data/')
new_complete_path = '/home/rylan/Data/lupolova_data/complete_database/'
complete_dbs = {7: complete_path + 'complete_7-mer_DB/',
                15: complete_path + 'complete_15-mer_DB/',
                31: complete_path + 'complete_31-mer_DB/',
                9: new_complete_path + 'complete_9-mer_DB/',
                11: new_complete_path + 'complete_11-mer_DB/'}

base_path = '/home/rylan/Data/lupolova_data/database/'
output_dbs = {7: base_path + '7-mer_output_DB/',
              15: base_path + '15-mer_output_DB/',
              31: base_path + '31-mer_output_DB/',
              9: base_path + '9-mer_output_DB/',
              11: base_path + '11-mer_output_DB/'}

def f_test_scores(data, target, indices):
    f_values = []
    p_values = []
    F, pval = f_classif(data, target)
    for x, i in enumerate(indices):
        f_values[x] = F[i]
        p_values[x] = pval[i]
    return f_values, p_values

def fdr_scores(data, target, indices):
    scores = []
    p_values = []
    sel = SelectFdr(score_func=f_classif, alpha=1e-5)
    sel.fit(data, target)
    for x, i in enumerate(indices):
        scores[x] = sel.scores_[i]
        p_values[x] = sel.pvalues_[i]
    return scores, p_values

def avg_counts(files, database, indices):
    data = get_counts(files, database)
    for x, i in enumerate(indices):
        means[x] = np.mean(data[:,index])
        stds[x] = np.mean(data[:,index])
    return means, stds

def model_scores(yaml_files):
    output = []
    for yf in yaml_files:
        filename = yf.split('/')[-1]
        filename = filename.split('_')

        selection = filename[-1].replace('.yml', '')
        dataset = filename[-3]
        model = ' '.join(filename[:-5]).title()

        with open(yf, 'r') as f:
            data = yaml.load(f)
            data = data['output']['important_features']
        feature_scores = {}
        for d in data:
            ranked_features = sorted(d, reverse=True, key=lambda k: d[k])
            for index, value in enumerate(ranked_features):
                score = (1/(2**index))/len(data)
                if value not in feature_scores:
                    feature_scores[value] = 0.0
                feature_scores[value] += score

        output.append({'model': model, 'sel': selection,
                       'dataset': dataset, 'scores': feature_scores})
    return output

def main():
    output = pd.DataFrame()
    scores = model_scores(snakemake.input)
    k = int(snakemake.wildcards['k'])

    kmers = {}
    for d in scores:
        kmers.update(d['scores'])
    kmers = kmers.keys()
    output['Kmer'] = kmers

    for d in scores:
        header = '{} {} {} Scores'.format(d['model'], d['sel'], d['dataset'])
        column = [d['scores'][k] if k in d['scores'] else 0.0 for k in kmers]
        output[header] = column

    kmer_names = get_kmer_names(output_dbs[k])
    kmer_indices = [np.where(kmer_names==k)[0][0] for k in kmers]

    args = {'kmer_kwargs': {'k': k, 'output_db': output_dbs[k]},
            'database': complete_dbs[k]}
    data, a, b, c = get_kmer_us_uk_split(**args)
    uk_data, uk_target, us_data, uk_target = data

    uk_fvals, uk_ftest_pvals = f_test_scores(uk_data, uk_target, kmer_indices)
    us_fvals, us_ftest_pvals = f_test_scores(us_data, us_target, kmer_indices)

    uk_fdr_scores, uk_fdr_pvals = fdr_scores(uk_data, uk_target, kmer_indices)
    us_fdr_scores, us_fdr_pvals = fdr_scores(us_data, us_target, kmer_indices)

    output['UK F-test Score'] = uk_fvals
    output['UK F-test pvalue'] = uk_ftest_pvals
    output['US F-test Score'] = us_fvals
    output['US F-test pvalue'] = us_f_test_pvals
    output['UK FDR Score'] = uk_fdr_scores
    output['UK FDR pvalue'] = uk_fdr_pvals
    output['US FDR Score'] = us_fdr_scores
    output['US FDR pvalue'] = us_fdr_pvals

    args = {'prefix': constants.ECOLI, 'suffix': '.fasta',
            'validate': True}
    uk_files, uk_target, us_files, us_target = parse_metadata(**args)
    uk_avg_counts, uk_std_counts = avg_counts(uk_files, output_dbs[k], kmer_indices)
    us_avg_counts, us_std_counts = avg_counts(us_files, output_dbs[k], kmer_indices)

    output['UK avg Count'] = uk_avg_counts
    output['UK std Count'] = uk_std_counts
    output['US avg Count'] = us_avg_counts
    output['US std Count'] = us_std_counts

    output.to_csv(snakemake.output[0], index=False)

if __name__ == "__main__":
    main()




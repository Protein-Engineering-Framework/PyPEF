
import os
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error

file_path = os.path.dirname(__file__)

# ["fold_random_5", "fold_modulo_5", "fold_contiguous_5", "fold_rand_multiples"]
target_folds = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
llm = 'prosst+esm'


def get_target_folder(split_method):
    return os.path.join(
        file_path, 'model_scores', 'supervised_substitutions', 
        split_method, 'pypef_hybrid', llm
    )


def get_df_fold_performances(df):
    fold_metrics = {
        'Mean_Spearman_corr': np.nan,
        'Mean_Pearson_corr': np.nan,
        'Mean_MSE': np.nan,
        'Folds': {},
        'Across_folds_Spearman_corr': np.nan,
        'Across_folds_Pearson_corr': np.nan,
        'Across_folds_MSE': np.nan
    }
    spears, pears, mses = [], [], []
    for fold in sorted(df['fold'].unique()):
        fold_data = df[df['fold'] == fold]
        y_true = fold_data['y']
        y_pred = fold_data['y_pred']

        spear_p = spearmanr(y_true, y_pred)[0]
        pearson_r = pearsonr(y_true, y_pred)[0]
        mse = mean_squared_error(y_true, y_pred)
        spears.append(spear_p)
        pears.append(pearson_r)
        mses.append(mse)

        fold_metrics['Folds'][fold] = {
            'Spearman_corr': spear_p,
            'Pearson_corr': pearson_r,
            'MSE': mse  # Check: B2L11_HUMAN_Dutta_2010_binding-Mcl-1.csv
        }
    fold_metrics['Mean_Spearman_corr'] = np.mean(spears)
    fold_metrics['Mean_Pearson_corr'] = np.mean(pears)
    fold_metrics['Mean_MSE'] = np.mean(mses)
    fold_metrics['Mean_Spearman_StdDev'] = np.std(spears, ddof=1)
    fold_metrics['Mean_Pearson_StdDev'] = np.std(pears, ddof=1)
    fold_metrics['Mean_MSE_StdDev'] = np.std(mses, ddof=1)
    fold_metrics['Across_folds_Spearman_corr'] = spearmanr(df['y'], df['y_pred'])[0]
    fold_metrics['Across_folds_Pearson_corr'] = pearsonr(df['y'], df['y_pred'])[0]
    fold_metrics['Across_folds_MSE'] = mean_squared_error(df['y'], df['y_pred'])
    return fold_metrics


results_across_split_technique_folds = defaultdict(list)
for tf in target_folds:
    print('\n' + '~' * 60 +  '\n' + f'~~~ {tf} ~~~\n' +'~' * 60)
    target_folder = get_target_folder(tf)
    all_spears, all_pears, all_mses = [], [], []
    all_across_spears, all_across_pears, all_across_mses = [], [], []
    for result_csv in os.listdir(target_folder):
        res = os.path.join(target_folder, result_csv)
        df = pd.read_csv(res)
        fold_metrics = get_df_fold_performances(df)
        print(f"CSV: {result_csv}\n"
              f"  Spearman corr={fold_metrics['Mean_Spearman_corr']:.3f}  (+-{fold_metrics['Mean_Spearman_StdDev']:.3f})  "
              f"|  no folds={fold_metrics['Across_folds_Spearman_corr']:.3f}\n"
              f"  Pearson corr={fold_metrics['Mean_Pearson_corr']:.3f}  (+-{fold_metrics['Mean_Pearson_StdDev']:.3f}) "
              f"|  no folds={fold_metrics['Across_folds_Pearson_corr']:.3f}\n"
              f"  MSE={fold_metrics['Mean_MSE']:.3f}  (+-{fold_metrics['Mean_MSE_StdDev']:.3f}) "
              f"|  no folds={fold_metrics['Across_folds_MSE']:.3f}\n\n"
        )
        all_spears.append(fold_metrics['Mean_Spearman_corr'])
        all_pears.append(fold_metrics['Mean_Pearson_corr'])
        all_mses.append(fold_metrics['Mean_MSE'])
        all_across_spears.append(fold_metrics['Across_folds_Spearman_corr'])
        all_across_pears.append(fold_metrics['Across_folds_Pearson_corr'])
        all_across_mses.append(fold_metrics['Across_folds_MSE'])
        results_across_split_technique_folds[result_csv].append(fold_metrics['Across_folds_Spearman_corr'])

    print('-' * 60 + '\n' +
          f"Mean Spearman corr. across all {len(all_spears)} datasets={np.mean(all_spears):.3f} | no folds={np.mean(all_across_spears):.3f}\n"
          f"Mean Pearson corr. across all {len(all_pears)} datasets={np.mean(all_pears):.3f} | no folds={np.mean(all_across_pears):.3f}\n"
          f"Mean MSE across all {len(all_mses)} datasets={np.mean(all_mses):.3f} | no folds={np.mean(all_across_mses):.3f}\n"
    )

all_means = []
print(f'\n\n--- Total performance across split techniques ---\n{target_folds}\n\n')
for k, v in results_across_split_technique_folds.items():
    mean = np.mean(v)
    print(k, f"{mean:.3f}")
    all_means.append(mean)
print('-' * 60 + f'\nMean={np.mean(all_means):.3f}')



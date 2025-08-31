
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

file_path = os.path.dirname(__file__)

# ["fold_random_5", "fold_modulo_5", "fold_contiguous_5", "fold_rand_multiples"]
target_folds = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5", "fold_rand_multiples"]
llm = 'prosst'


def get_target_folder(split_method):
    return os.path.join(
        file_path, 'model_scores', 'supervised_substitutions', 
        split_method, 'pypef_hybrid', llm
    )


def get_df_fold_performances(df):
    fold_metrics = {
        'Mean_Spearman_corr': np.nan,
        'Mean_Pearson_corr': np.nan,
        'Folds': {}
    }
    spears, pears = [], []
    for fold in sorted(df['fold'].unique()):
        fold_data = df[df['fold'] == fold]
        y_true = fold_data['y']
        y_pred = fold_data['y_pred']

        spear_p = spearmanr(y_true, y_pred)[0]
        pearson_r = pearsonr(y_true, y_pred)[0]
        spears.append(spear_p)
        pears.append(pearson_r)

        fold_metrics['Folds'][fold] = {
            'Spearman_corr': spear_p,
            'Pearson_corr': pearson_r
        }
    fold_metrics['Mean_Spearman_corr'] = np.mean(spears)
    fold_metrics['Mean_Pearson_corr'] = np.mean(pears)
    fold_metrics['Mean_Spearman_StdDev'] = np.std(spears, ddof=1)
    fold_metrics['Mean_Pearson_StdDev'] = np.std(pears, ddof=1)    
    return fold_metrics



for tf in target_folds:
    print(f'~~~ {tf} ~~~')
    target_folder = get_target_folder(tf)
    all_spears, all_pears = [], []
    for result_csv in os.listdir(target_folder):
        res = os.path.join(target_folder, result_csv)
        df = pd.read_csv(res)
        fold_metrics = get_df_fold_performances(df)
        print(f"CSV: {result_csv}")
        print(f"  Spearman corr={fold_metrics['Mean_Spearman_corr']:.3f}  (+-{fold_metrics['Mean_Spearman_StdDev']:.3f})")
        print(f"  Pearson corr={fold_metrics['Mean_Pearson_corr']:.3f}  (+-{fold_metrics['Mean_Pearson_StdDev']:.3f})")
        all_spears.append(fold_metrics['Mean_Spearman_corr'])
        all_pears.append(fold_metrics['Mean_Pearson_corr'])
    print('-' * 60 + '\n' +
          f"Mean Spearman corr. across all datasets={np.mean(all_spears):.3f}\n" +
          f"Mean Pearson corr. across all datasets={np.mean(all_pears):.3f}\n\n"
    )



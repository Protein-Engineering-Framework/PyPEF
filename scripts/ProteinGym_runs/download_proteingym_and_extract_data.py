import os
import urllib.request
import zipfile
import pandas as pd
import json
# To use unverified ssl you can add this to your code, taken from:
# https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context



def download_proteingym_data():
    url = 'https://marks.hms.harvard.edu/proteingym/DMS_substitutions.csv'
    print(f'Getting {url}...')
    urllib.request.urlretrieve(url, os.path.join(os.path.dirname(__file__), '_Description_DMS_substitutions_data.csv'))

    url = 'https://marks.hms.harvard.edu/proteingym/DMS_ProteinGym_substitutions.zip'
    print(f'Getting {url}...')
    urllib.request.urlretrieve(url, os.path.join(os.path.dirname(__file__), 'DMS_ProteinGym_substitutions.zip'))
    with zipfile.ZipFile(os.path.join(os.path.dirname(__file__), 'DMS_ProteinGym_substitutions.zip'), "r") as zip_ref:
        zip_ref.extractall(os.path.join(os.path.dirname(__file__), 'DMS_ProteinGym_substitutions'))
    os.remove(os.path.join(os.path.dirname(__file__), 'DMS_ProteinGym_substitutions.zip'))

    url = 'https://marks.hms.harvard.edu/proteingym/DMS_msa_files.zip'
    print(f'Getting {url}...')
    urllib.request.urlretrieve(url, os.path.join(os.path.dirname(__file__), 'DMS_msa_files.zip'))
    with zipfile.ZipFile(os.path.join(os.path.dirname(__file__), 'DMS_msa_files.zip'), "r") as zip_ref:
        zip_ref.extractall(os.path.join(os.path.dirname(__file__), 'DMS_msa_files'))
    os.remove(os.path.join(os.path.dirname(__file__), 'DMS_msa_files.zip'))

    url = 'https://marks.hms.harvard.edu/proteingym/ProteinGym_AF2_structures.zip'
    print(f'Getting {url}...')
    urllib.request.urlretrieve(url, os.path.join(os.path.dirname(__file__), 'ProteinGym_AF2_structures.zip'))
    with zipfile.ZipFile(os.path.join(os.path.dirname(__file__), 'ProteinGym_AF2_structures.zip'), "r") as zip_ref:
        zip_ref.extractall(os.path.join(os.path.dirname(__file__), 'ProteinGym_AF2_structures'))
    os.remove(os.path.join(os.path.dirname(__file__), 'ProteinGym_AF2_structures.zip'))


def get_single_or_multi_point_mut_data(csv_description_path, datasets_path=None, msas_path=None, pdbs_path=None, single: bool = True):
    """
    Get ProteinGym data, here only the single or multi-point mutant data (all data for 
    that target dataset having single- or multi-point mutated variants available).
    Reads the dataset description/overview CSV to search for available data in 
    the 'DMS_ProteinGym_substitutions' sub-directory.
    """
    if single:
        type_str = 'single'
    else:
        type_str = 'multi'
    file_dirname = os.path.abspath(os.path.dirname(__file__))
    if datasets_path is None:
        datasets_path = os.path.join(file_dirname, 'DMS_ProteinGym_substitutions')
    if msas_path is None:
        msas_path = os.path.join(file_dirname, 'DMS_msa_files')  # used to be DMS_msa_files/MSA_files/DMS
    msas = os.listdir(msas_path)
    if pdbs_path is None:
        pdbs_path = os.path.join(file_dirname, 'ProteinGym_AF2_structures')
    pdbs = os.listdir(pdbs_path)
    description_df = pd.read_csv(csv_description_path, sep=',')
    i_s = []
    for i, n_mp in enumerate(description_df['DMS_number_multiple_mutants'].to_list()):
        if n_mp > 0:
            if not single:
                i_s.append(i)
        else:
            if single:
                i_s.append(i)
            else:
                pass
    target_description_df = description_df.iloc[i_s, :]
    target_filenames = target_description_df['DMS_filename'].to_list()
    target_wt_seqs = target_description_df['target_seq'].to_list()
    target_msa_starts = target_description_df['MSA_start'].to_list()
    target_msa_ends = target_description_df['MSA_end'].to_list()
    print(f'Searching for CSV files in {datasets_path}...')
    csv_paths = [os.path.join(datasets_path, target_filename) for target_filename in target_filenames]
    print(f'Found {len(csv_paths)} {type_str}-point datasets, will check if all are available in datasets folder...')
    avail_filenames, avail_csvs, avail_wt_seqs = [], [], []
    for i, csv_path in enumerate(csv_paths):
        if not os.path.isfile(csv_path):
            print(f"Did not find CSV file {csv_path} - will remove it from prediction process!")
        else:
            avail_csvs.append(csv_path)
            avail_wt_seqs.append(target_wt_seqs[i]) 
            avail_filenames.append(os.path.splitext(target_filenames[i])[0])
    assert len(avail_wt_seqs) == len(avail_csvs)
    print(f'Getting data from {len(avail_csvs)} {type_str}-point mutation DMS CSV files...')
    dms_mp_data = {}
    for i, csv_path in enumerate(avail_csvs):
        begin = avail_filenames[i].split('_')[0] + '_' + avail_filenames[i].split('_')[1]
        msa_path=None
        for msa in msas:
            if msa.startswith(begin):
                msa_path = os.path.join(msas_path, msa)
        for pdb in pdbs:
            if pdb.startswith(begin):
                pdb_path = os.path.join(pdbs_path, pdb)
        if msa_path is None or pdb_path is None:
            print(f'Did not find a MSA or a PDB beginning with {begin}, continuing...')
            continue
        target_msa_start = target_msa_starts[i]
        target_msa_end = target_msa_ends[i]
        dms_mp_data.update({
            avail_filenames[i]: {
                'CSV_path': csv_path,
                'WT_sequence': avail_wt_seqs[i], 
                'MSA_path': msa_path,
                'MSA_start': target_msa_start,
                'MSA_end': target_msa_end,
                'PDB_path': pdb_path
            }
        })
    return dms_mp_data


if __name__ == '__main__':
    download_proteingym_data()
    single_mut_data = get_single_or_multi_point_mut_data(os.path.join(os.path.dirname(__file__), '_Description_DMS_substitutions_data.csv'), single=True)
    higher_mut_data = get_single_or_multi_point_mut_data(os.path.join(os.path.dirname(__file__), '_Description_DMS_substitutions_data.csv'), single=False)
    json_output_file_single = os.path.abspath(os.path.join(os.path.dirname(__file__), f"single_point_dms_mut_data.json"))
    json_output_file_higher = os.path.abspath(os.path.join(os.path.dirname(__file__), f"higher_point_dms_mut_data.json"))
    with open(json_output_file_single, 'w') as fp:
        json.dump(single_mut_data, fp, indent=4)
    with open(json_output_file_higher, 'w') as fp:
        json.dump(higher_mut_data, fp, indent=4)
    print(f"Saved path data information as JSON files at {json_output_file_single} and {json_output_file_higher}.")

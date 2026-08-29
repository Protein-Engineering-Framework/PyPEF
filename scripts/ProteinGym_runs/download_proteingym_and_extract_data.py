import os
import argparse
import urllib.request
import zipfile
import pandas as pd
import json
# To use unverified ssl you can add this to your code, taken from:
# https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context


def download_file_with_agent(url: str, output_path: str):
    """Downloads files with a custom User-Agent to prevent Zenodo 403 blocks."""
    req = urllib.request.Request(url, headers={'User-Agent': 'ProteinGym-Downloader/1.0'})
    with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
        while True:
            chunk = response.read(1024 * 1024)  # 1MB chunk size
            if not chunk:
                break
            out_file.write(chunk)


def download_proteingym_data(version: str = '1.3', source: str = 'harvard', zenodo_record_id: str = '13936340'):
    """
    Downloads ProteinGym data from either the Harvard Marks Lab server or Zenodo.
    
    Parameters:
    -----------
    version : str
        The version string used if source='harvard' (e.g., '1.3').
    source : str
        'harvard' to fetch from the Marks Lab server, or 'zenodo' to use Zenodo.
    zenodo_record_id : str
        The Zenodo record identifier used if source='zenodo'.
    """
    file_dir = os.path.dirname(__file__)
    source_type = source.strip().lower()

    if source_type == 'harvard':
        base_url = f'https://marks.hms.harvard.edu/proteingym/ProteinGym_v{version}'
        
        url = f'{base_url}/DMS_substitutions.csv'
        print(f'Getting {url} from Harvard...')
        download_file_with_agent(url, os.path.join(file_dir, '_Description_DMS_substitutions_data.csv'))

        url = f'{base_url}/DMS_ProteinGym_substitutions.zip'
        print(f'Getting {url} from Harvard...')
        zip_path = os.path.join(file_dir, 'DMS_ProteinGym_substitutions.zip')
        download_file_with_agent(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.join(file_dir, 'DMS_ProteinGym_substitutions', '..'))
        os.remove(zip_path)

        url = f'{base_url}/DMS_msa_files.zip'
        print(f'Getting {url} from Harvard...')
        zip_path = os.path.join(file_dir, 'DMS_msa_files.zip')
        download_file_with_agent(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.join(file_dir, 'DMS_msa_files', '..'))
        os.remove(zip_path)

        url = f'{base_url}/ProteinGym_AF2_structures.zip'
        print(f'Getting {url} from Harvard...')
        zip_path = os.path.join(file_dir, 'ProteinGym_AF2_structures.zip')
        download_file_with_agent(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.join(file_dir, 'ProteinGym_AF2_structures', '..'))
        os.remove(zip_path)

    elif source_type == 'zenodo':
        if not zenodo_record_id:
            raise ValueError("A valid zenodo_record_id string must be provided when source='zenodo'")
            
        print(f"Querying Zenodo API for Record ID: {zenodo_record_id}...")
        api_url = f"https://zenodo.org/api/records/{zenodo_record_id}"
        req = urllib.request.Request(api_url, headers={'User-Agent': 'ProteinGym-Downloader/1.0'})
        
        try:
            with urllib.request.urlopen(req) as response:
                record_data = json.loads(response.read().decode())
        except Exception as e:
            print(f"Failed to fetch metadata from Zenodo: {e}")
            return

        files = record_data.get('files', [])
        
        # Map discrete target files if the record contains them individually
        file_mapping = {
            'DMS_substitutions.csv': '_Description_DMS_substitutions_data.csv',
            'DMS_ProteinGym_substitutions.zip': 'DMS_ProteinGym_substitutions.zip',
            'DMS_msa_files.zip': 'DMS_msa_files.zip',
            'ProteinGym_AF2_structures.zip': 'ProteinGym_AF2_structures.zip'
        }
        
        has_individual_files = any(f['key'] in file_mapping for f in files)
        
        if has_individual_files:
            for f in files:
                filename = f['key']
                if filename in file_mapping:
                    download_url = f['links']['self']
                    local_name = file_mapping[filename]
                    local_path = os.path.join(file_dir, local_name)
                    
                    print(f"Downloading {filename} from Zenodo...")
                    download_file_with_agent(download_url, local_path)
                    
                    if local_name.endswith('.zip'):
                        extract_dir = local_name.replace('.zip', '')
                        print(f"Extracting {local_name}...")
                        with zipfile.ZipFile(local_path, "r") as zip_ref:
                            zip_ref.extractall(os.path.join(file_dir, extract_dir, '..'))
                        os.remove(local_path)
        else:
            # Fallback: If it's a monolithic zip file (like ProteinGym_v1.1.zip), unpack everything
            zip_files = [f for f in files if f['key'].endswith('.zip')]
            if zip_files:
                target_archive = zip_files[0]
                filename = target_archive['key']
                download_url = target_archive['links']['self']
                local_path = os.path.join(file_dir, filename)
                
                print(f"Individual tracks not found. Downloading monolithic archive {filename} from Zenodo...")
                download_file_with_agent(download_url, local_path)
                
                print(f"Extracting monolithic archive {filename}...")
                with zipfile.ZipFile(local_path, "r") as zip_ref:
                    zip_ref.extractall(file_dir)
                os.remove(local_path)
            else:
                print("Error: Could not locate appropriate datasets or zip files in this Zenodo record.")


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
        msas_path = os.path.join(file_dirname, 'DMS_msa_files')
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
    parser = argparse.ArgumentParser(description="Download ProteinGym data and extract single/multi-point mutation information.")
    parser.add_argument('--source', type=str, default='harvard', choices=['harvard', 'zenodo'],
                        help="Source to download ProteinGym data from: 'harvard' or 'zenodo'. Default is 'harvard'.")
    parser.add_argument('--version', type=str, default='1.3', help="Version of ProteinGym data to download (only used if source='harvard'). Default is '1.3'.")
    parser.add_argument('--zenodo_record_id', type=str, default='15293562', help="Zenodo record ID to download from (only used if source='zenodo'). Default is '15293562'.")
    args = parser.parse_args()
    # Options:
    # 1. Download from Harvard
    # 2. Download from Zenodo instead (uncomment below and adjust record ID if needed):
    if args.source == 'harvard':
        download_proteingym_data(source='harvard', version='1.3')
    elif args.source == 'zenodo':
        download_proteingym_data(source='zenodo', zenodo_record_id=args.zenodo_record_id)
    else:
        raise ValueError("Invalid source specified. Use 'harvard' or 'zenodo'.")

    single_mut_data = get_single_or_multi_point_mut_data(
        os.path.join(os.path.dirname(__file__), '_Description_DMS_substitutions_data.csv'), 
        single=True
    )
    higher_mut_data = get_single_or_multi_point_mut_data(
        os.path.join(os.path.dirname(__file__), '_Description_DMS_substitutions_data.csv'), 
        single=False
    )
    json_output_file_single = os.path.abspath(
        os.path.join(os.path.dirname(__file__), f"single_point_dms_mut_data.json")
    )
    json_output_file_higher = os.path.abspath(
        os.path.join(os.path.dirname(__file__), f"higher_point_dms_mut_data.json")
    )
    with open(json_output_file_single, 'w') as fp:
        json.dump(single_mut_data, fp, indent=4)
    with open(json_output_file_higher, 'w') as fp:
        json.dump(higher_mut_data, fp, indent=4)
    print(f"Saved path data information as JSON files at "
          f"{json_output_file_single} and {json_output_file_higher}.")
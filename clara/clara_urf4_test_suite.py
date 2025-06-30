import glob
import os
import traceback
import test_helpers.gen_rep_test_set as grts
import test_helpers.toi_importance as timp
import clara_toi_functions as ctoi
import clara_urf_predictor as cupred
import pandas as pd
from sklearn.metrics import auc
import random
from IPython.display import clear_output
import time as tm

def test_urf4_subvariant_runmodel(model_path, sector_data_path, sector_no, toi_fits_files=None, test_results_save_dir=None, gen_sample_subsets=True):

    try:
        all_fits_fnames = [x.split("/")[-1] for x in glob.glob(os.path.join(sector_data_path, '*.fits'))]
        if not all_fits_fnames:
            print("[-] No FITS files found in the sector data path.")
            return
    except Exception as e:
        print(f"[-] Error gathering FITS files: {e}")
        return

    if toi_fits_files is None:
        try:
            print("[+] Gathering Sector TOI Data\n")
            sector_toi_tics = ctoi.get_sector_tic_ids(sector=sector_no)
            toi_fits_files = [x for x in all_fits_fnames if x.endswith(".fits") and int(x.split("-")[2].lstrip("0")) in sector_toi_tics]
            print(f"[+] Number of TOI FITS in whole sector: {len(toi_fits_files)}")
        except Exception as e:
            print(f"[-] Error fetching TOI FITS files: {e}")
            return

    test_sample_size = 4000
    n_test_samples = 10

    test_sets_catalogues_save_dir = "../test/test_set_catalogues/"
    if test_results_save_dir is None:
        test_results_save_dir = "../test/results/"
    test_logs_save_dir = "../test/logs/"
    os.makedirs(test_sets_catalogues_save_dir, exist_ok=True)
    os.makedirs(test_results_save_dir, exist_ok=True)
    os.makedirs(test_logs_save_dir, exist_ok=True)
    
    error_log_path = os.path.join(test_logs_save_dir, "error_log.txt")

    max_workers = 4

    if gen_sample_subsets == True:

        # Step 1: Generate test sets
        for i in range(n_test_samples):
            try:
                seed = random.randint(0, 10000)
                this_test_sample = grts.generate_representative_test_set(
                    all_fits_fnames, toi_fits_files, sample_size=test_sample_size, seed=seed)
                this_test_sample_catalogue_file = os.path.join(test_sets_catalogues_save_dir, f"test_set_{i}_fits_files.txt")
                with open(this_test_sample_catalogue_file, "w") as f:
                    for fname in this_test_sample:
                        f.write(fname + "\n")
            except Exception as e:
                with open(error_log_path, "a") as elog:
                    elog.write(f"[Set Generation Error] Test Set {i}: {e}\n")
                    elog.write(traceback.format_exc())
                continue
    
        print(f"[+] Generated {n_test_samples} test sets in {test_sets_catalogues_save_dir}\n")

    # Step 2: Run model on each test set
    sample_files = glob.glob(os.path.join(test_sets_catalogues_save_dir, "*.txt"))
    c=0
    for test_sample in sample_files:
        c+=1
        clear_output(wait=True)
        print(f"{c}/{len(sample_files)}")
        try:
            save_csv_path = os.path.join(
                test_results_save_dir, os.path.basename(test_sample).replace(".txt", "_results.csv"))

            with open(test_sample, "r") as f:
                test_sample_fits_list = [line.strip() for line in f.readlines()]
            
            test_sample_fits_list = [x for x in test_sample_fits_list if x in all_fits_fnames]

            print(f"[+] Running model on {test_sample} ({len(test_sample_fits_list)} files)...")

            cupred.get_anomaly_scores_from_folder_parallelized_streamed_mp(
                folder_path=sector_data_path,
                model_path=model_path,
                save_csv=save_csv_path,
                save_dir=None,
                max_workers=max_workers,
                subset=test_sample_fits_list
            )
        except Exception as e:
            with open(error_log_path, "a") as elog:
                elog.write(f"[Model Run Error] Test Set File: {test_sample}\nError: {e}\n")
                elog.write(traceback.format_exc())
            print(f"[-] Error running model on {test_sample}, logged.")
            continue
        tm.sleep(300)
        
def calculate_auc_metrics_from_results(
    results_dir,
    sector_no,
    threshold_steps=100,
    first_n = None
):

    all_fits_fnames = [x.split("/")[-1] for x in glob.glob(os.path.join(f"../../downloaded_lc/tess_lc/{sector_no}/", '*.fits'))]
    print("Accessing TOI data from: https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv")
    sector_toi_tics = ctoi.get_sector_tic_ids(sector=sector_no)
    toi_filenames = [x for x in all_fits_fnames if x.endswith(".fits") and int(x.split("-")[2].lstrip("0")) in sector_toi_tics]
    
    all_results = []
    for path in glob.glob(os.path.join(results_dir, "*_results.csv")):
        try:
            df = pd.read_csv(path)
            all_results.append([df, os.path.basename(path)])
        except Exception as e:
            print(f"Skipping {path}: {e}")
    
    if not all_results:
        raise RuntimeError("No valid result CSVs found.")

    # df_all = pd.concat(all_results, ignore_index=True)
    recall_auc_list = []
    importance_auc_list = []
    toi_recall_list = []

    if first_n:
        all_results = all_results[:first_n]
    for result, sample_name in all_results:
        print(f"Calculating auc metrics for result: {sample_name}")
        result['is_toi'] = result['filename'].isin(toi_filenames)

        sample_tois = result[result["filename"].isin(toi_filenames)]["filename"].tolist()
        sample_tic_ids = [int(x.split("-")[2].lstrip("0")) for x in sample_tois]
        # result_toi_df = timp.get_toi_importance_scores_from_filenames(
                            #     toi_filenames=sample_tois,
                            #     sector_tic_ids=sample_tic_ids
                            # )
    
        total_tois = result['is_toi'].sum()
        total_files = len(result)

        if total_tois == 0:
            raise ValueError("No TOIs found in the combined results.")

        thresholds = sorted(result['anomaly_score'].quantile(q=i/threshold_steps) for i in range(threshold_steps + 1))

        recall_values = []
        importance_values = []

        for thresh in thresholds:
            df_thresh = result[result['anomaly_score'] >= thresh]
    
            if len(df_thresh) == 0:
                continue
            
            n_flagged = len(df_thresh)
            n_flagged_toi = df_thresh['is_toi'].sum()
    
            recall = n_flagged_toi / total_tois
            thresh_tois_df = df_thresh[df_thresh["filename"].isin(sample_tois)]
            # importance_norm = result_toi_df[result_toi_df["filename"].isin(thresh_tois_df["filename"])]["normalized_score"].mean()
    
            recall_values.append(recall)
            # importance_values.append(importance_norm)
    
        # recall_auc = auc(thresholds[:len(recall_values)], recall_values)
        # importance_auc = auc(thresholds[:len(importance_values)], importance_values)

        # recall_auc_list.append([recall_auc, sample_name])
        # importance_auc_list.append([importance_auc, sample_name])
        avg_rec = 0
        for rec in recall_values:
            avg_rec += rec
        avg_rec = avg_rec/len(recall_values)
        print(avg_rec)
    # return {
    #     "toi_recall_auc_list": recall_auc_list,
    #     "toi_importance_auc": importance_auc_list,
    # }

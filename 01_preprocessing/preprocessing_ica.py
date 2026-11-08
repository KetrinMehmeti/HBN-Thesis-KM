#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# ─────────────────────────────────────────────────────────────
# AWS / multiprocessing safety: set BEFORE numpy/scipy imports
# ─────────────────────────────────────────────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import concurrent.futures
import gc
import io
import json
import logging
import multiprocessing
import pickle
import shutil
import tempfile
import traceback
import warnings
from typing import Dict, List, Optional, Tuple

import boto3
import mne
import numpy as np
import pandas as pd
import s3fs
from mne.preprocessing import ICA

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

try:
    from mne_icalabel import label_components
    HAVE_ICALABEL = True
except Exception:
    HAVE_ICALABEL = False


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
HBN_S3_RELEASES = {
    **{f"R{i}": f"fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R{i}" for i in range(1, 10)},
    "RNC": "fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_NC",
}

NON_TASK_COLS = {
    "participant_id", "release_number", "sex", "age", "ehq_total",
    "commercial_use", "full_pheno", "p_factor", "attention",
    "internalizing", "externalizing", "site", "session_id", "cohort",
    "EHQ_Total", "Commercial", "CommercialUse",
}

SFREQ_TARGET = 200

ICA_FIT_HPASS = 1.0
APPLY_HPASS = 0.1
LPASS = 75.0
NOTCH = 60.0

SHORT_WIN_DUR = 2.0
FINAL_EPOCH_DUR = 10.0

MAD_MULTIPLIER = 3.0
PTP_PERCENTILE = 97
THRESH_MIN_UV = 100.0
THRESH_MAX_UV = 1500.0
MAX_FRAC_REJECT = 0.50

MUSCLE_BAND = (20.0, 40.0)
MUSCLE_ZMAX = 4.0
VAR_ZMAX = 4.0
PTP_ZMAX = 4.0

ICA_METHOD = "picard"
ICA_RANDOM_STATE = 42
ICA_MAX_ITER = 500
ICA_N_COMPONENTS = 0.99
MAX_ICA_EXCLUDE = 6
ICLABEL_THRESH = {
    "eye blink": 0.90,
    "muscle artifact": 0.90,
    "heart beat": 0.90,
    "channel noise": 0.90,
    "line noise": 0.95,
}

EOG_PROXY_CHANNELS = ["E1", "E8", "E14", "E21", "E25"]

FLUSH_EVERY = 10
FUTURE_TIMEOUT = 1800


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def detect_bad_channels(raw: mne.io.BaseRaw) -> List[str]:
    data = raw.get_data()
    var = np.var(data, axis=1)
    med = np.median(var) if len(var) else 0.0
    if med <= 0:
        return []
    bad_idx = np.where((var < med * 0.01) | (var > med * 1000))[0]
    return [raw.ch_names[i] for i in bad_idx]


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-12:
        return np.zeros_like(x, dtype=float)
    return 0.6745 * (x - med) / mad


def load_and_prepare_raw(set_path: str) -> Tuple[mne.io.BaseRaw, mne.io.BaseRaw]:
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)

    to_drop = [ch for ch in ("E129", "VREF") if ch in raw.ch_names]
    if to_drop:
        raw.drop_channels(to_drop)

    montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
    raw.set_montage(montage, match_case=False, on_missing="ignore")
    raw.pick("eeg")

    no_pos = [
        ch["ch_name"]
        for ch in raw.info["chs"]
        if np.allclose(ch["loc"][:3], 0.0) or np.any(np.isnan(ch["loc"][:3]))
    ]
    if no_pos:
        raw.drop_channels([ch for ch in no_pos if ch in raw.ch_names])

    bads = detect_bad_channels(raw)
    if bads:
        raw.info["bads"] = bads
        raw.interpolate_bads(reset_bads=True)

    if raw.info["sfreq"] > SFREQ_TARGET:
        raw.resample(SFREQ_TARGET, verbose=False)

    raw_apply = raw.copy().filter(l_freq=APPLY_HPASS, h_freq=LPASS, method="fir", verbose=False)
    raw_apply.notch_filter(freqs=NOTCH, verbose=False)
    raw_apply.set_eeg_reference("average", projection=False, verbose=False)

    raw_ica = raw.copy().filter(l_freq=ICA_FIT_HPASS, h_freq=LPASS, method="fir", verbose=False)
    raw_ica.notch_filter(freqs=NOTCH, verbose=False)
    raw_ica.set_eeg_reference("average", projection=False, verbose=False)

    del raw
    gc.collect()
    return raw_apply, raw_ica


def short_window_quality_scan(raw: mne.io.BaseRaw) -> Optional[Dict]:
    sf = raw.info["sfreq"]
    win_samp = int(round(SHORT_WIN_DUR * sf))
    data = raw.get_data()
    n_samp = data.shape[1]
    n_win = n_samp // win_samp
    if n_win == 0:
        return None

    trimmed = data[:, : n_win * win_samp]
    x = trimmed.reshape(data.shape[0], n_win, win_samp).transpose(1, 0, 2)

    ptp_max_uv = (x.max(axis=2) - x.min(axis=2)).max(axis=1) * 1e6
    var_med = np.median(np.var(x, axis=2), axis=1)

    muscle_raw = raw.copy().filter(MUSCLE_BAND[0], MUSCLE_BAND[1], method="fir", verbose=False)
    broad_raw = raw.copy().filter(1.0, 40.0, method="fir", verbose=False)

    mdat = muscle_raw.get_data()[:, : n_win * win_samp].reshape(data.shape[0], n_win, win_samp).transpose(1, 0, 2)
    bdat = broad_raw.get_data()[:, : n_win * win_samp].reshape(data.shape[0], n_win, win_samp).transpose(1, 0, 2)

    del muscle_raw, broad_raw
    gc.collect()

    muscle_pow = np.median(np.mean(mdat ** 2, axis=2), axis=1)
    broad_pow = np.median(np.mean(bdat ** 2, axis=2), axis=1)
    muscle_score = muscle_pow / np.maximum(broad_pow, 1e-12)

    z_ptp = robust_z(ptp_max_uv)
    z_var = robust_z(var_med)
    z_mus = robust_z(muscle_score)

    bad_mask = (z_ptp > PTP_ZMAX) | (z_var > VAR_ZMAX) | (z_mus > MUSCLE_ZMAX)

    return {
        "bad_mask": bad_mask,
        "stats": {
            "n_windows": int(n_win),
            "n_bad_windows": int(bad_mask.sum()),
            "bad_frac": float(round(bad_mask.mean(), 3)),
            "ptp_med_uv": float(round(np.median(ptp_max_uv), 1)),
            "ptp_p95_uv": float(round(np.percentile(ptp_max_uv, 95), 1)),
            "muscle_p95": float(round(np.percentile(muscle_score, 95), 4)),
        },
    }


def annotate_bad_windows(raw: mne.io.BaseRaw, qa: Dict) -> mne.io.BaseRaw:
    raw = raw.copy()
    onsets, durations, desc = [], [], []
    for i, is_bad in enumerate(qa["bad_mask"]):
        if is_bad:
            onsets.append(i * SHORT_WIN_DUR)
            durations.append(SHORT_WIN_DUR)
            desc.append("BAD_QA2S")
    if onsets:
        ann = mne.Annotations(onset=onsets, duration=durations, description=desc)
        raw.set_annotations(raw.annotations + ann)
    return raw


def fit_and_apply_ica(raw_apply: mne.io.BaseRaw, raw_ica: mne.io.BaseRaw):
    raw_ica_annot = raw_ica.copy()
    qa = short_window_quality_scan(raw_ica_annot)
    if qa is not None:
        raw_ica_annot = annotate_bad_windows(raw_ica_annot, qa)

    fit_params = {"ortho": False, "extended": True} if ICA_METHOD == "picard" else None
    ica = ICA(
        n_components=ICA_N_COMPONENTS,
        method=ICA_METHOD,
        fit_params=fit_params,
        max_iter=ICA_MAX_ITER,
        random_state=ICA_RANDOM_STATE,
        verbose=False,
    )
    ica.fit(raw_ica_annot, reject_by_annotation=True, verbose=False)

    exclude = []

    if HAVE_ICALABEL:
        labels = label_components(raw_ica_annot, ica, method="iclabel")
        for idx, (lab, prob) in enumerate(zip(labels["labels"], labels["y_pred_proba"])):
            thr = ICLABEL_THRESH.get(lab, None)
            if thr is not None and prob >= thr:
                exclude.append(idx)
    else:
        eog_chs = [ch for ch in EOG_PROXY_CHANNELS if ch in raw_apply.ch_names]
        try:
            if eog_chs:
                eog_idx, _ = ica.find_bads_eog(raw_apply, ch_name=eog_chs, threshold=3.0, verbose=False)
                exclude.extend(eog_idx)
        except Exception:
            pass
        try:
            ecg_idx, _ = ica.find_bads_ecg(raw_apply, method="correlation", threshold=0.3, verbose=False)
            exclude.extend(ecg_idx)
        except Exception:
            pass

    exclude = list(dict.fromkeys(exclude))[:MAX_ICA_EXCLUDE]
    ica.exclude = exclude

    raw_clean = raw_apply.copy()
    ica.apply(raw_clean, verbose=False)

    del raw_ica_annot, raw_ica, ica
    gc.collect()
    return raw_clean, qa, exclude


def make_final_epochs(raw_annot: mne.io.BaseRaw) -> mne.Epochs:
    epochs = mne.make_fixed_length_epochs(
        raw_annot,
        duration=FINAL_EPOCH_DUR,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )
    epochs.apply_baseline((None, None))
    return epochs


def robust_ptp_threshold(epochs: mne.Epochs) -> Tuple[float, np.ndarray, np.ndarray]:
    data = epochs.get_data()
    ptp_max = (data.max(axis=2) - data.min(axis=2)).max(axis=1) * 1e6

    med = np.median(ptp_max)
    mad = np.median(np.abs(ptp_max - med))

    if mad < 1e-9:
        thresh_uv = float(np.clip(med, THRESH_MIN_UV, THRESH_MAX_UV))
        clean_mask = np.ones_like(ptp_max, dtype=bool)
    else:
        clean_mask = ptp_max <= (med + MAD_MULTIPLIER * mad)
        if clean_mask.sum() < 5:
            thresh_uv = float(np.clip(med, THRESH_MIN_UV, THRESH_MAX_UV))
        else:
            thresh_uv = float(
                np.clip(np.percentile(ptp_max[clean_mask], PTP_PERCENTILE), THRESH_MIN_UV, THRESH_MAX_UV)
            )

    return thresh_uv, ptp_max, clean_mask


def reject_final_epochs(epochs: mne.Epochs):
    n_total = len(epochs)
    if n_total == 0:
        return None, {
            "n_total": 0,
            "n_kept": 0,
            "frac_rejected": 1.0,
            "reject_threshold_uv": None,
            "clean_frac_premask": 0.0,
        }

    thresh_uv, ptp_max, clean_mask = robust_ptp_threshold(epochs)
    keep_mask = ptp_max <= thresh_uv
    n_kept = int(keep_mask.sum())
    frac_rej = 1.0 - (n_kept / n_total)

    stats = {
        "n_total": int(n_total),
        "n_kept": int(n_kept),
        "frac_rejected": float(round(frac_rej, 3)),
        "reject_threshold_uv": float(round(thresh_uv, 1)),
        "clean_frac_premask": float(round(clean_mask.mean(), 3)),
        "ptp_median_uv": float(round(np.median(ptp_max), 1)),
        "ptp_p95_uv": float(round(np.percentile(ptp_max, 95), 1)),
    }

    if frac_rej > MAX_FRAC_REJECT or n_kept == 0:
        return None, stats
    return epochs[keep_mask], stats


def normalize_div100(data: np.ndarray) -> np.ndarray:
    return (data * 1e6 / 100.0).astype(np.float32)


def upload_channel_info(bucket: str, prefix: str, s3_client) -> None:
    key = f"{prefix}/channel_info.pkl"
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return
    except Exception:
        pass

    montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
    ch_names = montage.ch_names
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ_TARGET, ch_types="eeg")
    raw_fake = mne.io.RawArray(np.zeros((len(ch_names), SFREQ_TARGET)), info, verbose=False)
    raw_fake.set_montage(montage, match_case=False)
    raw_fake.pick("eeg")

    buf = io.BytesIO()
    pickle.dump(raw_fake.info["chs"], buf)
    buf.seek(0)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buf.read())


def flush_log(rows: List[Dict], bucket: str, key: str, s3_client) -> None:
    buf = io.BytesIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    buf.seek(0)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buf.read())


def load_hbn_metadata(fs: s3fs.S3FileSystem) -> pd.DataFrame:
    dfs = []
    for release, bucket_path in HBN_S3_RELEASES.items():
        try:
            with fs.open(f"{bucket_path}/participants.tsv", "r") as f:
                df = pd.read_csv(f, sep="\t")
                if release == "RNC":
                    df["release_number"] = "NC"
                dfs.append(df)
        except Exception:
            pass

    if not dfs:
        raise RuntimeError("Impossibile caricare participants.tsv dalle release HBN.")

    return pd.concat(dfs, ignore_index=True).drop_duplicates("participant_id")


def already_done(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def infer_release_path(release_number: str) -> str:
    rel = str(release_number).strip()
    if rel in {"NC", "RNC"}:
        return HBN_S3_RELEASES["RNC"]
    rel_num = rel.replace("R", "").split(".")[0].strip()
    return f"fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R{rel_num}"


def process_subject(job: Dict) -> List[Dict]:
    subj = job["participant_id"]
    bucket = job["output_bucket"]
    prefix = job["output_prefix"]
    release_path = job["release_path"]

    # Trasforma la lista passata in un Set per ricerca veloce (Tuple)
    done_tasks_for_worker = set(tuple(x) for x in job.get("done_tasks", []))

    fs_hbn = s3fs.S3FileSystem(anon=True)
    s3_client = boto3.client("s3", region_name=job["region_name"])
    tmp_dir = tempfile.mkdtemp(prefix=f"hbn_{subj}_")
    log_entries = []

    try:
        eeg_dir = f"{release_path}/{subj}/eeg"
        try:
            all_files = fs_hbn.ls(eeg_dir)
        except Exception:
            return [{"participant_id": subj, "task": "ALL", "status": "skip", "note": "Cartella EEG non trovata su S3"}]

        set_files = sorted([f for f in all_files if f.endswith(".set")])
        if not set_files:
            return [{"participant_id": subj, "task": "ALL", "status": "skip", "note": "Nessun file .set trovato"}]

        for eeg_key in set_files:
            filename = eeg_key.split("/")[-1]

            try:
                task_part = filename.split("task-")[1].split("_")[0]
                run_part = "_" + filename.split("run-")[1].split("_")[0] if "run-" in filename else ""
                task_name = task_part + run_part
            except IndexError:
                task_name = "UnknownTask"

            # IL NUOVO CONTROLLO PIÙ POTENTE!
            if (subj, task_name) in done_tasks_for_worker:
                 # Lo abbiamo già fatto (ok o escluso) nel file CSV, saltalo subito!
                 continue

            s3_final_key = f"{prefix}/{task_name}/{subj}.npy"
            if already_done(s3_client, bucket, s3_final_key):
                log_entries.append({
                    "participant_id": subj,
                    "task": task_name,
                    "status": "ok",
                    "note": "Gia processato"
                })
                continue

            log_entry = {"participant_id": subj, "task": task_name, "status": "", "note": ""}

            fdt_key = eeg_key.replace(".set", ".fdt")
            set_dst = os.path.join(tmp_dir, filename)
            fdt_dst = set_dst.replace(".set", ".fdt")

            try:
                with fs_hbn.open(eeg_key, "rb") as fi, open(set_dst, "wb") as fo:
                    shutil.copyfileobj(fi, fo)

                if fs_hbn.exists(fdt_key):
                    with fs_hbn.open(fdt_key, "rb") as fi, open(fdt_dst, "wb") as fo:
                        shutil.copyfileobj(fi, fo)

                raw_apply, raw_ica = load_and_prepare_raw(set_dst)

                raw_clean, qa_2s, exclude = fit_and_apply_ica(raw_apply, raw_ica)
                qa_stats = qa_2s["stats"] if qa_2s is not None else {}

                raw_annot = annotate_bad_windows(raw_clean, qa_2s) if qa_2s is not None else raw_clean
                epochs_10s = make_final_epochs(raw_annot)

                del raw_apply, raw_clean, raw_annot
                gc.collect()

                epochs_keep, mad_stats = reject_final_epochs(epochs_10s)

                del epochs_10s
                gc.collect()

                merged = {
                    **qa_stats,
                    **mad_stats,
                    "n_ica_excluded": len(exclude),
                    "ica_excluded_idx": exclude,
                    "icalabel_used": bool(HAVE_ICALABEL),
                }

                if epochs_keep is None:
                    log_entry.update({"status": "exclude_ica_mad", "note": json.dumps(merged)})
                else:
                    tensor = normalize_div100(epochs_keep.get_data())
                    del epochs_keep
                    gc.collect()

                    buf = io.BytesIO()
                    np.save(buf, tensor)
                    buf.seek(0)
                    s3_client.put_object(Bucket=bucket, Key=s3_final_key, Body=buf.read())

                    log_entry.update({
                        "status": "ok",
                        "note": json.dumps({**merged, "shape": list(tensor.shape)}),
                    })

                    del tensor
                    gc.collect()

            except Exception as exc:
                tb = traceback.format_exc(limit=8)
                log_entry.update({
                    "status": "error",
                    "note": f"{str(exc)[:220]} | {tb[:1200]}",
                })
            finally:
                if os.path.exists(set_dst):
                    os.remove(set_dst)
                if os.path.exists(fdt_dst):
                    os.remove(fdt_dst)
                gc.collect()

            log_entries.append(log_entry)

        return log_entries

    except Exception as exc:
        return [{
            "participant_id": subj,
            "task": "ALL",
            "status": "fatal",
            "note": str(exc)[:300],
        }]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        gc.collect()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="HBN EEG AWS pipeline: ICA + QA2S + MAD + div100")
    p.add_argument("--output-bucket", required=True)
    p.add_argument("--output-prefix", default="ica_qamad_ready")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--region-name", default="us-east-1")
    p.add_argument("--limit-subjects", type=int, default=None)
    p.add_argument("--flush-every", type=int, default=FLUSH_EVERY)
    return p.parse_args()


def main():
    args = parse_args()

    multiprocessing.set_start_method("spawn", force=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    s3 = boto3.client("s3", region_name=args.region_name)
    fs = s3fs.S3FileSystem(anon=True)

    upload_channel_info(args.output_bucket, args.output_prefix, s3)

    log.info("Caricamento metadati HBN...")
    unified = load_hbn_metadata(fs)
    log.info(f"Partecipanti totali: {len(unified)}")

    log_key = f"{args.output_prefix}/preprocessing_log_ica_qamad.csv"

    done_subjects = set()
    done_task_pairs = set()
    prior_rows = []

    try:
        obj = s3.get_object(Bucket=args.output_bucket, Key=log_key)
        df_log = pd.read_csv(io.BytesIO(obj["Body"].read()))
        # Modifica: Metti anche "error" tra gli stati che non vuoi ripetere,
        # a patto che tu non voglia riprovarli. Se vuoi riprovare gli error, lascialo così.
        done = df_log[df_log["status"].isin({"ok", "exclude_ica_mad"})]

        if {"participant_id", "task"}.issubset(done.columns):
            done_task_pairs = set(zip(done["participant_id"].astype(str), done["task"].astype(str)))

        prior_rows = df_log.to_dict("records")
        log.info(f"Resume attivo: {len(done_task_pairs)} task gia completati")
    except Exception:
        log.info("Nessun log precedente: partenza da zero")
        done_task_pairs = set() # ASSICURATI DI INIZIALIZZARE QUI

    jobs = []
    for _, row in unified.iterrows():
        subj = str(row["participant_id"])
        release_path = infer_release_path(row["release_number"])

        jobs.append({
            "participant_id": subj,
            "release_path": release_path,
            "output_bucket": args.output_bucket,
            "output_prefix": args.output_prefix,
            "region_name": args.region_name,
            "done_tasks": list(done_task_pairs) # <--- PASSIAMO LA LISTA AI WORKER!
        })

    if args.limit_subjects is not None:
        jobs = jobs[: args.limit_subjects]

    log.info(f"Soggetti in coda: {len(jobs)}")

    all_rows = list(prior_rows)
    completed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_subject, j): j for j in jobs}


        import time
        start_time = time.time()
        ok_count = sum(1 for r in prior_rows if r.get("status") == "ok")
        exclude_count = sum(1 for r in prior_rows if r.get("status") == "exclude_ica_mad")
        fatal_count = sum(1 for r in prior_rows if r.get("status") in {"fatal", "error"})

        for fut in concurrent.futures.as_completed(futures):
            completed += 1
            job = futures[fut]

            try:
                entries = fut.result(timeout=FUTURE_TIMEOUT)
                n_ok_now = sum(1 for e in entries if e.get("status") == "ok")
                n_excl_now = sum(1 for e in entries if e.get("status") == "exclude_ica_mad")
                n_err_now = sum(1 for e in entries if e.get("status") in {"fatal", "error"})

                ok_count += n_ok_now
                exclude_count += n_excl_now
                fatal_count += n_err_now

                elapsed_h = max((time.time() - start_time) / 3600.0, 1e-9)
                task_done = ok_count + exclude_count
                rate = task_done / elapsed_h

                remaining_jobs = len(jobs) - completed
                eta_h = remaining_jobs / max(completed / elapsed_h, 1e-9)

                log.info(
                    f"[{completed}/{len(jobs)} subj] "
                    f"ok={ok_count} excl={exclude_count} err={fatal_count} "
                    f"rate={rate:.1f} task/h ETA_subj={eta_h:.1f}h"
                )
            except Exception as exc:
                entries = [{
                    "participant_id": job["participant_id"],
                    "task": "ALL",
                    "status": "fatal",
                    "note": str(exc)[:300],
                }]

            for entry in entries:
                key = (str(entry["participant_id"]), str(entry["task"]))
                if key in done_task_pairs and entry.get("status") in {"ok", "exclude_ica_mad"}:
                    continue
                all_rows.append(entry)
                if entry.get("status") in {"ok", "exclude_ica_mad"}:
                    done_task_pairs.add(key)

            ok_tasks = [e["task"] for e in entries if e.get("status") == "ok"]
            if ok_tasks:
                log.info(f"[{completed}/{len(jobs)}] {job['participant_id']} -> {', '.join(ok_tasks)}")
            else:
                log.info(f"[{completed}/{len(jobs)}] {job['participant_id']} -> {entries[0]['status']}")

            if completed % args.flush_every == 0:
                flush_log(all_rows, args.output_bucket, log_key, s3)
                log.info(f"Log flush su s3://{args.output_bucket}/{log_key}")

    flush_log(all_rows, args.output_bucket, log_key, s3)
    df_final = pd.DataFrame(all_rows)
    if not df_final.empty and "status" in df_final.columns:
        log.info("=== RIEPILOGO ===\n" + df_final["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
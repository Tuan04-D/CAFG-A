"""Preprocessing steps shared by both corpora."""

import mne
import numpy as np

TARGET_SFREQ = 256.0
EPOCH_SAMPLES = 1280


def filter_continuous(data, sfreq, l_freq, h_freq, notch_freq):
    # IIR: an FIR filter needs a length several times the signal, which a
    # 5 s epoch cannot supply.
    data = mne.filter.notch_filter(data, sfreq, notch_freq, method="iir",
                                   verbose=False)
    data = mne.filter.filter_data(data, sfreq, l_freq, h_freq, method="iir",
                                  verbose=False)
    return data


def resample_continuous(data, sfreq, target_sfreq=TARGET_SFREQ):
    return mne.filter.resample(data, up=target_sfreq, down=sfreq, verbose=False)


def common_average_reference(data):
    return data - data.mean(axis=0, keepdims=True)


def remove_eog_ica(raw, eog_proxy_channels, n_components=20, random_state=97):
    ica = mne.preprocessing.ICA(n_components=n_components, method="fastica",
                                random_state=random_state, max_iter="auto")
    ica.fit(raw, verbose=False)
    try:
        eog_indices, _ = ica.find_bads_eog(raw, ch_name=eog_proxy_channels,
                                           verbose=False)
        ica.exclude = eog_indices
    except RuntimeError:
        ica.exclude = []
    ica.apply(raw, verbose=False)
    return raw, ica.exclude


def drop_bad_epochs(x, k=5.0):
    ptp = (x.max(axis=2) - x.min(axis=2)).max(axis=1)
    median = np.median(ptp)
    mad = np.median(np.abs(ptp - median))
    return ptp <= median + k * 1.4826 * mad


def zscore_per_channel(x):
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True)
    return (x - mean) / np.where(std == 0, 1.0, std)

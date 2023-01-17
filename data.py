import hyperparams as hp
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
from text import text_to_sequence
import collections
from scipy import signal

# LJDataset Load
class LJDatasets(Dataset):
    def __init__(self, csv_file, root_dir):
        '''
        Args:
            csv_file(string) : annotaion의 경로가 적힌 csv file
            root_dir(string) : wav file들의 root 디렉토리
        '''
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def load_wav(self, filename):
        return librosa.load(filename, sr = hp.sample_rate)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, self.landmarks_fram.iloc[idx, 0]) + '.wav'
        text = self.landmarks_frame.iloc[idx, 1]
        text = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)
        wav = np.asarray(self.load_wav(wav_name)[0], dtype=np.float32)
        data_dict = {'text': text, 'wav' : wav}

        return data_dict

def collate_fn(batch):
    # data_dict (batch_size 만큼의 데이터를 가지고 있다.)
    if isinstance(batch[0], collections.Mapping):
        keys = list()

        text = [d['text'] for d in batch]
        wav = [d['text'] for d in batch]

        magnitude = np.array([spectrogram(w) for w in wav]) # spectrum은 magnitude를 가지지만 phase는 가지지 못한다.
        mel = np.array([melspectrogram(w) for w in wav])
        timesteps = mel.shape[-1]

        if timesteps % hp.outputs_per_step != 0: # 나머지가 0이 아니면 zero padding 수행
            magnitude = _pad_per_step(magnitude)
            mel = _pad_per_step(mel)

        return text, magnitude, mel

    raise TypeError(('batch must contain tensors, numbers, dicts or lists'))

        
#-- data 전처리 함수들
_mel_basis = None

def save_wav(wav, path):
    ''' waveform 저장 function'''

    # int의 range는 32768 to 32767이므로 normalize를 위해 밑의 식을 이용하여 wav 저장
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    librosa.output.wrtie_wav(path, wav.astype(np.int16), hp.sample_rate)

def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()

    return np.dot(_mel_basis, spectrogram)

def _build_mel_basis():
    n_fft = (hp.num_freq - 1) * 2
    return librosa.filters.mel(hp.sample_rate, n_fft, n_mels=hp.num_mels)

def _normalize(S):
    return np.clip((S - hp.min_level_db) / -hp.min_level_db, 0, 1)

def _denormalize(S):
    return (np.clip(S, 0, 1) * -hp.min_level_db) + hp.min_level_db

def _stft_parameters():
        n_fft = (hp.num_freq - 1) * 2
        hop_length = int(hp.frame_shift_ms / 1000 * hp.sample_rate) #window끼리 겹치는 정도
        win_length = int(hp.frame_legth_ms / 1000 * hp.sample_rate) # window 길이

        return n_fft, hop_length, win_length

def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))  # amplitude -> db

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def preemphasis(x):
        return signal.lfilter([1, -hp.preemphasis], [1], x)

def inv_preemphasis(x):
        return signal.lfilter([1], [1, -hp.preemphasis], x)

def _stft(y):
    ''' stft 수행'''
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _istft(y):
    _, hop_length, win_length = _stft_parameters()
    return librosa.isrfr(y, hop_length=hop_length, win_length=win_length)

def spectrogram(y):
    ''' spectrogram generate '''
    D = _stft(preemphasis(y))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db
    return _normalize(S)

def inv_spectrogram(spectrogram):
    S = _denormalize(spectrogram)
    S = _db_to_amp(S + hp.ref_level_db)

    return inv_preemphasis(_griffin_lim(S ** hp.power))

def _griffin_lim(S):
    ''' griffin_lim 공식 -> phase 를 추정한다. '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)

    y = _istft(S_complex * angles)

    for i in range(hp.griffin_lim_iters):
        angles= np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)

    return y

def melspectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(_linear_to_mel(np.abs(D)))
    return _normalize(S)

def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
      window_length = int(hp.sample_rate * min_silence_sec)
      hop_length = int(window_length / 4)
      threshold = _db_to_amp(threshold_db)

      for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x:x+window_length]) < threshold:
          return x + hop_length
      return len(wav)

def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

def _pad_per_step(inputs):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, hp.outputs_per_step - (timesteps % hp.outputs_per_step)]], mode='constant', constant_values=0.0)

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params

def get_dataset():
    return LJDatasets(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))

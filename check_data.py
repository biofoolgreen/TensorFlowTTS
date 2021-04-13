import numpy as np
import os

class DataTypeError(Exception):
    pass

class SingleData(object):
    def __init__(self, uids, root_path, is_train=True):
        self.uids = uids
        base_folder_name = 'train' if is_train else 'valid'
        self.base_path = os.path.join(root_path, base_folder_name)
        self._set_path()

    def _set_path(self):
        self.dur_path = os.path.join(self.base_path, 'duration')
        self.ids_path = os.path.join(self.base_path, 'ids')
        self.mel_path = os.path.join(self.base_path, 'norm-feats')
        self.raw_feats_path = os.path.join(self.base_path, 'raw-feats')
        self.pitch_path = os.path.join(self.base_path, 'raw-f0')
        self.energy_path = os.path.join(self.base_path, 'raw-energies')
        self.wav_path = os.path.join(self.base_path, 'wavs')
    
    def get_path(self, name):
        path_dict = {
            "duration": self.dur_path,
            "mel": self.mel_path,
            "pitch": self.pitch_path,
            "wav": self.wav_path,
            "energy": self.energy_path,
            "ids": self.ids_path,
            "raw-feats": self.raw_feats_path
        }
        return path_dict[name]
    
    def fetch(self, uid):
        utt_id = np.load(self.ids_path+f"/{uid}-ids.npy")
        dur = np.load(self.dur_path+f"/{uid}-durations.npy")
        mel = np.load(self.mel_path+f"/{uid}-norm-feats.npy")
        pitch = np.load(self.pitch_path+f"/{uid}-raw-f0.npy")
        energy = np.load(self.energy_path+f"/{uid}-raw-energy.npy")
        raw_feat = np.load(self.raw_feats_path+f"/{uid}-raw-feats.npy")
        wav = np.load(self.wav_path+f"/{uid}-wave.npy")
        return (utt_id, pitch, energy, dur, raw_feat, mel, wav)
    
    def fetch_all(self):
        if isinstance(self.uids, str):
            return [self.fetch(self.uids)]
        elif isinstance(self.uids, list):
            samples = []
            for uid in self.uids:
                samples.append(self.fetch(uid))
            return samples
        else:
            raise DataTypeError(f"`uids` should either be List or str type, not {type(self.uids)}")

def test(is_train=True):
    root_path = "/fsx/home/guoyingli/speech/TensorFlowTTS/dump_ljspeech"
    if is_train:
        utts = np.load(root_path+"/train_utt_ids.npy")
    else:
        utts = np.load(root_path+"/valid_utt_ids.npy")

    uids = np.random.choice(utts, size=3)
    print(f"Choose 3 uid: {uids}")
    data = SingleData(uids=uids, root_path=root_path, is_train=is_train)

    utt_id, pitch, energy, dur, raw_feat, mel, wav = data.fetch(uids[0])
    print(f"phoneme:{utt_id.shape}\npitch:{pitch.shape}\nenergy:{energy.shape}\nduration:{dur.shape}\nraw_feat:{raw_feat.shape}\nmel:{mel.shape}\nwav:{wav.shape}")
    print(f"phoneme: {utt_id}\nduration: {dur}")
    print(f"sum(duration)={sum(dur)}")


if __name__ == '__main__':
    test()
import os
from tqdm import tqdm
import numpy as np

def add0(nums, dtype=np.int32):
    nums = np.append(nums, 0).astype(dtype)
    return nums

def run(root_path):
    train_dur_path = os.path.join(root_path, 'train', 'duration')
    valid_dur_path = os.path.join(root_path, 'valid', 'duration')
    print(f"Processing train duration...")
    for tdur in tqdm(os.listdir(train_dur_path)):
        tfn = os.path.join(train_dur_path, tdur)
        tdur_arr = np.load(tfn)
        tdur_arr = add0(tdur_arr)
        np.save(tfn, tdur_arr)
    

    print(f"Processing valid duration...")
    for vdur in tqdm(os.listdir(valid_dur_path)):
        vfn = os.path.join(valid_dur_path, vdur)
        vdur_arr = np.load(vfn)
        vdur_arr = add0(vdur_arr)
        np.save(vfn, vdur_arr)
    print(f"Complete...")


if __name__ == '__main__':
    root_path = "/fsx/home/guoyingli/speech/TensorFlowTTS/dump_ljspeech"
    run(root_path)

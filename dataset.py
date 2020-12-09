import torch
import torchaudio
import pandas as pd


class MelSpecAudioDataset(torch.utils.data.Dataset):
    """Custom dataset containing text and audio."""

    def __init__(self, root='../input/dlaht4dataset/LJSpeech-1.1/', csv_path='metadata.csv', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file.
            root (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.csv = pd.read_csv(root+csv_path, sep='|', header=None)
        self.csv = self.csv.drop(columns=[1]).rename(columns={0:'filename', 2:'norm_text'})  # leave only normilized
        self.csv = self.csv.dropna().reset_index()
        self.transform = transform


    def __len__(self):
        return self.csv.shape[0]


    def __getitem__(self, idx):
        utt_name = self.root + 'wavs/' + self.csv.loc[idx, 'filename'] + '.wav'
        utt = torchaudio.load(utt_name)[0].squeeze()

        if self.transform:
            utt = self.transform(utt)

        sample = {'audio': utt}
        return sample



def tr_transform(wav, len_sample=15104):
    start = torch.randint(low=0, high=wav.size(0)-len_sample-1, size=(1,)).item()
    return wav[start:start+len_sample]

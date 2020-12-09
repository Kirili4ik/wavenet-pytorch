import torch
from tqdm import tqdm

import wandb



from my_utils import set_seed, count_parameters, MelSpectrogram, MelSpectrogramConfig
from dataset import MelSpecAudioDataset, tr_transform
from wavenet import WaveNet
from val_inf import validate, inference



if __name__ == '__main__':
    BATCH_SIZE = 5
    NUM_EPOCHS=7
    set_seed(21)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)


    ### Dataset and loaders
    my_dataset = MelSpecAudioDataset(csv_path='metadata.csv', transform=tr_transform)
    my_dataset_size = len(my_dataset)
    train_len = int(my_dataset_size * 0.8)
    val_len = my_dataset_size - train_len
    train_set, val_set = torch.utils.data.random_split(my_dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=1, pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=1, pin_memory=True)

    ### Featurizer
    featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
    ### Model
    model = WaveNet(hidden_ch=120, skip_ch=240, num_layers=30, mu=256)
    model = model.to(device)
    wandb.watch(model)
    print('num of model parameters', count_parameters(model))
    ### Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    ### Encoder and decoder for mu-law
    mu_law_encoder = torchaudio.transforms.MuLawEncoding(quantization_channels=256).to(device)
    mu_law_decoder = torchaudio.transforms.MuLawDecoding(quantization_channels=256).to(device)

    wandb.init(project='wavenet-pytorch')

    ### Train loop
    for i in tqdm(range(NUM_EPOCHS)):
        for el in train_loader:
            wav = el['audio'].to(device)
            melspec = featurizer(wav)
            wav = mu_law_encoder(wav).unsqueeze(1).type(torch.float)

            opt.zero_grad()

            new_wav = model(melspec, wav[:, :, :-1])

            new_wav = new_wav.transpose(-1, -2)

            ans = wav.type(torch.long)[:, 0, 1:]
            loss = F.cross_entropy(new_wav.reshape(-1, 256), ans.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            opt.step()
            wandb.log({'train_loss':loss.item()})

        #torch.save({'model_state_dict': model.state_dict()}, 'epoch_'+str(i))
        validate(model, val_loader, featurizer, mu_law_encoder)

    inference(model, val_loader, featurizer, my_law_encoder)

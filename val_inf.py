import torch


@torch.no_grad()
def validate(model, loader, featurizer, mu_law_encoder):
    total_loss = 0
    for el in loader:
        wav = el['audio'].to(device)
        melspec = featurizer(wav)
        wav = mu_law_encoder(wav).unsqueeze(1).type(torch.float)

        new_wav = model(melspec, wav[:, :, :-1])
        new_wav = new_wav.transpose(-1, -2)

        ans = wav.type(torch.long)[:, 0, 1:]
        loss = F.cross_entropy(new_wav.reshape(-1, 256), ans.reshape(-1))
        wandb.log({'val_item_loss':loss.item()})
        total_loss = total_loss + loss.item()
    wandb.log({'val_loss':total_loss})



@torch.no_grad()
def inference(model, loader, featurizer, mu_law_encoder):
    for el in loader:
        wav = el['audio'][:, :4096].to(device)
        melspec = featurizer(wav)
        wav = mu_law_encoder(wav).unsqueeze(1).type(torch.float)

        new_wav = model.inference(melspec)

        plt.plot(mu_law_decoder(wav.squeeze().detach().cpu()))
        plt.show()
        plt.plot(mu_law_decoder(new_wav.squeeze().detach().cpu()))
        plt.show

        break

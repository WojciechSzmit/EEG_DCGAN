import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from generator import Generator1D
from discriminator import Discriminator1D
from initialization import initialize_weights

from dataset import EEGDataset
from utils import load_eeg_data, plot_signal, plot_signal_to_image

def train(file_path, learning_rate, batch_size, noise_dim, num_epochs, features_disc, features_gen):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    eeg_data = load_eeg_data(file_path)
    SIGNAL_LENGTH = eeg_data.shape[1]

    dataset = EEGDataset(eeg_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    gen = Generator1D(noise_dim, channels_eeg=1, features_g=features_gen, signal_length=SIGNAL_LENGTH).to(device)
    disc = Discriminator1D(channels_eeg=1, features_d=features_disc, signal_length=SIGNAL_LENGTH).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, noise_dim, 1).to(device)
    writer_real = SummaryWriter("logs/real")
    writer_fake = SummaryWriter("logs/fake")
    step = 0

    gen.train()
    disc.train()

    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            noise = torch.randn(batch_size, noise_dim, 1).to(device)
            fake = gen(noise)

            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 1 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch {batch_idx+1}/{len(dataloader)} Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    real_signal = real[0]
                    fake_signal = fake[0]

                    plot_signal(real_signal, epoch+1, batch_idx, prefix='Real')
                    plot_signal(fake_signal, epoch+1, batch_idx, prefix='Fake')

                    real_image = plot_signal_to_image(real_signal, epoch+1, batch_idx, prefix='Real')
                    fake_image = plot_signal_to_image(fake_signal, epoch+1, batch_idx, prefix='Fake')

                    writer_real.add_image("Real_Signal", real_image, global_step=step)
                    writer_fake.add_image("Fake_Signal", fake_image, global_step=step)

                    real_signal_audio = real_signal.clamp(-1, 1)
                    fake_signal_audio = fake_signal.clamp(-1, 1)

                step += 1

        os.makedirs("models", exist_ok=True)
        torch.save(gen.state_dict(), f"models/generator_epoch_{epoch+1}.pth")
        torch.save(disc.state_dict(), f"models/discriminator_epoch_{epoch+1}.pth")

    writer_real.close()
    writer_fake.close()

if __name__ == "__main__":
    train()
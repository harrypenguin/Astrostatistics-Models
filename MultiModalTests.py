import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

###
num_lat = 0
###

for num_lat in range(2, 32, 2):
    print(f'Number of latent features: {num_lat}')
    # Load your data
    SFH_data = np.load('Latent data/SFHdata.npy')
    labels_data = np.load('Latent data/labels_vectors.npy')
    mass_sfr = np.load('Latent data/masspresentsfr.npy')

    # Step 1: Split the data into train, validation, and test sets
    SFH_train_val, SFH_test, labels_train_val, labels_test, mass_sfr_train_val, mass_sfr_test = train_test_split(
        SFH_data, labels_data, mass_sfr, test_size=0.2, random_state=42)

    SFH_train, SFH_val, labels_train, labels_val, mass_sfr_train, mass_sfr_val = train_test_split(
        SFH_train_val, labels_train_val, mass_sfr_train_val, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2 validation

    # Step 2: Convert the data splits to tensors
    SFH_train_tensor = torch.tensor(SFH_train, dtype=torch.float32)
    SFH_val_tensor = torch.tensor(SFH_val, dtype=torch.float32)
    SFH_test_tensor = torch.tensor(SFH_test, dtype=torch.float32)

    labels_train_tensor = torch.tensor(labels_train, dtype=torch.float32)
    labels_val_tensor = torch.tensor(labels_val, dtype=torch.float32)
    labels_test_tensor = torch.tensor(labels_test, dtype=torch.float32)

    mass_sfr_train_tensor = torch.tensor(mass_sfr_train, dtype=torch.float32)
    mass_sfr_val_tensor = torch.tensor(mass_sfr_val, dtype=torch.float32)
    mass_sfr_test_tensor = torch.tensor(mass_sfr_test, dtype=torch.float32)

    # Step 3: Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(SFH_train_tensor, labels_train_tensor, mass_sfr_train_tensor)
    val_dataset = TensorDataset(SFH_val_tensor, labels_val_tensor, mass_sfr_val_tensor)
    test_dataset = TensorDataset(SFH_test_tensor, labels_test_tensor, mass_sfr_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=200, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

    device = torch.device("mps")
    # Building an MLP model
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR

    class ConvAutoencoder(nn.Module):
        def __init__(self):
            super(ConvAutoencoder, self).__init__()
            
            # Encoder
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4, padding=1)
            self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4, padding=1)
            self.conv3 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, padding=1)
            self.conv4 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5, padding=1)
            self.pool = nn.MaxPool1d(2, 2)
            self.fc1 = nn.Linear(16 * 7, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, num_lat)  # Latent space
            
            # Decoder (Pathway 1)
            self.fc4 = nn.Linear(num_lat, 32)
            self.fc5 = nn.Linear(32, 64)
            self.fc6 = nn.Linear(64, 16 * 7)
            self.deconv1 = nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=4, stride=2)
            self.deconv2 = nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=4, stride=2)
            self.deconv3 = nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
            self.deconv4 = nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, stride=2)

            # Pathway 2: MLP for simulation type classification
            self.fc_sim1 = nn.Linear(num_lat, 100)
            self.fc_sim2 = nn.Linear(100, 100)
            self.fc_sim3 = nn.Linear(100, 10)
            
            # Pathway 3: MLP for SM and SFR prediction
            self.fc_sfr1 = nn.Linear(num_lat, 200)
            self.fc_sfr2 = nn.Linear(200, 200)
            self.fc_sfr3 = nn.Linear(200, 200) 
            self.fc_sfr4 = nn.Linear(200, 2)  # Predicting SM and SFR

        def encoder(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            x = self.pool(torch.relu(self.conv4(x)))
            x = x.view(-1, 16 * 7)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            encoded = self.fc3(x)
            return encoded

        def decoder(self, x):
            x = torch.relu(self.fc4(x))
            x = torch.relu(self.fc5(x))
            x = torch.relu(self.fc6(x))
            x = x.view(-1, 16, 7)
            x = torch.relu(self.deconv1(x))
            x = torch.relu(self.deconv2(x))
            x = torch.relu(self.deconv3(x))
            x = torch.relu(self.deconv4(x))
            return x

        def sim_type_classifier(self, x):
            x = torch.relu(self.fc_sim1(x))
            x = torch.relu(self.fc_sim2(x))
            sim_type_output = self.fc_sim3(x)
            return sim_type_output

        def sfr_predictor(self, x):
            x = torch.relu(self.fc_sfr1(x))
            x = torch.relu(self.fc_sfr2(x))
            x = torch.relu(self.fc_sfr3(x))
            sfr_output = self.fc_sfr4(x)
            return sfr_output
        
        def forward(self, x):
            latent = self.encoder(x)
            
            # Pathway 1: Reconstruct SFH
            sfh_output = self.decoder(latent)
            
            # Pathway 2: Classify sim type
            sim_type_output = self.sim_type_classifier(latent)
            
            # Pathway 3: Predict SM and SFR
            sfr_output = self.sfr_predictor(latent)
            
            return sfh_output, sim_type_output, sfr_output


    def compute_loss(sfh_output, sfh_target, sim_type_output, sim_type_target, mass_sfr_output, mass_sfr_target, w_reg, w_cl):
        mse_loss = nn.MSELoss()
        cross_entropy_loss = nn.CrossEntropyLoss()

        loss_sfh = mse_loss(sfh_output, sfh_target)
        loss_sim_type = cross_entropy_loss(sim_type_output, sim_type_target)
        loss_mass_sfr = mse_loss(mass_sfr_output, mass_sfr_target)

        total_loss = loss_sfh + w_reg * loss_mass_sfr + w_cl * loss_sim_type
        # print(f'Loss: {total_loss:.3f} | SFH Loss: {loss_sfh:.3f} | Sim Type Loss: {loss_sim_type:.3f} | Mass-SFR Loss: {loss_mass_sfr:.3f}')
        return total_loss, loss_sfh, loss_sim_type, loss_mass_sfr

    AE = ConvAutoencoder().to(device)
    optimizer = optim.Adam(AE.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
    num_epochs = 200
    losses = [] # To store training losses
    losses_sfh = []  # To store SFH losses
    losses_sim_type = []  # To store sim type losses
    losses_mass_sfr = []  # To store mass-sfr losses
    val_losses = []  # To store validation losses


    for epoch in range(num_epochs):
        total_loss = 0.0
        for data in train_loader:
            inputs, sim_labels, mass_sfr = data
            optimizer.zero_grad()
            inputs = inputs.to(device)
            sim_labels = sim_labels.to(device)
            mass_sfr = mass_sfr.to(device)
            sfh_output, sim_type_output, mass_sfr_output = AE(inputs.unsqueeze(1))
            sfh_output = sfh_output.squeeze(1)
            loss = compute_loss(sfh_output, inputs, sim_type_output, sim_labels, mass_sfr_output, mass_sfr, 1, 1)
            loss = loss[0]
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        total_loss /= len(train_loader)
        losses.append(total_loss)
        
        # Validation phase
        AE.eval()
        val_loss = 0.0
        sfh_loss = 0.0
        sim_type_loss = 0.0
        mass_sfr_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, sim_labels, mass_sfr = data
                inputs = inputs.to(device)
                sim_labels = sim_labels.to(device)
                mass_sfr = mass_sfr.to(device)
                sfh_output, sim_type_output, mass_sfr_output = AE(inputs.unsqueeze(1))
                loss, loss_sfh, loss_sim_type, loss_mass_sfr = compute_loss(sfh_output, inputs, sim_type_output, sim_labels, mass_sfr_output, mass_sfr, 1, 1)
                val_loss += loss.item()
                sfh_loss += loss_sfh.item()
                sim_type_loss += loss_sim_type.item()
                mass_sfr_loss += loss_mass_sfr.item()
                
        
        val_loss /= len(val_loader)
        sfh_loss /= len(val_loader)
        sim_type_loss /= len(val_loader)
        mass_sfr_loss /= len(val_loader)
        val_losses.append(val_loss)
        losses_sfh.append(sfh_loss)
        losses_sim_type.append(sim_type_loss)
        losses_mass_sfr.append(mass_sfr_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Training Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, SFH Loss: {sfh_loss:.4f}, Sim Type Loss: {sim_type_loss:.4f}, Mass-SFR Loss: {mass_sfr_loss:.4f}')
        scheduler.step()

    print('Training complete')
    import matplotlib.pyplot as plt
    plt.plot(losses_sfh, label='SFH Loss')
    plt.plot(losses_sim_type, label='Sim Type Loss')
    plt.plot(losses_mass_sfr, label='Mass-SFR Loss')
    # show labels
    plt.legend()
    plt.title('Validation Losses')
    torch.save(AE, f'MultiModal/{num_lat}loss{val_losses[-1]}.pth')
    np.save(f'MultiModal/{num_lat}_losses_sfh.npy', val_losses)
    np.save(f'MultiModal/{num_lat}_losses.npy', losses_sfh)
    np.save(f'MultiModal/{num_lat}_losses_sim.npy', losses_sim_type)
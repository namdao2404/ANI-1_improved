import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchani
import matplotlib.pyplot as plt
import torchani.data

model_path = '/global/scratch/users/namdao2404/c142_ugrad/proj/training_model_final.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def init_aev_computer():
    Rcr = 5.2
    Rca = 3.5
    EtaR = torch.tensor([16], dtype=torch.float, device=device)
    ShfR = torch.tensor([
        0.900000, 1.168750, 1.437500, 1.706250, 
        1.975000, 2.243750, 2.512500, 2.781250, 
        3.050000, 3.318750, 3.587500, 3.856250, 
        4.125000, 4.393750, 4.662500, 4.931250
    ], dtype=torch.float, device=device)


    EtaA = torch.tensor([8], dtype=torch.float, device=device)
    Zeta = torch.tensor([32], dtype=torch.float, device=device)
    ShfA = torch.tensor([0.90, 1.55, 2.20, 2.85], dtype=torch.float, device=device)
    ShfZ = torch.tensor([
        0.19634954, 0.58904862, 0.9817477, 1.37444680, 
        1.76714590, 2.15984490, 2.5525440, 2.94524300
    ], dtype=torch.float, device=device)

    num_species = 4
    aev_computer = torchani.AEVComputer(
        Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species
    )
    return aev_computer

aev_computer = init_aev_computer()
aev_dim = aev_computer.aev_length
print(aev_dim)

def load_ani_dataset(dspath):
    self_energies = torch.tensor([
        0.500607632585, -37.8302333826,
        -54.5680045287, -75.0362229210
    ], dtype=torch.float, device=device)
    energy_shifter = torchani.utils.EnergyShifter(None)
    species_order = ['H', 'C', 'N', 'O']

    dataset = torchani.data.load(dspath)
    dataset = dataset.subtract_self_energies(energy_shifter, species_order)
    dataset = dataset.species_to_indices(species_order)
    dataset = dataset.shuffle()
    return dataset

dataset = load_ani_dataset("/global/home/users/namdao2404/ani_gdb_s01_to_s04.h5")

class EnhancedANITrainer:
    def __init__(self, model, batch_size, learning_rate, epoch, 
                 l2=0.0,  # L2 regularization
                 lr_schedule='constant',  # 'constant', 'step', 'exponential', 'cosine'
                 lr_step_size=10,  # For step decay schedule
                 lr_gamma=0.1,  # Learning rate decay factor
                 gradient_clip_val=None,  # Maximum gradient norm
                 early_stop_patience=5,  # Epochs without improvement before stopping
                 add_noise=False,  # Data augmentation via coordinate noise
                 noise_level=0.01,  # Noise standard deviation
                 validation_split=0.2,  # Ensure proper validation split
                 cross_validation=False,  # Use k-fold cross validation
                 k_folds=5  # Number of folds for cross-validation
                ):
        self.model = model
        
        num_params = sum(item.numel() for item in model.parameters())
        print(f"{model.__class__.__name__} - Number of parameters: {num_params}")
        
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=l2  # L2 regularization
        )
        self.epoch = epoch
        self.lr_schedule = lr_schedule
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.gradient_clip_val = gradient_clip_val
        self.early_stop_patience = early_stop_patience
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.validation_split = validation_split
        self.cross_validation = cross_validation
        self.k_folds = k_folds
        
        # Initialize learning rate scheduler
        if lr_schedule == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=lr_step_size, gamma=lr_gamma
            )
        elif lr_schedule == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=lr_gamma
            )
        elif lr_schedule == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epoch
            )
        else:  # 'constant'
            self.scheduler = None
    
    def train(self, train_data, val_data, early_stop=True, draw_curve=True):
        self.model.train()
        
        # Initialize data loader
        print("Initialize training data...")
        train_data_loader = train_data.collate(self.batch_size).cache()
        
        # Loss function
        loss_func = nn.MSELoss()
        
        # Track losses
        train_loss_list = []
        val_loss_list = []
        lowest_val_loss = np.inf
        patience_counter = 0
        best_weights = None
        
        for i in tqdm(range(self.epoch), leave=True):
            train_epoch_loss = 0.0
            batch_count = 0
            
            for train_data_batch in train_data_loader:
                # Get batch data
                species = train_data_batch['species'].to(device)
                coords = train_data_batch['coordinates'].to(device)
                
                # Data augmentation by adding noise to coordinates
                if self.add_noise:
                    noise = torch.randn_like(coords) * self.noise_level
                    coords = coords + noise
                
                true_energies = train_data_batch['energies'].to(device).float()
                _, pred_energies = self.model((species, coords))
                
                # Compute loss
                batch_loss = loss_func(pred_energies, true_energies)
                
                # Gradient step
                self.optimizer.zero_grad()
                batch_loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )
                
                self.optimizer.step()
                
                batch_importance = len(species)
                train_epoch_loss += batch_loss.item() * batch_importance
                batch_count += 1
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Epoch {i+1}, Learning Rate: {current_lr:.6f}")
            
            # Evaluate on validation set
            val_epoch_loss = self.evaluate(val_data)
            
            # Record losses
            train_loss_list.append(train_epoch_loss / len(train_data))
            val_loss_list.append(val_epoch_loss)
            
            # Print progress
            print(f"Epoch {i+1}/{self.epoch}, Train Loss: {train_loss_list[-1]:.6f}, Val Loss: {val_epoch_loss:.6f}")
            
            # Early stopping with patience
            if early_stop:
                if val_epoch_loss < lowest_val_loss:
                    lowest_val_loss = val_epoch_loss
                    best_weights = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stop_patience:
                        print(f"Early stopping triggered after {i+1} epochs")
                        break
        
        if draw_curve:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
            ax.set_yscale("log")
            ax.plot(range(len(train_loss_list)), train_loss_list, label='Train')
            ax.plot(range(len(val_loss_list)), val_loss_list, label='Validation')
            ax.legend()
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss (MSE)")
        
        # Load best model
        if early_stop and best_weights is not None:
            self.model.load_state_dict(best_weights)
        
        return train_loss_list, val_loss_list
    
    def evaluate(self, data, draw_plot=False):
        self.model.eval()  # Set model to evaluation mode
        
        # Initialize data loader
        data_loader = data.collate(self.batch_size).cache()
        
        # Loss function
        loss_func = nn.MSELoss()
        total_loss = 0.0
        
        if draw_plot:
            true_energies_all = []
            pred_energies_all = []
            
        with torch.no_grad():
            for batch_data in data_loader:
                # Get batch data
                species = batch_data['species'].to(device)
                coords = batch_data['coordinates'].to(device)
                true_energies = batch_data['energies'].to(device).float()
                _, pred_energies = self.model((species, coords))
                
                # Compute loss
                batch_loss = loss_func(pred_energies, true_energies)

                batch_importance = len(species)
                total_loss += batch_loss.item() * batch_importance
                
                if draw_plot:
                    true_energies_all.append(true_energies.detach().cpu().numpy().flatten())
                    pred_energies_all.append(pred_energies.detach().cpu().numpy().flatten())

        if draw_plot:
            true_energies_all = np.concatenate(true_energies_all)
            pred_energies_all = np.concatenate(pred_energies_all)
            hartree2kcalmol = 627.51
            mae = np.mean(np.abs(true_energies_all - pred_energies_all)) * hartree2kcalmol
            fig, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
            ax.scatter(true_energies_all, pred_energies_all, label=f"MAE: {mae:.2f} kcal/mol", s=2)
            ax.set_xlabel("Ground Truth")
            ax.set_ylabel("Predicted")
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            vmin, vmax = min(xmin, ymin), max(xmax, ymax)
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            ax.plot([vmin, vmax], [vmin, vmax], color='red')
            ax.legend()
            
        return total_loss / len(data)


def torchani_kfold_validate(model_path, full_dataset, k=5, epochs=100):
    """K-fold CV using torchani's native dataset methods"""
    fold_losses = []
    best_models = []
    all_train_losses = []
    all_val_losses = []

    # Model construction function
    def build_model():
        net_H = AtomicNet()
        net_C = AtomicNet()
        net_N = AtomicNet()
        net_O = AtomicNet()
        ani_net = torchani.ANIModel([net_H, net_C, net_N, net_O])
        return nn.Sequential(aev_computer, ani_net).to(device)

    for fold in range(k):
        print(f"\n=== Fold {fold+1}/{k} ===")
        
        # Create fresh model instance with saved weights
        model = build_model()
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Shuffle and split using TorchANI's TransformableIterable methods
        shuffled = full_dataset.shuffle()
        train_data, val_data = shuffled.split(1 - 1/k, None)

        # Use TorchANI's collate and cache as usual
        train_loader = train_data.collate(8192).cache()
        val_loader = val_data.collate(8192).cache()

        # Initialize trainer with original anti-overfitting config
        trainer = EnhancedANITrainer(
            model=model,
            batch_size=8192,  # Match your actual batch size
            learning_rate=1e-3,
            epoch=epochs,
            l2=1e-4,
            lr_schedule='step',
            lr_step_size=15,
            lr_gamma=0.1,
            gradient_clip_val=0.5,
            add_noise=True,
            noise_level=0.005
        )
        
        # Train and get per-epoch losses
        train_loss, val_loss = trainer.train(
            train_data=train_loader,
            val_data=val_loader,
            early_stop=False,
            draw_curve=False  # Disable individual fold plots
        )
        
        # Store losses for final plot
        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)
        
        # Record final validation loss
        final_val_loss = val_loss[-1]
        fold_losses.append(final_val_loss)
        best_models.append(model.state_dict())
        print(f"Fold {fold+1} Final Val Loss: {final_val_loss:.6f}")

    # Plot and save all K-fold curves
    plt.figure(figsize=(10, 6))
    for fold_idx in range(k):
        plt.plot(all_train_losses[fold_idx], label=f'Fold {fold_idx+1} Train', alpha=0.7)
        plt.plot(all_val_losses[fold_idx], label=f'Fold {fold_idx+1} Val', linestyle='--', alpha=0.7)
    
    plt.title(f'K-Fold Cross-Validation (k={k}) Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (log scale)')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save high-quality figure
    plt.savefig(f'kfold_cv_curves_k{k}_epochs{epochs}.png', dpi=300, bbox_inches='tight')
    plt.close()

    return np.mean(fold_losses), np.std(fold_losses), fold_losses, best_models

# CORRECTED USAGE: Use original dataset before any splits
mean_loss, std_loss, all_losses, fold_models = torchani_kfold_validate(
    model_path='trained_model_final.pth',
    full_dataset=dataset,  # Use original unsplit dataset
    k=5,
    epochs=100
)


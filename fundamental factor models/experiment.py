from factor_model import *
from data_factory import *
import torch.optim as optim
import os
import matplotlib.pyplot as plt
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class Experiment:
    def __init__(self, args):
        self.args = args
        assets_of_interest = self.args.assets_of_interest
        start_date = self.args.start_date
        end_date = self.args.end_date
        target_col = self.args.target_col
        data_factory = DataFactory(assets_of_interest, start_date, end_date, target_col = target_col)
        df_list_aligned = data_factory.get_data()
        numerics, onehots = data_factory.df_to_np(df_list_aligned)
        self.continuous_vars, self.categorical_vars, self.scaler = data_factory.temporal_split(numerics, onehots) # need the scaler for inverse transform
        self.train_loader = DataLoader(FactorCSDataset(self.continuous_vars["train"], self.categorical_vars["train"]), batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        self.val_loader = DataLoader(FactorCSDataset(self.continuous_vars["val"], self.categorical_vars["val"]), batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        self.test_loader = DataLoader(FactorCSDataset(self.continuous_vars["test"], self.categorical_vars["test"]), batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
    
    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_model(self):
        model = LinearFactorModel(n_factors=self.args.num_factors)
        return model
    
    def get_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    


    def run(self):
        device = self.get_device()
        model = self.get_model()
        criterion = self.get_criterion()
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        print("start training")
        early_stopper = EarlyStopper(patience=self.args.patience, min_delta=self.args.min_delta)
        # for epoch in range(self.args.epochs):
        #     model.train()
        #     running_loss = 0.0
        #     for batch_idx, (exposures, returns) in enumerate(self.train_loader):
        #         '''
        #         input: exposure matrix observed at t-1 and the return at t
        #         output: the return at t; the model itself can be comprehend as the "factor return"
        #         '''
        #         exposures = exposures.to(device)
        #         returns = returns.to(device)


        #         optimizer.zero_grad()
        #         r_hat = model(exposures)
        #         # print("the value of r_hat: ", r_hat)
        #         # print("-----------------------")
        #         loss  = criterion(r_hat, returns)
        #         loss.backward()
        #         optimizer.step()

        #         running_loss += loss.item() * exposures.size(0)   # sum over batches

        #     train_loss = running_loss / len(self.train_loader.dataset)
        #     print(f"Epoch {epoch:02d}  |  train MSE {train_loss:.5f}")
        #     val_loss = self.evaluate(model, criterion, device)
        #     print(f"Epoch {epoch:02d}  |  val MSE {val_loss:.5f}")
        #     if early_stopper.early_stop(val_loss):
        #         print(f"Early stopping at epoch {epoch}")
        #         # save the model
        #         os.makedirs("checkpoints", exist_ok=True)
        #         torch.save(model.state_dict(), f"checkpoints/best_model.pth")
        #         break
        
        self.test(model, criterion, device)

    def evaluate(self, model, criterion, device):
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_idx, (exposures, returns) in enumerate(self.val_loader):
                exposures = exposures.to(device)
                returns = returns.to(device)
                r_hat = model(exposures)
                loss = criterion(r_hat, returns)
                running_loss += loss.item() * exposures.size(0)
        val_loss = running_loss / len(self.val_loader.dataset)
        return val_loss
    
    def test(self, model, criterion, device):
        model.eval()
        running_loss = 0.0
        # load the best model
        model.load_state_dict(torch.load(f"checkpoints/best_model.pth"))
        all_predictions = []
        all_returns = []
        with torch.no_grad():
            for batch_idx, (exposures, returns) in enumerate(self.test_loader):
                exposures = exposures.to(device)
                returns = returns.to(device)
                r_hat = model(exposures)
                loss = criterion(r_hat, returns)
                running_loss += loss.item() * exposures.size(0)
                # note that the test is never shuffled, so we can use the order of the test set
                all_predictions.append(r_hat.cpu().numpy())
                all_returns.append(returns.cpu().numpy())
        test_loss = running_loss / len(self.test_loader.dataset)
        print(f"test MSE {test_loss:.5f}")
        # save the predictions and returns
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_returns = np.concatenate(all_returns, axis=0)
        print(all_predictions.shape)
        print(all_returns.shape)
        mu_r    = self.scaler.mean_[-1]
        sigma_r = self.scaler.scale_[-1]
        all_predictions = all_predictions * sigma_r + mu_r
        all_returns = all_returns * sigma_r + mu_r
        # plot the predictions and returns, one for each asset
        for i in range(all_predictions.shape[1]):
            plt.plot(all_predictions[:, i], label="predictions")
            plt.plot(all_returns[:, i], label="returns")
            plt.legend()
            plt.show()


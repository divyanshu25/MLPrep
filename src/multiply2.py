import torch
import torch.utils.data
import wandb
from tqdm import tqdm
import uuid
import os

class Dataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=1000):
        self.x = 2 * torch.randn(n_samples, 1) # shape: (n_samples, 1)
        self.y = 2 * self.x # shape: (n_samples, 1)

    def __len__(self):
        return len(self.x) # number of samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx] # return the x and y for the given index

# Define the model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 10)
        self.fc2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.fc2(self.fc1(x))

class Trainer:
    def __init__(self):
        self.model = Model()
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(Dataset(), [0.8, 0.2])
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=10, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=10, shuffle=True)
        self.wandb = wandb.init(project="multiply2")

    def train(self, epochs = 3):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            print("Training...")
            train_loss_per_epoch = 0
            test_loss_per_epoch = 0
            for x, y in tqdm(self.train_loader, desc="Training"):
                y_pred = self.model(x) # this will give the prdicted value for the given x
                loss = self.loss_function(y_pred, y) # compute the loss between the predicted and the actual value
                self.optimizer.zero_grad() # zero the gradients to remove the gradients accumulated from the previous iteration
                loss.backward() # compute the gradients of the loss with respect to the model parameters
                self.optimizer.step() # update the model parameters based on the gradients
                # self.wandb.log({"loss": loss.item()}) # log the loss to wandb
                # time.sleep(0.1)
                train_loss_per_epoch += loss.item()
            
            self.wandb.log({"train_loss": train_loss_per_epoch / len(self.train_loader), "epoch": epoch+1})
            
            if epoch % 3 == 0:
                print("Testing...")
                self.model.eval()
                for x, y in tqdm(self.test_loader, desc="Testing"):
                    with torch.no_grad(): # Do this to avoid computing the gradients
                        y_pred = self.model(x) # this will give the prdicted value for the given x
                        loss = self.loss_function(y_pred, y) # compute the loss between the predicted and the actual value
                        # self.wandb.log({"test_loss": loss.item()}) # log the loss to wandb
                        test_loss_per_epoch += loss.item()
                        # time.sleep(0.1)
                print("Testing complete")
                self.wandb.log({"test_loss": test_loss_per_epoch / len(self.test_loader), "epoch": epoch+1})
                self.model.train()
        

        print("Training complete")
        # save the model for future use
        os.makedirs("models", exist_ok=True)
        torch.save(self.model.state_dict(), f"models/multiply2_model_version_{str(uuid.uuid4())}.pth")

    def run_model_inference(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        total_loss = 0
        for x, y in tqdm(self.test_loader, desc="Testing"):
            with torch.no_grad(): # Do this to avoid computing the gradients
                y_pred = self.model(x) # this will give the prdicted value for the given x
                loss = self.loss_function(y_pred, y) # compute the loss between the predicted and the actual value
                total_loss += loss.item()
                for y, y_pred, x in zip(y, y_pred, x):
                    print(f"Actual: {y.item()}, Predicted: {y_pred.item()}, Input: {x.item()}\n")
        self.model.train()
        print(f"Total loss: {total_loss}")

if __name__ == "__main__":
    # trainer = Trainer()
    # trainer.train(epochs=10)
    # wandb.finish()
    trainer = Trainer()
    trainer.run_model_inference(model_path="models/multiply2_model_version_a5cd7338-edf9-4003-bc90-59e448667799.pth")

import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import time

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self,input_dim:int,output_dim:int):
        super().__init__()

        # attach different layers/activations to self
        self.fc1 = torch.nn.Linear(input_dim,20)
        self.fc_out = torch.nn.Linear(20,output_dim)
        self.activation_fn = torch.nn.Tanh()
        self.activation_fn1 = torch.nn.LogSigmoid()

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        result=self.fc1(x)
        result=self.activation_fn(result)
        result=self.activation_fn1(result)
        logits=self.fc_out(result)
        return logits

def fit_MLP1(x_train,x_test,y_train,y_test):

    #----- convert features and labels to tensors -----#
    x_train_t = torch.from_numpy(x_train).float()
    x_test_t = torch.from_numpy(x_test).float()

    y_train_t = torch.from_numpy(y_train).long()
    y_test_t = torch.from_numpy(y_test).long()

    #----- compute mean and std of training images -----#
    x_std, x_mean = torch.std_mean(x_train_t)

    #----- standardize images -----#
    x_train_t = (x_train_t - x_mean) / (x_std + 1e-6)
    x_test_t = (x_test_t - x_mean) / (x_std + 1e-6)

    #----- convert lables to one-hot encoding -----#
    y_train_t = torch.nn.functional.one_hot(y_train_t, num_classes=4).float()
    y_test_t = torch.nn.functional.one_hot(y_test_t, num_classes=4).float()

    #----- setup dataset -----#
    train_ds = TensorDataset(x_train_t, y_train_t)
    test_ds = TensorDataset(x_test_t, y_test_t)

    #----- setup a Weighted Random Sampler to account for Lable imbalance -----#
    class_counts = [np.size(np.where(y_train==0)),np.size(np.where(y_train==1))]
    num_samples = sum(class_counts)
    class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
    weights = [class_weights[y_train[i]] for i in range(int(num_samples))]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(weights),num_samples=int(num_samples), replacement=True)

    #----- setup train and test loader -----#
    train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=True)

    #----- generate model -----#
    model = MultiLayerPerceptron(x_train.shape[1],4)

    #----- define loss function -----#
    loss_fn = torch.nn.CrossEntropyLoss()

    #----- Setup the optimizer (adaptive learning rate method) -----#
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    #----- Perform training and testing -----#

    for epoch in range(50):
        # reset statistics trackers
        train_loss_cum = 0.0
        acc_cum = 0.0
        num_samples_epoch = 0
        t = time.time()

        # Go once through the training dataset (-> epoch)
        for x_batch, y_batch in train_loader:
            # zero grads and put model into train mode
            optim.zero_grad()
            model.train()

            # forward pass
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)

            # backward pass and gradient step
            loss.backward()
            optim.step()

            # keep track of train stats
            num_samples_batch = x_batch.shape[0]
            num_samples_epoch += num_samples_batch
            train_loss_cum += loss * num_samples_batch
            acc_cum += accuracy(logits, y_batch) * num_samples_batch

        # average the accumulated statistics
        avg_train_loss = train_loss_cum / num_samples_epoch
        avg_acc = acc_cum / num_samples_epoch
        test_acc = evaluate(model,test_loader)
        epoch_duration = time.time() - t

        # print some infos
        print(f'Epoch {epoch} | Train loss: {train_loss_cum.item():.4f} | '
              f' Train accuracy: {avg_acc.item():.4f} | Test accuracy: {test_acc.item():.4f} |'
              f' Duration {epoch_duration:.2f} sec')

    return 0


def accuracy(logits:torch.Tensor,label:torch.tensor) -> torch.Tensor:
    # computes the classification accuracy
    correct_label = torch.argmax(logits, axis=-1) == torch.argmax(label, axis=-1)
    assert correct_label.shape == (logits.shape[0],)
    acc = torch.mean(correct_label.float())
    assert 0. <= acc <= 1.
    return acc

def evaluate(model: torch.nn.Module,test_loader) -> torch.Tensor:
  # goes through the test dataset and computes the test accuracy
  model.eval()  # bring the model into eval mode
  with torch.no_grad():
    acc_cum = 0.0
    num_eval_samples = 0
    for x_batch_test, y_label_test in test_loader:
      batch_size = x_batch_test.shape[0]
      num_eval_samples += batch_size
      acc_cum += accuracy(model(x_batch_test), y_label_test) * batch_size
    avg_acc = acc_cum / num_eval_samples
    assert 0 <= avg_acc <= 1
    return avg_acc
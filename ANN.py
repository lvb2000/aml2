import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import time
from sklearn.metrics import f1_score

class MultiLayerPerceptron1(torch.nn.Module):

    def __init__(self,input_dim:int,output_dim:int):
        super().__init__()

        # attach different layers/activations to self
        self.fc1 = torch.nn.Linear(input_dim,120)
        self.fc_out = torch.nn.Linear(120,output_dim)
        self.activation_fn = torch.nn.Sigmoid()
        self.activation_fn1 = torch.nn.LogSigmoid()

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        result=self.fc1(x)
        result=self.activation_fn(result)
        result=self.activation_fn1(result)
        logits=self.fc_out(result)
        return logits

class MultiLayerPerceptron2(torch.nn.Module):

    def __init__(self,input_dim:int,output_dim:int):
        super().__init__()

        # attach different layers/activations to self
        self.fc1 = torch.nn.Linear(input_dim,120)
        self.fc_out = torch.nn.Linear(120,output_dim)
        self.activation_fn = torch.nn.Sigmoid()
        self.activation_fn1 = torch.nn.LogSigmoid()

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        result=self.fc1(x)
        result=self.activation_fn(result)
        result=self.activation_fn1(result)
        logits=self.fc_out(result)
        return logits

class MultiLayerPerceptron3(torch.nn.Module):

    def __init__(self,input_dim:int,output_dim:int):
        super().__init__()

        # attach different layers/activations to self
        self.fc1 = torch.nn.Linear(input_dim,60)
        self.fc_out = torch.nn.Linear(60,output_dim)
        self.activation_fn = torch.nn.Sigmoid()

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        result=self.fc1(x)
        result=self.activation_fn(result)
        logits=self.fc_out(result)
        return logits

class MultiLayerPerceptron4(torch.nn.Module):

    def __init__(self,input_dim:int,output_dim:int):
        super().__init__()

        # attach different layers/activations to self
        self.fc1 = torch.nn.Linear(input_dim,30)
        self.fc_out = torch.nn.Linear(30,output_dim)
        self.activation_fn1 = torch.nn.LogSigmoid()

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        result=self.fc1(x)
        result=self.activation_fn1(result)
        logits=self.fc_out(result)
        return logits


class MLP():

    def __init__(self,num_classes:int,epochs:int,predict=False, MLP='MLP1'):
        # define general parameters
        self.num_classes = num_classes
        self.epochs = epochs
        self.predict = predict
        if MLP != 'MLP1' and MLP != 'MLP2' and MLP != 'MLP3' and MLP != 'MLP4':
            assert ('no such model exists')
        self.MLP = MLP

    def load_data(self,x_train,x_test,y_train,y_test):
        # ----- convert features and labels to tensors -----#
        x_train_t = torch.from_numpy(x_train).float()
        y_train_t = torch.from_numpy(y_train).long()

        if not self.predict:
            x_test_t = torch.from_numpy(x_test).float()
            y_test_t = torch.from_numpy(y_test).long()

        # ----- compute mean and std of training images -----#
        x_std, x_mean = torch.std_mean(x_train_t)

        # ----- standardize-----#
        x_train_t = (x_train_t - x_mean) / (x_std + 1e-6)
        if not self.predict:
            x_test_t = (x_test_t - x_mean) / (x_std + 1e-6)

        # ----- convert lables to one-hot encoding -----#
        y_train_t = torch.nn.functional.one_hot(y_train_t, num_classes=self.num_classes).float()
        if not self.predict:
            y_test_t = torch.nn.functional.one_hot(y_test_t, num_classes=self.num_classes).float()

        # ----- setup dataset -----#
        train_ds = TensorDataset(x_train_t, y_train_t)
        if not self.predict:
            test_ds = TensorDataset(x_test_t, y_test_t)

        # ----- setup a Weighted Random Sampler to account for lable imbalance -----#
        class_counts = 0
        if self.num_classes == 2:
            class_counts = [np.size(np.where(y_train == 0)), np.size(np.where(y_train == 1))]
        elif self.num_classes == 3:
            class_counts = [np.size(np.where(y_train == 0)), np.size(np.where(y_train == 1)),
                            np.size(np.where(y_train == 2))]
        elif self.num_classes == 4:
            class_counts = [np.size(np.where(y_train == 0)), np.size(np.where(y_train == 1)),
                            np.size(np.where(y_train == 2)), np.size(np.where(y_train == 3))]
        else:
            assert("Number of classes out of bounds")

        num_samples = sum(class_counts)
        class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
        weights = [class_weights[y_train[i]] for i in range(int(num_samples))]
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=int(num_samples),
                                        replacement=True)

        # ----- setup train and test loader -----#
        self.train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler)
        if not self.predict:
            self.test_loader = DataLoader(test_ds, batch_size=512, shuffle=True)

    def load_model(self,x_train):
        # ----- generate model -----#
        if self.MLP == 'MLP1':
            self.model = MultiLayerPerceptron1(x_train.shape[1], self.num_classes)
        elif self.MLP == 'MLP2':
            self.model = MultiLayerPerceptron2(x_train.shape[1], self.num_classes)
        elif self.MLP == 'MLP3':
            self.model = MultiLayerPerceptron3(x_train.shape[1], self.num_classes)
        elif self.MLP == 'MLP4':
            self.model = MultiLayerPerceptron4(x_train.shape[1], self.num_classes)

        # ----- define loss function -----#
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # ----- Setup the optimizer (adaptive learning rate method) -----#
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    def fit_one_epoch(self):
        for x_batch, y_batch in self.train_loader:
            # zero grads and put model into train mode
            self.optim.zero_grad()
            self.model.train()

            # forward pass
            logits = self.model(x_batch)
            loss = self.loss_fn(logits, y_batch)

            # backward pass and gradient step
            loss.backward()
            self.optim.step()

            # keep track of train stats
            num_samples_batch = x_batch.shape[0]
            self.num_samples_epoch += num_samples_batch
            self.train_loss_cum += loss * num_samples_batch
            self.acc_cum += self.accuracy(logits, y_batch) * num_samples_batch

        return

    def train(self,x_train,y_train,x_test=0,y_test=0):

        self.load_data(x_train,x_test,y_train,y_test)

        self.load_model(x_train)

        for epoch in range(self.epochs):
            # reset statistics trackers
            self.train_loss_cum = 0.0
            self.acc_cum = 0.0
            self.num_samples_epoch = 0
            t = time.time()

            self.fit_one_epoch()

            # average the accumulated statistics
            avg_train_loss = self.train_loss_cum / self.num_samples_epoch
            avg_acc = self.acc_cum / self.num_samples_epoch
            test_acc = torch.tensor(0.0)
            f1=-1
            if not self.predict:
                test_acc = self.evaluate(self.model, self.test_loader)
                f1 = self.f1(self.model, self.test_loader)
            epoch_duration = time.time() - t

            # print some infos
            print(f'Epoch {epoch} | Train loss: {self.train_loss_cum.item():.4f} | '
                  f' Train accuracy: {avg_acc.item():.4f} | Test accuracy: {test_acc.item():.4f} |'
                  f' Duration {epoch_duration:.2f} sec |'
                  f' F1 score {f1:.2f}')

    def accuracy(self,logits: torch.Tensor, label: torch.tensor) -> torch.Tensor:
        # computes the classification accuracy
        correct_label = torch.argmax(logits, axis=-1) == torch.argmax(label, axis=-1)
        assert correct_label.shape == (logits.shape[0],)
        acc = torch.mean(correct_label.float())
        assert 0. <= acc <= 1.
        return acc

    def evaluate(self,model: torch.nn.Module, test_loader) -> torch.Tensor:
        # goes through the test dataset and computes the test accuracy
        model.eval()  # bring the model into eval mode
        with torch.no_grad():
            acc_cum = 0.0
            num_eval_samples = 0
            for x_batch_test, y_label_test in test_loader:
                batch_size = x_batch_test.shape[0]
                num_eval_samples += batch_size
                acc_cum += self.accuracy(model(x_batch_test), y_label_test) * batch_size
            avg_acc = acc_cum / num_eval_samples
            assert 0 <= avg_acc <= 1
            return avg_acc

    def f1(self,model: torch.nn.Module, test_loader) -> torch.Tensor:
        # goes through the test dataset and computes the test accuracy
        model.eval()  # bring the model into eval mode
        with torch.no_grad():
            f1_cum = 0.0
            num_eval_batches = 0
            for x_batch_test, y_label_test in test_loader:
                num_eval_batches += 1
                f1_cum += f1_score(torch.argmax(self.model(x_batch_test), axis=-1).numpy(),torch.argmax(y_label_test, dim=1).numpy(),average='micro')
            avg_f1 = f1_cum / num_eval_batches
            assert 0 <= avg_f1 <= 1
            return avg_f1

    def predict_test_set(self,x_test):
        if not self.predict:
            self.prediction = 0

        x_test_t = torch.from_numpy(x_test).float()
        x_std, x_mean = torch.std_mean(x_test_t)
        x_test_t = (x_test_t - x_mean) / (x_std + 1e-6)

        self.model.eval()
        with torch.no_grad():
            self.prediction = torch.argmax(self.model(x_test_t), axis=-1).numpy()



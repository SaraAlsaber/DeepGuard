
from flask import Flask, request, render_template
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



app = Flask(__name__)

class HybridModel(nn.Module):
    def __init__(self, num_classes = 3):
        super(HybridModel, self).__init__()

        # Convolutional block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(128)
        self.mp1 = nn.MaxPool1d(kernel_size=1)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.mp2 = nn.MaxPool1d(kernel_size=1)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(32)
        self.mp3 = nn.MaxPool1d(kernel_size=1)

        self.conv4 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(16)
        self.mp4 = nn.MaxPool1d(kernel_size=1)

        # GRU layers
        self.gru1 = nn.GRU(input_size=1, hidden_size=128, num_layers=3, batch_first=True)
        self.gru2 = nn.GRU(input_size=128, hidden_size=64, num_layers=3, batch_first=True)
        self.gru3 = nn.GRU(input_size=64, hidden_size=32, num_layers=3, batch_first=True)

        # Feedforward network
        self.fc1 = nn.Linear(528 + 32, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input_x):
        # Convolutional block
        x = torch.tanh(self.bn1(self.conv1(input_x))) #activation function
        x = self.mp1(x)

        x = torch.tanh(self.bn2(self.conv2(x)))
        x = self.mp2(x)

        x = torch.tanh(self.bn3(self.conv3(x)))
        x = self.mp3(x)

        x = torch.tanh(self.bn4(self.conv4(x)))
        x = self.mp4(x)

        # Flatten output of convolutional block
        x1 = x.view(-1, x.shape[1] * x.shape[2])

        # GRU layers
        input_x = input_x.transpose(1, 2)  # Transpose input for GRU layers (shuffeling)
        x, _ = self.gru1(input_x)
        x, _ = self.gru2(x)
        x2, _ = self.gru3(x)


        # Reshape x2 to (batch_size, 32)
        x2 = x2[:, -1, :]  # Take the last hidden state

        # Concatenate convolutional and GRU outputs
        x = torch.cat((torch.flatten(x1, start_dim=1), x2), dim=1)

        # Feedforward network
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x #what is x?



class MyDataset(Dataset):
    def __init__(self, features, labels, range_=(1, 4)):
        self.features = features
        self.labels = labels
        self.range = range_

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return {
            'feature': torch.tensor(feature, dtype=torch.float),
            'label': torch.tensor([1 if i == label else 0 for i in range(self.range[0], self.range[1])],
                                  dtype=torch.float)
        }


global normalC, dosC, commandInjC

def update_counter(prediction_output):
    global normalC, dosC, reconnC, backdoorC, commandInjC
    if prediction_output == 0:
        normalC += 1
    elif prediction_output == 1:
        dosC += 1
    elif prediction_output == 2:
        commandInjC += 1



@app.route('/')
def upload_file():
   return render_template('index.html') #for deepguard website


@app.route('/predict', methods=['POST'])
def predict():
    global normalC, dosC, commandInjC
    uploaded_file = request.files['data_file']
    if uploaded_file.filename != '':
        normalC, dosC, commandInjC= 0,0,0
        data = pd.read_csv(uploaded_file)

        data = data.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
        test_features = np.array(data.iloc[:, 0:41].values, dtype="float")
        test_labels = data.iloc[:, -1].values

        batch_size = 8192
        test_dataset = MyDataset(test_features, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        device = "cpu"
        model = HybridModel(num_classes=3).to(device)
        model.load_state_dict(torch.load('CNN_GRU-3class2.pth', map_location=torch.device(device)))
        model.eval()

        test_preds = []

        with torch.no_grad():
            for batch in test_loader:
                outputs = model(batch['feature'].unsqueeze(1).to(device))
                preds = torch.argmax(outputs, 1)
                test_preds.extend(preds.cpu().numpy())

        for output in test_preds:
            update_counter(output.item())


        sum = normalC + dosC + commandInjC
        normalP = round((normalC / sum), 4)
        dosP = round((dosC / sum), 4)
        commandInjP = round((commandInjC / sum), 4)

        return render_template('index.html', normalC=normalC, dosC=dosC, commandInjC=commandInjC, normalP=normalP, dosP=dosP, commandInjP=commandInjP)
    else:
        return render_template('index.html', message="no file")


if __name__ == '__main__':
  app.run(debug=True)

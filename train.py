import torch
import torch.nn.functional as F

import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TopKPooling, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        embed_dim = 128
        input_dim = 52706   # df.item_id.max()

        self.item_embedding = torch.nn.Embedding(num_embeddings=input_dim + 10, embedding_dim=embed_dim)
        self.conv1 = SAGEConv(embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x)
        x = x.squeeze(1)
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = gap(x, batch)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = gap(x, batch)
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x3 = gap(x, batch)
        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)
        return x


class YooChooseBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(YooChooseBinaryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['yoochoose_click_binary_1M_sess.dataset']

    def download(self):
        pass

    def process(self):
        df = pd.read_csv('data/yoochoose-clicks.dat', header=None)
        df.columns = ['session_id', 'timestamp', 'item_id', 'category']

        buy_df = pd.read_csv('data/yoochoose-buys.dat', header=None)
        buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']

        item_encoder = LabelEncoder()
        df['item_id'] = item_encoder.fit_transform(df.item_id)
        df.head()

        sampled_session_id = np.random.choice(df.session_id.unique(), 100000, replace=False)
        df = df.loc[df.session_id.isin(sampled_session_id)]
        df.nunique()

        df['label'] = df.session_id.isin(buy_df.session_id)
        df.head()

        data_list = []

        # process by session_id
        grouped = df.groupby('session_id')
        for session_id, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id']].sort_values(
                'sess_item_id').item_id.drop_duplicates().values

            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]

            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            x = node_features
            y = torch.FloatTensor([group.label.values[0]])

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def train(dataset, train_loader):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data
        # print('data',data)
        optimizer.zero_grad()
        output = model(data)
        label = data.y
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(dataset)


def evalute(loader, model):
    model.eval()

    prediction = []
    labels = []

    with torch.no_grad():
        for data in loader:
            data = data#.to(device)
            pred = model(data)#.detach().cpu().numpy()

            label = data.y#.detach().cpu().numpy()
            prediction.append(pred)
            labels.append(label)
    prediction =  np.hstack(prediction)
    labels = np.hstack(labels)

    return roc_auc_score(labels,prediction)


dataset = YooChooseBinaryDataset(root='data/')
train_loader = DataLoader(dataset, batch_size=64)

model = Net()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
crit = torch.nn.BCELoss()

for epoch in range(10):
    print('epoch:', epoch)
    loss = train(dataset, train_loader)
    print(loss)

for epoch in range(1):
    roc_auc_score = evalute(train_loader, model)
    print('roc_auc_score',roc_auc_score)

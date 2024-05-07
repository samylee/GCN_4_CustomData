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


class Net(torch.nn.Module):  # 针对图进行分类任务
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
        x, edge_index, batch = data.x, data.edge_index, data.batch  # x:n*1,其中每个图里点的个数是不同的
        # print(x)
        x = self.item_embedding(x)  # n*1*128 特征编码后的结果
        # print('item_embedding',x.shape)
        x = x.squeeze(1)  # n*128
        # print('squeeze',x.shape)
        x = F.relu(self.conv1(x, edge_index))  # n*128
        # print('conv1',x.shape)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)  # pool之后得到 n*0.8个点
        # print('self.pool1',x.shape)
        # print('self.pool1',edge_index)
        # print('self.pool1',batch)
        # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x1 = gap(x, batch)
        # print('gmp',gmp(x, batch).shape) # batch*128
        # print('cat',x1.shape) # batch*256
        x = F.relu(self.conv2(x, edge_index))
        # print('conv2',x.shape)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        # print('pool2',x.shape)
        # print('pool2',edge_index)
        # print('pool2',batch)
        # x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x2 = gap(x, batch)
        # print('x2',x2.shape)
        x = F.relu(self.conv3(x, edge_index))
        # print('conv3',x.shape)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        # print('pool3',x.shape)
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x3 = gap(x, batch)
        # print('x3',x3.shape)# batch * 256
        x = x1 + x2 + x3  # 获取不同尺度的全局特征

        x = self.lin1(x)
        # print('lin1',x.shape)
        x = self.act1(x)
        x = self.lin2(x)
        # print('lin2',x.shape)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)  # batch个结果
        # print('sigmoid',x.shape)
        return x


class YooChooseBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(YooChooseBinaryDataset, self).__init__(root, transform, pre_transform)  # transform就是数据增强，对每一个数据都执行
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):  # 检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件
        # 如有文件不存在，则调用download()方法执行原始文件下载
        return []

    @property
    def processed_file_names(self):  # 检查self.processed_dir目录下是否存在self.processed_file_names属性方法返回的所有文件，没有就会走process
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

        print(df.item_id.max())

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
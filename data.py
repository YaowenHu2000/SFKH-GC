import torch
print("Start")

def get_dataset(dataset,device):
    if dataset in {"pubmed", "corafull", "computer", "photo", "cs", "physics","cora", "citeseer"}:
        file_path = "dataset/"+dataset+".pt"

        data_list = torch.load(file_path)
        adj = data_list[0]
        features = data_list[1]
        labels = data_list[2]


        idx_train = torch.tensor(data_list[3],dtype=torch.long)
        idx_val = torch.tensor(data_list[4],dtype=torch.long)
        idx_test = torch.tensor(data_list[5],dtype=torch.long)

    elif dataset in {"aminer", "reddit", "Amazon2M"}:
        file_path = './dataset/'+dataset+'.pt'
        data = torch.load(file_path)
        values = torch.ones(data.edge_index.size(1))
        adj = torch.sparse.FloatTensor(data.edge_index, values=values)
        features = data.x
        labels = data.y
        idx_train = data.train_mask.nonzero(as_tuple=False).view(-1)
        idx_val = data.val_mask.nonzero(as_tuple=False).view(-1)
        idx_test = data.test_mask.nonzero(as_tuple=False).view(-1)

    elif dataset in {"ogbn_arxiv",  "ogbn_products"}:
        print(dataset)

        file_path = './dataset/' + dataset + '.pt'
        data = torch.load(file_path)

        # 将每个张量移动到指定的设备
        features = data['node_features']
        labels = data['node_labels']
        labels = labels.squeeze()
        print("-------")
        print(labels.shape)
        print("-------")
        edge_index = data['edge_index']
        idx_train = data['train_idx']
        idx_val = data['val_idx']
        idx_test = data['test_idx']
        num_nodes = max(edge_index.max().item() + 1, labels.size(0))
        # 创建邻接矩阵
        values = torch.ones(edge_index.size(1))
        adj = torch.sparse.FloatTensor(edge_index, values, (num_nodes, num_nodes))

        print(idx_train.shape)
        print(labels.shape)
    else:
        # 处理其他情况或抛出异常
        raise ValueError(f"Unsupported dataset: {dataset}")
    return adj, features, labels, idx_train, idx_val, idx_test





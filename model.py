
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks,input_dim, num_classes=10):
        super(ResNet, self).__init__()
        self.input_dim = input_dim
        self.final_ln1 = nn.LayerNorm(self.input_dim)

        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 1, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 1, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 1, num_blocks[3], stride=1)
        self.linear = nn.Linear(358 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.final_ln1(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(num_class,input_dim):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_class,input_dim =input_dim)


class SFKH_GC_MLP(nn.Module):
    def __init__(self,
                 hops,
                 n_class,
                 input_dim,
                 n_layers=6,
                 hidden_dim=256,
                 dropout_rate1=0.5,
                 dropout_rate2=0.2,
                 dropout_rate3=0.0):
        super(SFKH_GC_MLP, self).__init__()
        self.hops = hops
        self.seq_len = hops + 1
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_class = n_class
        self.k = hops+1
        self.final_ln1 = nn.LayerNorm(self.input_dim)
        self.att_embeddings_nope1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.dropout1 = torch.nn.Dropout(dropout_rate1)
        self.fc1 = nn.Linear((self.seq_len*self.hidden_dim) , 64)
        self.dropout2 = torch.nn.Dropout(dropout_rate2)
        self.fc2 = nn.Linear(64,n_class)
    def forward(self, x):
        x = self.final_ln1(x)
        x = F.relu(self.att_embeddings_nope1(x))
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        output = self.fc2(x)

        return torch.log_softmax(output, dim=1)

class SFKH_GC_CNN(nn.Module):
    def __init__(self,
                 hops,
                 n_class,
                 input_dim,
                 hidden_dim=512,
                 dropout_rate1=0.5,
                 dropout_rate2=0.3,
                 Kernels_per_Layer=6,

                 ):
        super(SFKH_GC_CNN, self).__init__()
        self.hops = hops
        self.seq_len = hops + 1
        self.n_class = n_class
        self.k = hops+1
        self.input_dim=int(input_dim)
        self.hidden_dim=hidden_dim
        self.final_ln = nn.LayerNorm(self.input_dim)
        self.Kernels_per_Layer=Kernels_per_Layer

        self.conv1 = nn.Conv2d(1,self.Kernels_per_Layer, kernel_size=3, stride=1, padding=0)
        self.att_embeddings_nope1 = nn.Linear(self.input_dim, self.hidden_dim)

        feature_row_cov1=int(self.input_dim-2)
        feature_col_cov1=self.k-2

        self.conv2 = nn.Conv2d(self.Kernels_per_Layer, self.Kernels_per_Layer, kernel_size=3, stride=1, padding=0)
        feature_row_cov2=feature_row_cov1-2
        feature_col_cov2=feature_col_cov1-2

        self.conv3 = nn.Conv2d(self.Kernels_per_Layer, self.Kernels_per_Layer, kernel_size=3, stride=1, padding=0)
        feature_row_cov3 = feature_row_cov2 - 2
        feature_col_cov3 = feature_col_cov2 - 2
        self.conv4 = nn.Conv2d(self.Kernels_per_Layer, self.Kernels_per_Layer, kernel_size=3, stride=1, padding=0)
        feature_row_cov4 = feature_row_cov3 - 2
        feature_col_cov4 = feature_col_cov3 - 2

        self.fc1 = nn.Linear(feature_row_cov3*feature_col_cov3* self.Kernels_per_Layer, hidden_dim)
        self.dropout1 = torch.nn.Dropout(dropout_rate1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = torch.nn.Dropout(dropout_rate2)
        self.fc3 = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        #x = F.relu(self.att_embeddings_nope1(x))
        x = x.unsqueeze(1)
        x = self.final_ln(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #x = self.conv4(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x1 = F.relu(self.fc1(x))
        x1 = self.dropout1(x1)
        x2 = F.relu(self.fc2(x1))
        x = x1+x2
        x =self.dropout2(x)
        output = self.fc3(x)
        return torch.log_softmax(output, dim=1)
def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)



def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

class SFKH_GC_TransformerModel(nn.Module):
    def __init__(
        self,
        hops, 
        n_class,
        input_dim, 
        pe_dim,
        n_layers=6,
        num_heads=8,
        hidden_dim=64,
        ffn_dim=64, 
        dropout_rate=0.0,
        attention_dropout_rate=0.1
    ):
        super().__init__()
        self.hops=hops
        self.seq_len = hops+1
        self.pe_dim = pe_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = 2 * hidden_dim
        self.num_heads = num_heads
        
        self.n_layers = n_layers
        self.n_class = n_class

        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)

        encoders = [EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads,self.seq_len)
                    for _ in range(self.n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)
        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))

        self.Linear1 = nn.Linear(int(self.hidden_dim/2), self.n_class)

        self.scaling = nn.Parameter(torch.ones(1) * 0.5)


        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):


        tensor = self.att_embeddings_nope(batched_data)

        
        # transformer encoder
        for enc_layer in self.layers:
            tensor = enc_layer(tensor)
        
        output = self.final_ln(tensor)

        target = output[:,0,:].unsqueeze(1).repeat(1,self.seq_len-1,1)

        split_tensor = torch.split(output, [1, self.seq_len-1], dim=1)

        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]

        x = torch.cat((target, neighbor_tensor))

        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        #这里是采用mlp的形式在计算注意力权重，尝试使用叉乘形式。

        layer_atten = F.softmax(layer_atten, dim=1)
        neighbor_tensor = neighbor_tensor * layer_atten
        #这里可以引入超参数，来放缩K跳邻居信息对节点原始表示的影响大小
        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)

        output = (node_tensor + neighbor_tensor).squeeze()

        output = self.Linear1(torch.relu(self.out_proj(output)))

    
        return torch.log_softmax(output, dim=1)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads,seq_len):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        #self.ffn_norm = nn.BatchNorm1d(seq_len)

        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x







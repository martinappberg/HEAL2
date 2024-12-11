import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.readout import sum_nodes
import dgl
from entmax import entmax_bisect

def bert_init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.BatchNorm1d):
        torch.nn.init.normal_(module.weight.data, mean=1, std=0.02)
        torch.nn.init.constant_(module.bias.data, 0)

class MLP(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, n_layers, activation=nn.GELU(), bias=False, dropout=0, use_batchnorm=False, out_batchnorm=False):
        super().__init__()
        self.n_layers = n_layers
        self.use_batchnorm = use_batchnorm
        self.out_batchnorm = out_batchnorm
        self.linear_list = nn.ModuleList()
        if n_layers == 1:
            self.linear_list.append(nn.Linear(d_input, d_output, bias=bias))
        else:
            self.linear_list.append(nn.Linear(d_input, d_hidden, bias=bias))
            for _ in range(n_layers-2):
                self.linear_list.append(nn.Linear(d_hidden, d_hidden, bias=bias))
            self.linear_list.append(nn.Linear(d_hidden, d_output, bias=bias))
        if use_batchnorm:
            self.batch_norm_list = nn.ModuleList()
            for _ in range(n_layers):
                self.batch_norm_list.append(nn.BatchNorm1d((d_hidden)))
        self.activation=activation
        self.dropout = nn.Dropout(dropout)
    def forward(self, h):
        for i in range(self.n_layers-1):
            h = self.linear_list[i](h)
            if self.use_batchnorm:
                self.batch_norm_list[i](h)
            h = self.dropout(self.activation(h))
        h = self.linear_list[-1](h)
        if self.use_batchnorm and self.out_batchnorm:
            self.batch_norm_list[-1](h)
        if self.out_batchnorm:
            h = self.dropout(self.activation(h))
        return h

class AttentiveReadout(nn.Module):
    def __init__(self, in_feats):
        super(AttentiveReadout, self).__init__()
        self.in_feats = in_feats
        self.key_layer = nn.Linear(in_feats, in_feats)
        self.query_layer = nn.Sequential(
            nn.Linear(in_feats, 1, bias=False),
            nn.Sigmoid()
        )
        self.value_layer = nn.Linear(in_feats, in_feats)
        self.proj = nn.Linear(2*in_feats, in_feats)
    def forward(self, g, feats):
        with g.local_scope():
            keys = self.key_layer(feats)
            g.ndata['w'] = self.query_layer(keys)
            g.ndata['v'] = self.value_layer(feats)
            h = sum_nodes(g, 'v', 'w')
            return h, g.ndata['w']

class AttentiveReadoutWithSAE(nn.Module):
    def __init__(self, in_feats, sae_hidden_dim, sae_activation=nn.ReLU(), sparsity_coef=1e-3):
        super(AttentiveReadoutWithSAE, self).__init__()
        self.in_feats = in_feats
        # Original attention layers
        self.key_layer = nn.Linear(in_feats, in_feats)
        self.query_layer = nn.Sequential(
            nn.Linear(in_feats, 1, bias=False),
            nn.Sigmoid()
        )
        self.value_layer = nn.Linear(in_feats, in_feats)
        # Sparse Autoencoder layers
        self.encoder = nn.Linear(in_feats, sae_hidden_dim)
        self.decoder = nn.Linear(sae_hidden_dim, in_feats)
        self.sae_activation = sae_activation
        self.sparsity_coef = sparsity_coef

    def forward(self, g, feats):
        with g.local_scope():
            keys = self.key_layer(feats)
            g.ndata['w'] = self.query_layer(keys)
            g.ndata['v'] = self.value_layer(feats)

            # Compute attention-weighted node features like before
            g.ndata['z'] = g.ndata['v'] * g.ndata['w']

            # Apply Sparse Autoencoder to attention-weighted features
            z = g.ndata['z']
            encoded = self.sae_activation(self.encoder(z))
            decoded = self.decoder(encoded)
            sae_loss = F.mse_loss(decoded, z)
            sparsity_loss = torch.mean(torch.abs(encoded))
            total_sae_loss = sae_loss + self.sparsity_coef * sparsity_loss

            # Use the encoded features for readout
            g.ndata['z_sae'] = decoded
            h = dgl.sum_nodes(g, 'z_sae', weight='w')  # Sum over nodes

            return h, g.ndata['w'], total_sae_loss, g.ndata['z_sae']

class PRSNet(torch.nn.Module):
    def __init__(self, d_input=11, d_hidden=64, n_gene_encode_layer=1, n_layers=1, n_genes=19836, n_predictor_layer=2, mlp_hidden_ratio=1, n_covariates=0):
        super().__init__()
        self.activation = nn.GELU()
        self.n_layers = n_layers
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.n_covariates = n_covariates

        ## Gene Encoder
        self.gene_encoder = MLP(d_input=d_input, d_hidden=d_hidden, d_output=d_hidden, n_layers=n_gene_encode_layer, activation=self.activation, bias=True, use_batchnorm=True, out_batchnorm=True)
        self.gene_embeddings = nn.Embedding(n_genes, d_hidden)
        ## GIN
        self.gnn_layer_list = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()
        for _ in range(n_layers): 
            mlp = MLP(d_hidden, d_hidden*mlp_hidden_ratio, d_hidden, n_layers=n_gene_encode_layer, activation=self.activation, bias=False, use_batchnorm=True, out_batchnorm=False)
            self.gnn_layer_list.append(
                GINConv(mlp, learn_eps=False, aggregator_type='sum')
            )
        self.batch_norm_list = nn.ModuleList()
        for _ in range(n_layers):
            self.batch_norm_list.append(nn.BatchNorm1d(d_hidden))
        ## Attentive readout
        self.readout = AttentiveReadoutWithSAE(
            in_feats=d_hidden,
            sae_hidden_dim=d_hidden,
            sparsity_coef=1
        )
        ## Predictor
        self.predictor = MLP(d_input=d_hidden + n_covariates, d_hidden=d_hidden, d_output=1, n_layers=n_predictor_layer, dropout=0, activation=self.activation, bias=True, use_batchnorm=True, out_batchnorm=False)
        ## Parameter initialization
        self.apply(lambda module: bert_init_params(module))
    def forward(self, g, x, ancestries=None, covariates=torch.empty(0)):
        ## Gene encoding
        x = x.reshape(x.shape[0],-1,self.d_input)
        batch_size, n_gene, d_feats = x.shape
        x = x.reshape(-1,self.d_input)
        h = self.gene_encoder(x)
        h = h.reshape(batch_size, n_gene, self.d_hidden)
        h = h + self.gene_embeddings.weight
        h = h.reshape(-1,self.d_hidden)
        ## GNN
        hidden_rep = [h]
        for i in range(self.n_layers):
            h = self.gnn_layer_list[i](g, h)
            h = self.batch_norm_list[i](h)
            h = F.gelu(h)
            hidden_rep.append(h)
        ## Readout
        g_h, weights, sae_loss, z_sae = self.readout(g, hidden_rep[-1])
        ## Prediction
        if self.n_covariates > 0:
            g_h = torch.cat([g_h, covariates], dim=1)
        preds = self.predictor(g_h)
        return preds, weights, sae_loss, z_sae

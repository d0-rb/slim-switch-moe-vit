import torch as th
import torch_geometric as pyg


# attention = th.rand(1, 5, 5)
# edges = pyg.utils.dense_to_sparse(attention)

# print(edges[0].shape)

# attention = th.rand(2, 5, 5)
# edges = pyg.utils.dense_to_sparse(attention)

# print(edges[0].shape)

# batch_idx = th.arange(2).tile(25, 1)
# print(batch_idx, batch_idx.shape)


# print(batch_idx.T.reshape(-1))


def construct_graph(batch_adj):
    shape = batch_adj.shape
    assert shape[-1] == shape[-2]
    edges, edge_attrs = pyg.utils.dense_to_sparse(batch_adj)
    batch_idx = th.arange(shape[0]).tile(shape[-1] ** 2, 1).T.reshape(-1)
    return edges, edge_attrs, batch_idx


attention = th.rand(10, 5, 5)
attention[0, 0] = 0
attention[0, :, 0] = 0
graph = construct_graph(attention)
print(graph[0], graph[-1])

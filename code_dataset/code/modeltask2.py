# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # from keras import backend as K
# from collections import defaultdict
# import numpy as np
# import os
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# class GNN1(nn.Module):
#     def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
#         super(GNN1, self).__init__(**kwargs)
#         self.kg, self.dict1 = dataset["dataset1"], dict1
#         self.drug_name, self.args = drug_name, args
#         self.drug_embed = nn.Embedding(num_embeddings=846, embedding_dim=args.embedding_num)
#         self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset1"], embedding_dim=args.embedding_num)
#         self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset1"], embedding_dim=args.embedding_num)
#         self.W1 = nn.Parameter(torch.randn(size=(846, args.embedding_num, args.embedding_num)))
#         self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
#         self.W2 = nn.Parameter(torch.randn(size=(846, args.embedding_num, args.embedding_num)))
#         self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
#         self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
#                                      nn.ReLU(),
#                                      nn.BatchNorm1d(args.embedding_num))
#
#         self.relu = nn.ReLU()
#         self.soft = nn.Softmax(dim=1)
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, datas):
#         kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
#         adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)
#
#         # drug_name = torch.LongTensor(drug_name)
#         # adj_tail = torch.LongTensor(adj_tail)
#         # adj_relation = torch.LongTensor(adj_relation)
#
#         # drug_embedding = self.drug_embed(drug_name)
#         # rela_embedding = self.rela_embed(adj_relation)
#         # ent_embedding = self.ent_embed(adj_tail)
#
#         drug_name=torch.LongTensor(drug_name).to(device)
#         adj_tail=torch.LongTensor(adj_tail).to(device)
#         adj_relation=torch.LongTensor(adj_relation).to(device)
#
#
#         drug_embedding = self.drug_embed(drug_name).to(device)
#         rela_embedding = self.rela_embed(adj_relation).to(device)
#         ent_embedding = self.ent_embed(adj_tail).to(device)
#
#         drug_rel = drug_embedding.reshape((846, 1, args.embedding_num)) * rela_embedding
#         drug_rel_weigh = drug_rel.matmul(self.W1) + self.b1
#         drug_rel_weigh = self.relu(drug_rel_weigh)
#         drug_rel_weigh = drug_rel_weigh.matmul(self.W2) + self.b2
#         drug_rel_score = torch.sum(drug_rel_weigh, axis=-1, keepdims=True)
#         drug_rel_score = self.soft(drug_rel_score)
#         weighted_ent = drug_rel_score.reshape((846, 1, args.neighbor_sample_size)).matmul(ent_embedding)
#         drug_e = torch.cat(
#             [weighted_ent.reshape(846, args.embedding_num), drug_embedding.reshape((846, args.embedding_num))], dim=1)
#         drug_f = self.Linear1(drug_e)
#         idx, train_or_test, test_adj = datas[0], datas[1], datas[2]
#         if train_or_test == 1:
#             for i in test_adj[0].keys():
#                 pos = test_adj[0][i][0]
#                 length = len(pos)
#                 drug_f[i] = torch.sum(drug_f[pos], dim=0) / length
#         return drug_f, idx, test_adj, train_or_test
#
#     def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=846):
#         adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
#         adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
#         for i in drug_name_id:
#             all_neighbors = kg[drug_name_id[i]]
#             n_neighbor = len(all_neighbors)
#             sample_indices = np.random.choice(
#                 n_neighbor,
#                 neighbor_sample_size,
#                 replace=False if n_neighbor >= neighbor_sample_size else True
#             )
#             adj_tail[drug_name_id[i]] = np.array([all_neighbors[i][0] for i in sample_indices])
#             adj_relation[drug_name_id[i]] = np.array([all_neighbors[i][1] for i in sample_indices])
#         return adj_tail, adj_relation
#
#
# class GNN2(nn.Module):
#     def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
#         super(GNN2, self).__init__(**kwargs)
#         self.kg, self.dict1 = dataset["dataset2"], dict1
#         self.drug_name, self.args = drug_name, args
#         self.drug_embed = nn.Embedding(num_embeddings=846, embedding_dim=args.embedding_num)
#         self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset2"], embedding_dim=args.embedding_num)
#         self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset2"], embedding_dim=args.embedding_num)
#         self.W1 = nn.Parameter(torch.randn(size=(846, args.embedding_num, args.embedding_num)))
#         self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
#         self.W2 = nn.Parameter(torch.randn(size=(846, args.embedding_num, args.embedding_num)))
#         self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
#         self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
#                                      nn.ReLU(),
#                                      nn.BatchNorm1d(args.embedding_num))
#         self.relu = nn.ReLU()
#         self.soft = nn.Softmax(dim=1)
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, arguments):
#         kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
#         gnn1_embedding, idx, test_adj, train_or_test = arguments
#
#         adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)
#         # drug_name = torch.LongTensor(drug_name)
#         # adj_tail = torch.LongTensor(adj_tail)
#         # adj_relation = torch.LongTensor(adj_relation)
#         # drug_embedding = self.drug_embed(drug_name)
#         # rela_embedding = self.rela_embed(adj_relation)
#         # ent_embedding = self.ent_embed(adj_tail)
#
#         drug_name=torch.LongTensor(drug_name).to(device)
#         adj_tail=torch.LongTensor(adj_tail).to(device)
#         adj_relation=torch.LongTensor(adj_relation).to(device)
#
#
#         drug_embedding = self.drug_embed(drug_name).to(device)
#         rela_embedding = self.rela_embed(adj_relation).to(device)
#         ent_embedding = self.ent_embed(adj_tail).to(device)
#
#         drug_rel = drug_embedding.reshape((846, 1, args.embedding_num)) * rela_embedding
#         drug_rel_weigh = drug_rel.matmul(self.W1) + self.b1
#         drug_rel_weigh = self.relu(drug_rel_weigh)
#         drug_rel_weigh = drug_rel_weigh.matmul(self.W2) + self.b2
#         drug_rel_score = torch.sum(drug_rel_weigh, axis=-1, keepdims=True)
#         drug_rel_score = self.soft(drug_rel_score)
#         weighted_ent = drug_rel_score.reshape((846, 1, args.neighbor_sample_size)).matmul(ent_embedding)
#         drug_e = torch.cat(
#             [weighted_ent.reshape(846, args.embedding_num), drug_embedding.reshape((846, args.embedding_num))], dim=1)
#         drug_f = self.Linear1(drug_e)
#         if train_or_test == 1:
#             for i in test_adj[1].keys():
#                 pos = test_adj[1][i][0]
#                 length = len(pos)
#                 drug_f[i] = torch.sum(drug_f[pos], dim=0) / length
#         return drug_f, gnn1_embedding, idx, test_adj, train_or_test
#
#     def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=846):
#         adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
#         adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
#         for i in drug_name_id:
#             all_neighbors = kg[drug_name_id[i]]
#             n_neighbor = len(all_neighbors)
#             sample_indices = np.random.choice(
#                 n_neighbor,
#                 neighbor_sample_size,
#                 replace=False if n_neighbor >= neighbor_sample_size else True
#             )
#             adj_tail[drug_name_id[i]] = np.array([all_neighbors[i][0] for i in sample_indices])
#             adj_relation[drug_name_id[i]] = np.array([all_neighbors[i][1] for i in sample_indices])
#         return adj_tail, adj_relation
#
#
# class GNN3(nn.Module):
#     def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
#         super(GNN3, self).__init__(**kwargs)
#         self.kg, self.dict1 = dataset["dataset3"], dict1
#         self.drug_name, self.args = drug_name, args
#         self.drug_embed = nn.Embedding(num_embeddings=846, embedding_dim=args.embedding_num)
#         self.rela_embed = nn.Embedding(num_embeddings=67, embedding_dim=args.embedding_num)
#         self.ent_embed = nn.Embedding(num_embeddings=846, embedding_dim=args.embedding_num)
#         self.W1 = nn.Parameter(torch.randn(size=(846, args.embedding_num, args.embedding_num)))
#         self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
#         self.W2 = nn.Parameter(torch.randn(size=(846, args.embedding_num, args.embedding_num)))
#         self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
#         self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
#                                      nn.ReLU(),
#                                      nn.BatchNorm1d(args.embedding_num))
#         self.relu = nn.ReLU()
#         self.soft = nn.Softmax(dim=1)
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, arguments):
#         kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
#         gnn2_embedding, gnn1_embedding, idx, test_adj, train_or_test = arguments
#
#         adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)
#         # drug_name = torch.LongTensor(drug_name)
#         # adj_tail = torch.LongTensor(adj_tail)
#         # adj_relation = torch.LongTensor(adj_relation)
#         # drug_embedding = self.drug_embed(drug_name)
#         # rela_embedding = self.rela_embed(adj_relation)
#         # ent_embedding = self.ent_embed(adj_tail)
#
#         drug_name=torch.LongTensor(drug_name).to(device)
#         adj_tail=torch.LongTensor(adj_tail).to(device)
#         adj_relation=torch.LongTensor(adj_relation).to(device)
#
#
#         drug_embedding = self.drug_embed(drug_name).to(device)
#         rela_embedding = self.rela_embed(adj_relation).to(device)
#         ent_embedding = self.ent_embed(adj_tail).to(device)
#
#         drug_rel = drug_embedding.reshape((846, 1, args.embedding_num)) * rela_embedding
#         drug_rel_weigh = drug_rel.matmul(self.W1) + self.b1
#         drug_rel_weigh = self.relu(drug_rel_weigh)
#         drug_rel_weigh = drug_rel_weigh.matmul(self.W2) + self.b2
#         drug_rel_score = torch.sum(drug_rel_weigh, axis=-1, keepdims=True)
#         drug_rel_score = self.soft(drug_rel_score)
#         weighted_ent = drug_rel_score.reshape((846, 1, args.neighbor_sample_size)).matmul(ent_embedding)
#         drug_e = torch.cat(
#             [weighted_ent.reshape(846, args.embedding_num), drug_embedding.reshape((846, args.embedding_num))], dim=1)
#         drug_f = self.Linear1(drug_e)
#         if train_or_test == 1:
#             for i in test_adj[3].keys():
#                 pos = test_adj[3][i][0]
#                 length = len(pos)
#                 drug_f[i] = torch.sum(drug_f[pos], dim=0) / length
#         return drug_f, gnn2_embedding, gnn1_embedding, idx
#
#     def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=846, tails_num=570, relations_num=73):
#         drug_number = []
#         drug_list = []
#         for i in drug_name_id:
#             drug_number.append(drug_name_id[i])
#         for key in kg:
#             drug_list.append(key)
#         surplus = set(drug_number).difference(set(drug_list))
#         for i in list(surplus):
#             kg[i].append((tails_num + 1, relations_num + 1))
#         adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
#         adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
#         for i in drug_name_id:
#             all_neighbors = kg[drug_name_id[i]]
#             n_neighbor = len(all_neighbors)
#             sample_indices = np.random.choice(
#                 n_neighbor,
#                 neighbor_sample_size,
#                 replace=False if n_neighbor >= neighbor_sample_size else True
#             )
#             adj_tail[drug_name_id[i]] = np.array([all_neighbors[i][0] for i in sample_indices])
#             adj_relation[drug_name_id[i]] = np.array([all_neighbors[i][1] for i in sample_indices])
#         return adj_tail, adj_relation
#
#
# class GNN4(nn.Module):
#     def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
#         super(GNN4, self).__init__(**kwargs)
#         self.kg, self.dict1 = dataset["dataset3"], dict1
#         self.drug_name, self.args = drug_name, args
#         self.drug_embed = nn.Embedding(num_embeddings=846, embedding_dim=args.embedding_num)
#         self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset3"], embedding_dim=args.embedding_num)
#         self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset3"], embedding_dim=args.embedding_num)
#         self.W1 = nn.Parameter(torch.randn(size=(846, args.embedding_num, args.embedding_num)))
#         self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
#         self.W2 = nn.Parameter(torch.randn(size=(846, args.embedding_num, args.embedding_num)))
#         self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
#
#         self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
#                                      nn.ReLU(),
#                                      nn.BatchNorm1d(args.embedding_num))
#         self.relu = nn.ReLU()
#         self.soft = nn.Softmax(dim=1)
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, arguments):
#         kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
#         gnn3_embedding, gnn2_embedding, gnn1_embedding, idx, test_adj, train_or_test = arguments
#         adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)
#         drug_name = torch.LongTensor(drug_name)
#         adj_tail = torch.LongTensor(adj_tail)
#         adj_relation = torch.LongTensor(adj_relation)
#         drug_embedding = self.drug_embed(drug_name)
#         rela_embedding = self.rela_embed(adj_relation)
#         ent_embedding = self.ent_embed(adj_tail)
#         drug_rel = drug_embedding.reshape((846, 1, args.embedding_num)) * rela_embedding
#         drug_rel_weigh = drug_rel.matmul(self.W1) + self.b1
#         drug_rel_weigh = self.relu(drug_rel_weigh)
#         drug_rel_weigh = drug_rel_weigh.matmul(self.W2) + self.b2
#         drug_rel_score = torch.sum(drug_rel_weigh, axis=-1, keepdims=True)
#         drug_rel_score = self.soft(drug_rel_score)
#         weighted_ent = drug_rel_score.reshape((846, 1, args.neighbor_sample_size)).matmul(ent_embedding)
#         drug_e = torch.cat(
#             [weighted_ent.reshape(846, args.embedding_num), drug_embedding.reshape((846, args.embedding_num))], dim=1)
#         drug_f = self.Linear1(drug_e)
#         if train_or_test == 1:
#             for i in test_adj[2].keys():
#                 pos = test_adj[2][i][0]
#                 length = len(pos)
#                 drug_f[i] = torch.sum(drug_f[pos], dim=0) / length
#         return drug_f, gnn3_embedding, gnn2_embedding, gnn1_embedding, idx
#
#     def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=846):
#         adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
#         adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
#         for i in drug_name_id:
#             all_neighbors = kg[drug_name_id[i]]
#             n_neighbor = len(all_neighbors)
#             sample_indices = np.random.choice(
#                 n_neighbor,
#                 neighbor_sample_size,
#                 replace=False if n_neighbor >= neighbor_sample_size else True
#             )
#             adj_tail[drug_name_id[i]] = np.array([all_neighbors[i][0] for i in sample_indices])
#             adj_relation[drug_name_id[i]] = np.array([all_neighbors[i][1] for i in sample_indices])
#         return adj_tail, adj_relation
#
#
# class FusionLayer(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.fullConnectionLayer1 = nn.Sequential(
#             nn.Linear(args.embedding_num * 3 * 2, args.embedding_num * 3),
#             nn.ReLU(),
#             nn.BatchNorm1d(args.embedding_num * 3),
#             nn.Dropout(args.dropout),
#             nn.Linear(args.embedding_num * 3, args.embedding_num * 2),
#             nn.ReLU(),
#             nn.BatchNorm1d(args.embedding_num * 2),
#             nn.Dropout(args.dropout),
#             nn.Linear(args.embedding_num * 2, 73))
#         self.fullConnectionLayer2 = nn.Sequential(
#             nn.Linear(args.embedding_num * 3 * 2 + 73, args.embedding_num * 3 + 73),
#             nn.ReLU(),
#             nn.BatchNorm1d(args.embedding_num * 3 + 73),
#             nn.Dropout(args.dropout),
#             nn.Linear(args.embedding_num * 3 + 73, args.embedding_num * 2 + 73),
#             nn.ReLU(),
#             nn.BatchNorm1d(args.embedding_num * 2 + 73),
#             nn.Dropout(args.dropout),
#             nn.Linear(args.embedding_num * 2 + 73, 73))
#         self.fullConnectionLayer3 = nn.Sequential(
#             nn.Linear(args.embedding_num * 3 * 2 + 73 * 2, args.embedding_num * 3 + 73 * 2),
#             nn.ReLU(),
#             nn.BatchNorm1d(args.embedding_num * 3 + 73 * 2),
#             nn.Dropout(args.dropout),
#             nn.Linear(args.embedding_num * 3 + 73 * 2, args.embedding_num * 2 + 73 * 2),
#             nn.ReLU(),
#             nn.BatchNorm1d(args.embedding_num * 2 + 73 * 2),
#             nn.Dropout(args.dropout),
#             nn.Linear(args.embedding_num * 2 + 73 * 2, 73))
#         self.fullConnectionLayer4 = nn.Sequential(
#             nn.Linear(args.embedding_num * 3 * 2 + 73 * 3, args.embedding_num * 3 + 73 * 3),
#             nn.ReLU(),
#             nn.BatchNorm1d(args.embedding_num * 3 + 73 * 3),
#             nn.Dropout(args.dropout),
#             nn.Linear(args.embedding_num * 3 + 73 * 3, args.embedding_num * 2 + 73 * 3),
#             nn.ReLU(),
#             nn.BatchNorm1d(args.embedding_num * 2 + 73 * 3),
#             nn.Dropout(args.dropout),
#             nn.Linear(args.embedding_num * 2 + 73 * 3, 73))
#         self.fullConnectionLayer5 = nn.Sequential(
#             nn.Linear(args.embedding_num * 3 * 2 + 73 * 4, args.embedding_num * 3 + 73 * 4),
#             nn.ReLU(),
#             nn.BatchNorm1d(args.embedding_num * 3 + 73 * 4),
#             nn.Dropout(args.dropout),
#             nn.Linear(args.embedding_num * 3 + 73 * 4, args.embedding_num * 2 + 73 * 4),
#             nn.ReLU(),
#             nn.BatchNorm1d(args.embedding_num * 2 + 73 * 4),
#             nn.Dropout(args.dropout),
#             nn.Linear(args.embedding_num * 2 + 73 * 4, 73))
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, arguments):
#         # gnn4_embedding[drugA],,gnn4_embedding[drugB]  gnn4_embedding, gnn4_embedding,gnn4_embedding[drugA],,gnn4_embedding[drugB]gnn4_embedding,gnn4_embedding[drugA],,gnn4_embedding[drugB]
#         gnn3_embedding, gnn2_embedding, gnn1_embedding, idx = arguments
#         idx = idx.cpu().numpy().tolist()
#         drugA = []
#         drugB = []
#         for i in idx:
#             drugA.append(i[0])
#             drugB.append(i[1])
#
#         Embedding = torch.cat(
#             [gnn1_embedding[drugA], gnn2_embedding[drugA], gnn3_embedding[drugA], \
#              gnn1_embedding[drugB], gnn2_embedding[drugB], gnn3_embedding[drugB]], 1).float()
#         # print(Embedding.shape)
#         # os.system("pause")
#         # prediction11 = F.softmax(prediction1,dim=1)
#         # prediction22 = F.softmax(prediction2, dim=1)
#         # prediction33 = F.softmax(prediction3, dim=1)
#         # prediction44 = F.softmax(prediction4, dim=1)
#
#
#
#         prediction1 = self.fullConnectionLayer1(Embedding)
#         Embedding1 = torch.cat([prediction1,Embedding], 1).float()
#         prediction2 = self.fullConnectionLayer2(Embedding1)
#         Embedding2 = torch.cat([prediction2,prediction1, Embedding], 1).float()
#         prediction3 = self.fullConnectionLayer3(Embedding2)
#         Embedding3 = torch.cat([prediction3,prediction2,prediction1, Embedding], 1).float()
#         prediction4 = self.fullConnectionLayer4(Embedding3)
#         Embedding4 = torch.cat([prediction4,prediction3,prediction2,prediction1, Embedding], 1).float()
#         prediction5 = self.fullConnectionLayer5(Embedding4)
#
#         return prediction1, prediction2, prediction3, prediction4, prediction5
#
#         # prediction1 = self.fullConnectionLayer1(Embedding)
#         # Embedding1 = torch.cat([prediction1, Embedding], 1).float()
#         # prediction2 = self.fullConnectionLayer2(Embedding1)
#         # Embedding2 = torch.cat([prediction2, Embedding], 1).float()
#         # prediction3 = self.fullConnectionLayer3(Embedding2)
#         # Embedding3 = torch.cat([prediction3, Embedding], 1).float()
#         # prediction4 = self.fullConnectionLayer4(Embedding3)
#         # Embedding4 = torch.cat([prediction4, Embedding], 1).float()
#         # prediction5 = self.fullConnectionLayer5(Embedding4)
#  # self.fullConnectionLayer1 = nn.Sequential(
#  #            nn.Linear(args.embedding_num * 3 * 2, args.embedding_num * 3),
#  #            nn.ReLU(),
#  #            nn.BatchNorm1d(args.embedding_num * 3),
#  #            nn.Dropout(args.dropout),
#  #            nn.Linear(args.embedding_num * 3, args.embedding_num * 2),
#  #            nn.ReLU(),
#  #            nn.BatchNorm1d(args.embedding_num * 2),
#  #            nn.Dropout(args.dropout),
#  #            nn.Linear(args.embedding_num * 2, 73))
#  #        self.fullConnectionLayer2 = nn.Sequential(
#  #            nn.Linear(args.embedding_num * 3 * 2+73, args.embedding_num * 3),
#  #            nn.ReLU(),
#  #            nn.BatchNorm1d(args.embedding_num * 3),
#  #            nn.Dropout(args.dropout),
#  #            nn.Linear(args.embedding_num * 3, args.embedding_num * 2),
#  #            nn.ReLU(),
#  #            nn.BatchNorm1d(args.embedding_num * 2),
#  #            nn.Dropout(args.dropout),
#  #            nn.Linear(args.embedding_num * 2, 73))
#  #        self.fullConnectionLayer3 = nn.Sequential(
#  #            nn.Linear(args.embedding_num * 3 * 2 + 73, args.embedding_num * 3),
#  #            nn.ReLU(),
#  #            nn.BatchNorm1d(args.embedding_num * 3),
#  #            nn.Dropout(args.dropout),
#  #            nn.Linear(args.embedding_num * 3, args.embedding_num * 2),
#  #            nn.ReLU(),
#  #            nn.BatchNorm1d(args.embedding_num * 2),
#  #            nn.Dropout(args.dropout),
#  #            nn.Linear(args.embedding_num * 2, 73))
#  #        self.fullConnectionLayer4 = nn.Sequential(
#  #            nn.Linear(args.embedding_num * 3 * 2 + 73, args.embedding_num * 3),
#  #            nn.ReLU(),
#  #            nn.BatchNorm1d(args.embedding_num * 3),
#  #            nn.Dropout(args.dropout),
#  #            nn.Linear(args.embedding_num * 3, args.embedding_num * 2),
#  #            nn.ReLU(),
#  #            nn.BatchNorm1d(args.embedding_num * 2),
#  #            nn.Dropout(args.dropout),
#  #            nn.Linear(args.embedding_num * 2, 73))
#  #        self.fullConnectionLayer5 = nn.Sequential(
#  #            nn.Linear(args.embedding_num * 3 * 2 + 73, args.embedding_num * 3),
#  #            nn.ReLU(),
#  #            nn.BatchNorm1d(args.embedding_num * 3),
#  #            nn.Dropout(args.dropout),
#  #            nn.Linear(args.embedding_num * 3, args.embedding_num * 2),
#  #            nn.ReLU(),
#  #            nn.BatchNorm1d(args.embedding_num * 2),
#  #            nn.Dropout(args.dropout),
#  #            nn.Linear(args.embedding_num * 2, 73))

import torch
import torch.nn as nn
import torch.nn.functional as F
# from keras import backend as K
from collections import defaultdict
import numpy as np
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Tailed_FocalLoss(nn.Module):
    def __init__(self, threshold, alphas: list = None, gamma=2, reduction: str = 'mean', ):
        super().__init__()
        self.alphas = alphas  # The inverse of frequency of class
        self.gamma = gamma
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, probs, target):
        '''
        formula for focal loss: average(-alpha*(1-Pt)log(Pt))
        my probs = [[],[],[]]
        tagert = [[31],[11],[3]]
        '''
        Pt_index = target.squeeze()
        # print(Pt_index)
        # print(np.arange(len(Pt_index)), Pt_index)
        # print(probs)
        # print(gate)
        # os.system("pause")
        # Pt_dash = torch.where(gate == 0, 1, Pt)
        # print(FL_loss)
        # os.system("pause")
        # print(Pt)
        Pt = probs[np.arange(len(Pt_index)), Pt_index]
        FL_loss = -((1 - Pt) ** self.gamma) * torch.log2(Pt)
        gate = torch.where(Pt_index > self.threshold, 1, 0)
        Pt_dash = Pt ** gate
        # loss = -torch.log2(ratio*Pt)
        loss = FL_loss + (-torch.log2(Pt_dash ** 2))

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, probs, target):
        '''
        formula for focal loss: average(-alpha*(1-Pt)log(Pt))
        my probs = [[],[],[]]
        tagert = [[31],[11],[3]]
        '''

        # Pt_index = target.squeeze()
        Pt_index = target
        Pt = probs[np.arange(len(Pt_index)), Pt_index]
        # print(Pt)
        loss = -((1 - Pt) ** self.gamma) * torch.log2(Pt)
        # print(loss)
        # os.system("pause")

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


class GNN1(nn.Module):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super(GNN1, self).__init__(**kwargs)
        self.kg, self.dict1 = dataset["dataset1"], dict1
        self.drug_name, self.args = drug_name, args
        self.drug_embed = nn.Embedding(num_embeddings=846, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset1"], embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset1"], embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(size=(846, args.embedding_num, args.embedding_num),device="cpu").to(device))
        self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num),device="cpu").to(device))
        self.W2 = nn.Parameter(torch.randn(size=(846, args.embedding_num, args.embedding_num),device="cpu").to(device))
        self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num),device="cpu").to(device))
        self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(args.embedding_num))

        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, datas):
        kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
        adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)
        drug_name = torch.LongTensor(drug_name).to(device)
        adj_tail = torch.LongTensor(adj_tail).to(device)
        adj_relation = torch.LongTensor(adj_relation).to(device)
        drug_embedding = self.drug_embed(drug_name)
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)
        drug_rel = drug_embedding.reshape((846, 1, args.embedding_num)) * rela_embedding
        drug_rel_weigh = drug_rel.matmul(self.W1) + self.b1
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = drug_rel_weigh.matmul(self.W2) + self.b2
        drug_rel_score = torch.sum(drug_rel_weigh, axis=-1, keepdims=True)
        drug_rel_score = self.soft(drug_rel_score)
        weighted_ent = drug_rel_score.reshape((846, 1, args.neighbor_sample_size)).matmul(ent_embedding)
        drug_e = torch.cat(
            [weighted_ent.reshape(846, args.embedding_num), drug_embedding.reshape((846, args.embedding_num))], dim=1)
        drug_f = self.Linear1(drug_e)
        idx, train_or_test, test_adj = datas[0], datas[1], datas[2]
        if train_or_test == 1:
            for i in test_adj[0].keys():
                pos = test_adj[0][i][0]
                length = len(pos)
                drug_f[i] = torch.sum(drug_f[pos], dim=0) / length
        return drug_f, idx, test_adj, train_or_test

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=846):
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[i][0] for i in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[i][1] for i in sample_indices])
        return adj_tail, adj_relation


class GNN2(nn.Module):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super(GNN2, self).__init__(**kwargs)
        self.kg, self.dict1 = dataset["dataset2"], dict1
        self.drug_name, self.args = drug_name, args
        self.drug_embed = nn.Embedding(num_embeddings=846, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset2"], embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset2"], embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(size=(846, args.embedding_num, args.embedding_num),device="cpu").to(device))
        self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num),device="cpu").to(device))
        self.W2 = nn.Parameter(torch.randn(size=(846, args.embedding_num, args.embedding_num),device="cpu").to(device))
        self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num),device="cpu").to(device))
        self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(args.embedding_num))
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
        gnn1_embedding, idx, test_adj, train_or_test = arguments

        adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)
        drug_name = torch.LongTensor(drug_name).to(device)
        adj_tail = torch.LongTensor(adj_tail).to(device)
        adj_relation = torch.LongTensor(adj_relation).to(device)
        drug_embedding = self.drug_embed(drug_name)
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)
        drug_rel = drug_embedding.reshape((846, 1, args.embedding_num)) * rela_embedding
        drug_rel_weigh = drug_rel.matmul(self.W1) + self.b1
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = drug_rel_weigh.matmul(self.W2) + self.b2
        drug_rel_score = torch.sum(drug_rel_weigh, axis=-1, keepdims=True)
        drug_rel_score = self.soft(drug_rel_score)
        weighted_ent = drug_rel_score.reshape((846, 1, args.neighbor_sample_size)).matmul(ent_embedding)
        drug_e = torch.cat(
            [weighted_ent.reshape(846, args.embedding_num), drug_embedding.reshape((846, args.embedding_num))], dim=1)
        drug_f = self.Linear1(drug_e)
        if train_or_test == 1:
            for i in test_adj[1].keys():
                pos = test_adj[1][i][0]
                length = len(pos)
                drug_f[i] = torch.sum(drug_f[pos], dim=0) / length
        return drug_f, gnn1_embedding, idx, test_adj, train_or_test

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=846):
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[i][0] for i in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[i][1] for i in sample_indices])
        return adj_tail, adj_relation


class GNN3(nn.Module):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super(GNN3, self).__init__(**kwargs)
        self.kg, self.dict1 = dataset["dataset3"], dict1
        self.drug_name, self.args = drug_name, args
        self.drug_embed = nn.Embedding(num_embeddings=846, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=102, embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=846, embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(size=(846, args.embedding_num, args.embedding_num),device="cpu").to(device))
        self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num),device="cpu").to(device))
        self.W2 = nn.Parameter(torch.randn(size=(846, args.embedding_num, args.embedding_num),device="cpu").to(device))
        self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num),device="cpu").to(device))
        self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(args.embedding_num))
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
        gnn2_embedding, gnn1_embedding, idx, test_adj, train_or_test = arguments

        adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)
        drug_name = torch.LongTensor(drug_name).to(device)
        adj_tail = torch.LongTensor(adj_tail).to(device)
        adj_relation = torch.LongTensor(adj_relation).to(device)
        drug_embedding = self.drug_embed(drug_name)
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)
        drug_rel = drug_embedding.reshape((846, 1, args.embedding_num)) * rela_embedding
        drug_rel_weigh = drug_rel.matmul(self.W1) + self.b1
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = drug_rel_weigh.matmul(self.W2) + self.b2
        drug_rel_score = torch.sum(drug_rel_weigh, axis=-1, keepdims=True)
        drug_rel_score = self.soft(drug_rel_score)
        weighted_ent = drug_rel_score.reshape((846, 1, args.neighbor_sample_size)).matmul(ent_embedding)
        drug_e = torch.cat(
            [weighted_ent.reshape(846, args.embedding_num), drug_embedding.reshape((846, args.embedding_num))], dim=1)
        drug_f = self.Linear1(drug_e)
        if train_or_test == 1:
            for i in test_adj[3].keys():
                pos = test_adj[3][i][0]
                length = len(pos)
                drug_f[i] = torch.sum(drug_f[pos], dim=0) / length
        return drug_f, gnn2_embedding, gnn1_embedding, idx

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=846, tails_num=570, relations_num=73):
        drug_number = []
        drug_list = []
        for i in drug_name_id:
            drug_number.append(drug_name_id[i])
        for key in kg:
            drug_list.append(key)
        surplus = set(drug_number).difference(set(drug_list))
        for i in list(surplus):
            kg[i].append((tails_num + 1, relations_num + 1))
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[i][0] for i in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[i][1] for i in sample_indices])
        return adj_tail, adj_relation


class GNN4(nn.Module):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super(GNN4, self).__init__(**kwargs)
        self.kg, self.dict1 = dataset["dataset4"], dict1
        self.drug_name, self.args = drug_name, args
        self.drug_embed = nn.Embedding(num_embeddings=846, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset4"], embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset4"], embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(size=(846, args.embedding_num, args.embedding_num)))
        self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.W2 = nn.Parameter(torch.randn(size=(846, args.embedding_num, args.embedding_num)))
        self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))

        self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(args.embedding_num))
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
        gnn3_embedding, gnn2_embedding, gnn1_embedding, idx, test_adj, train_or_test = arguments
        adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)
        drug_name = torch.LongTensor(drug_name)
        adj_tail = torch.LongTensor(adj_tail)
        adj_relation = torch.LongTensor(adj_relation)
        drug_embedding = self.drug_embed(drug_name)
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)
        drug_rel = drug_embedding.reshape((846, 1, args.embedding_num)) * rela_embedding
        drug_rel_weigh = drug_rel.matmul(self.W1) + self.b1
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = drug_rel_weigh.matmul(self.W2) + self.b2
        drug_rel_score = torch.sum(drug_rel_weigh, axis=-1, keepdims=True)
        drug_rel_score = self.soft(drug_rel_score)
        weighted_ent = drug_rel_score.reshape((846, 1, args.neighbor_sample_size)).matmul(ent_embedding)
        drug_e = torch.cat(
            [weighted_ent.reshape(846, args.embedding_num), drug_embedding.reshape((846, args.embedding_num))], dim=1)
        drug_f = self.Linear1(drug_e)
        if train_or_test == 1:
            for i in test_adj[2].keys():
                pos = test_adj[2][i][0]
                length = len(pos)
                drug_f[i] = torch.sum(drug_f[pos], dim=0) / length
        return drug_f, gnn3_embedding, gnn2_embedding, gnn1_embedding, idx

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=846):
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[i][0] for i in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[i][1] for i in sample_indices])
        return adj_tail, adj_relation


class FusionLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fullConnectionLayer1 = nn.Sequential(
            nn.Linear(args.embedding_num * 3 * 2, args.embedding_num * 3),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 3),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 3, args.embedding_num * 2),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 2),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 2, 73))
        self.fullConnectionLayer2 = nn.Sequential(
            nn.Linear(args.embedding_num * 3 * 2+73, args.embedding_num * 3 +73),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 3 +73),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 3 +73, args.embedding_num * 2+73),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 2+73),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 2+73, 73))
        self.fullConnectionLayer3 = nn.Sequential(
            nn.Linear(args.embedding_num * 3 * 2 + 73*2, args.embedding_num * 3+ 73*2),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 3+ 73*2),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 3+ 73*2, args.embedding_num * 2+ 73*2),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 2+ 73*2),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 2+ 73*2, 73))
        self.fullConnectionLayer4 = nn.Sequential(
            nn.Linear(args.embedding_num * 3 * 2 + 73*3, args.embedding_num * 3+73*3),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 3+73*3),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 3+73*3, args.embedding_num * 2+73*3),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 2+73*3),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 2+73*3, 73))
        self.fullConnectionLayer5 = nn.Sequential(
            nn.Linear(args.embedding_num * 3 * 2 + 73*4, args.embedding_num * 3+ 73*4),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 3+ 73*4),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 3+ 73*4, args.embedding_num * 2+ 73*4),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 2+ 73*4),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 2+ 73*4, 73))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        # gnn4_embedding[drugA],,gnn4_embedding[drugB]  gnn4_embedding, gnn4_embedding,gnn4_embedding[drugA],,gnn4_embedding[drugB]gnn4_embedding,gnn4_embedding[drugA],,gnn4_embedding[drugB]
        gnn3_embedding, gnn2_embedding, gnn1_embedding, idx = arguments
        idx = idx.cpu().numpy().tolist()
        drugA = []
        drugB = []
        for i in idx:
            drugA.append(i[0])
            drugB.append(i[1])
        Embedding = torch.cat(
            [gnn1_embedding[drugA], gnn2_embedding[drugA], gnn3_embedding[drugA], \
             gnn1_embedding[drugB], gnn2_embedding[drugB], gnn3_embedding[drugB]], 1).float()

        prediction1 = self.fullConnectionLayer1(Embedding)
        Embedding1 = torch.cat([prediction1, Embedding], 1).float()
        prediction2 = self.fullConnectionLayer2(Embedding1)
        Embedding2 = torch.cat([prediction1,prediction2, Embedding], 1).float()
        prediction3 = self.fullConnectionLayer3(Embedding2)
        Embedding3 = torch.cat([prediction1,prediction2,prediction3, Embedding], 1).float()
        prediction4 = self.fullConnectionLayer4(Embedding3)
        Embedding4 = torch.cat([prediction1,prediction2,prediction3,prediction4, Embedding], 1).float()
        prediction5 = self.fullConnectionLayer5(Embedding4)

        return prediction1, prediction2, prediction3, prediction4, prediction5

        # self.fullConnectionLayer3 = nn.Sequential(
        #     nn.Linear(args.embedding_num * 3 * 2 + 73*2, args.embedding_num * 3+ 73*2),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(args.embedding_num * 3+ 73*2),
        #     nn.Dropout(args.dropout),
        #     nn.Linear(args.embedding_num * 3+ 73*2, args.embedding_num * 2+ 73*2),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(args.embedding_num * 2+ 73*2),
        #     nn.Dropout(args.dropout),
        #     nn.Linear(args.embedding_num * 2+ 73*2, 73))
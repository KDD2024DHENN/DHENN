# import argparse
# import pandas as pd
# import numpy as np
# # import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import sqlite3
# import csv
# import torch.nn.functional as F
# from torch.utils.data import DataLoader,TensorDataset,Dataset
# from sklearn.model_selection import StratifiedKFold
# # from sklearn.metrics import accuracy_score
# # from pytorchtools import EarlyStopping
# from task2_layer5 import FusionLayer,GNN1,GNN2,GNN3,GNN4
# from collections import defaultdict
# import os
# import random
#
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import auc
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
# from sklearn.preprocessing import label_binarize
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import precision_score
# import warnings
#
# warnings.filterwarnings("ignore")
#
# parser = argparse.ArgumentParser(description='GNN based on the whole datas')
# parser.add_argument("--epoches",type=int,choices=[73,500,730,2000],default=73)
# parser.add_argument("--batch_size",type=int,choices=[2048,1024,512,256,128],default=1024)
# parser.add_argument("--weigh_decay",type=float,choices=[1e-1,1e-2,1e-3,1e-4,1e-8],default=1e-8)
# parser.add_argument("--lr",type=float,choices=[1e-3,1e-4,1e-5,4*1e-3],default=5*1e-3) #5*1e-3
# parser.add_argument("--layers",type=int,choices=[1,2,3,4],default=5)
#
# parser.add_argument("--neighbor_sample_size",choices=[4,6,10,16],type=int,default=6)
# parser.add_argument("--event_num",type=int,default=65)
#
# parser.add_argument("--n_drug",type=int,default=572)
# parser.add_argument("--seed",type=int,default=2)
# parser.add_argument("--dropout",type=float,default=0.5)
# parser.add_argument("--embedding_num",type=int,choices=[128,64,256,32],default=256)
# args = parser.parse_args()
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # seed = 0
#
# random.seed(args.seed)
# os.environ['PYTHONHASHSEED'] = str(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False
#
#
# def read_dataset(drug_name_id,num):
#     kg = defaultdict(list)
#     tails = {}
#     relations = {}
#     drug_list=[]
#     filename = "./dataset/dataset"+str(num)+".txt"
#     with open(filename, encoding="utf8") as reader:
#         for line in reader:
#             string= line.rstrip().split('//',2)
#             head=string[0]
#             tail=string[1]
#             relation=string[2]
#             drug_list.append(drug_name_id[head])
#             if tail not in tails:
#                 tails[tail] = len(tails)
#             if relation not in relations:
#                 relations[relation] = len(relations)
#             if num==3:
#                 kg[drug_name_id[head]].append((drug_name_id[tail], relations[relation]))
#                 kg[drug_name_id[tail]].append((drug_name_id[head], relations[relation]))
#             else:
#                 kg[drug_name_id[head]].append((tails[tail], relations[relation]))
#     return kg,len(tails),len(relations)
#
# def prepare(mechanism, action):
#     d_label = {}
#     d_event = []
#     new_label = []
#     for i in range(len(mechanism)):
#         d_event.append(mechanism[i] + " " + action[i])
#     count = {}
#     for i in d_event:
#         if i in count:
#             count[i] += 1
#         else:
#             count[i] = 1
#     list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
#     for i in range(len(list1)):
#         d_label[list1[i][0]] = i
#     for i in range(len(d_event)):
#         new_label.append(d_label[d_event[i]])
#     return new_label,len(count)
#
# def l2_re(parameter):
#     reg=0
#     for param in parameter:
#         reg+=0.5*(param**2).sum()
#     return reg
#
# def roc_aupr_score(y_true, y_score, average="macro"):
#     def _binary_roc_aupr_score(y_true, y_score):
#         precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
#         return auc(recall, precision)
#     def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
#         if average == "binary":
#             return binary_metric(y_true, y_score)
#         if average == "micro":
#             y_true = y_true.ravel()
#             y_score = y_score.ravel()
#         if y_true.ndim == 1:
#             y_true = y_true.reshape((-1, 1))
#         if y_score.ndim == 1:
#             y_score = y_score.reshape((-1, 1))
#         n_classes = y_score.shape[1]
#         score = np.zeros((n_classes,))
#         for c in range(n_classes):
#             y_true_c = y_true.take([c], axis=1).ravel()
#             y_score_c = y_score.take([c], axis=1).ravel()
#             score[c] = binary_metric(y_true_c, y_score_c)
#         return np.average(score)
#     return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)
#
# def evaluate(pred_type, pred_score, y_test, event_num):
#     all_eval_type = 11
#     result_all = np.zeros((all_eval_type, 1), dtype=float)
#     each_eval_type = 6
#     result_eve = np.zeros((event_num, each_eval_type), dtype=float)
#     # label_binarize:返回一个one_hot的类型
#     y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
#     # pred_one_hot = label_binarize(pred_type,classes= np.arange(event_num))
#     # pred_one_hot1 = label_binarize(pred_type[0], classes=np.arange(event_num))
#     result_all[0] = accuracy_score(y_test, pred_type[0])
#     result_all[1] = roc_aupr_score(y_one_hot, pred_score[1], average='micro')
#     result_all[2] = 0.0
#     result_all[3] = roc_auc_score(y_one_hot, pred_score[2], average='micro')
#     result_all[4] = 0.0
#     result_all[5] = 0.0
#     result_all[6] = f1_score(y_test, pred_type[3], average='macro')
#     result_all[7] = 0.0
#     result_all[8] = precision_score(y_test, pred_type[4], average='macro')
#     result_all[9] = 0.0
#     result_all[10] = recall_score(y_test, pred_type[5], average='macro')
#     for i in range(event_num):
#         result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), label_binarize(pred_type[0], classes=np.arange(event_num)).take([i], axis=1).ravel())
#         result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), label_binarize(pred_type[1], classes=np.arange(event_num)).take([i], axis=1).ravel(),
#                                           average=None)
#         result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), label_binarize(pred_type[2], classes=np.arange(event_num)).take([i], axis=1).ravel(),
#                                          average=None)
#         result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), label_binarize(pred_type[3], classes=np.arange(event_num)).take([i], axis=1).ravel(),
#                                     average='binary')
#         result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), label_binarize(pred_type[4], classes=np.arange(event_num)).take([i], axis=1).ravel(),
#                                            average='binary')
#         result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), label_binarize(pred_type[5], classes=np.arange(event_num)).take([i], axis=1).ravel(),
#                                         average='binary')
#     return [result_all, result_eve]
#
#     # all_eval_type = 11
#     # result_all = np.zeros((all_eval_type, 1), dtype=float)
#     # each_eval_type = 6
#     # result_eve = np.zeros((event_num, each_eval_type), dtype=float)
#     # # label_binarize:返回一个one_hot的类型
#     # y_one_hot = label_binarize(y_test, classes = np.arange(event_num))
#     # pred_one_hot = label_binarize(pred_type,classes = np.arange(event_num))
#     # result_all[0] = accuracy_score(y_test, pred_type)
#     # result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
#     # result_all[2] = 0.0
#     # result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
#     # result_all[4] = 0.0
#     # result_all[5] = 0.0
#     # result_all[6] = f1_score(y_test, pred_type, average='macro')
#     # result_all[7] = 0.0
#     # result_all[8] = precision_score(y_test, pred_type, average='macro')
#     # result_all[9] = 0.0
#     # result_all[10] = recall_score(y_test, pred_type, average='macro')
#     # for i in range(event_num):
#     #     result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
#     #     result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
#     #                                       average=None)
#     #     result_eve[i, 2] = 0.0
#     #     result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
#     #                                 average='binary')
#     #     result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
#     #                                        average='binary')
#     #     result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
#     #                                     average='binary')
#     # return [result_all, result_eve]
#
# def save_result(filepath,result_type,result):
#     with open(filepath+result_type +'task2'+ '.csv', "w", newline='',encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile)
#         for i in result:
#             writer.writerow(i)
#     return 0
#
# def train(train_x,train_y,test_x,test_y,net,test_adj):
#     net = net.to(device)
#     loss_function = nn.CrossEntropyLoss()
#     opti = torch.optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.weigh_decay)
#     test_loss, test_acc, train_l = 0, 0, 0
#     train_a = []
#     train_x1 = train_x.copy()
#     train_x[:,[0,1]] = train_x[:,[1,0]]
#     train_x_total = torch.LongTensor(np.concatenate([train_x1, train_x], axis=0))
#     train_y = torch.LongTensor(np.concatenate([train_y,train_y]))
#     train_data = TensorDataset(train_x_total, train_y)
#     train_iter = DataLoader(train_data, args.batch_size, shuffle=True)
#
#     anslists = []
#     ansdatas = []
#     for i in range(1, 7, 1):
#         # 保存5层的数据的容器
#         anslists.append([[], [], [], [], []])
#         ansdatas.append([torch.zeros((0, 65), dtype=torch.float), torch.zeros((0, 65), dtype=torch.float),
#                          torch.zeros((0, 65), dtype=torch.float), torch.zeros((0, 65), dtype=torch.float),
#                          torch.zeros((0, 65), dtype=torch.float)])
#
#     # test_list = []
#     # max_test_output = torch.zeros((0,65),dtype=torch.float)
#     test_x = torch.LongTensor(test_x)
#     test_x = test_x.to(device)
#     f_input_test = list()
#     f_input_test.append(test_x)
#     f_input_test.append(1)
#     f_input_test.append(test_adj)
#     # f_input_test = f_input_test.to(device)
#
#     for epoch in range(args.epoches):
#         test_loss, test_score, train_l = 0, 0, 0
#         train_a = []
#         net.train()
#         for x, y in train_iter:
#             opti.zero_grad()
#             train_acc = 0
#             train_label = torch.LongTensor(y)
#             x = torch.LongTensor(x)
#             x = x.to(device)
#
#             train_label = train_label.to(device)
#
#             f_input = list()
#             f_input.append(x)
#             f_input.append(0)
#             f_input.append(defaultdict(list))
#             # f_input = x.to(f_input)
#
#             output1, output2, output3, output4, output = net(f_input)
#             l = 1 * loss_function(output, train_label) + 0.1 * loss_function(output1,
#                                                                                train_label) + 0.2 * loss_function(
#                 output2, train_label) + 0.4 * loss_function(output3, train_label) + 0.8 * loss_function(output4,
#                                                                                                       train_label)
#             l.backward()
#
#             # output = net(f_input)
#             # l = loss_function(output, train_label)
#             # l.backward()
#             opti.step()
#             train_l += l.item()
#             train_acc = accuracy_score(torch.argmax(output,dim=1).cpu(), train_label.cpu())
#             train_a.append(train_acc)
#         net.eval()
#         with torch.no_grad():
#
#             raw_output1, raw_output2, raw_output3, raw_output4, raw_output5 = net(f_input_test)
#             test_output = F.softmax(raw_output5, dim=1)
#             # for name,parameters in net.named_parameters():
#             #     print(name," : ",parameters)
#             test_output1, test_output2, test_output3, test_output4 = F.softmax(raw_output1, dim=1), F.softmax(
#                 raw_output2, dim=1), F.softmax(raw_output3, dim=1), F.softmax(raw_output4, dim=1)
#             test_label = torch.LongTensor(test_y)
#             test_label = test_label.to(device)
#             loss = loss_function(test_output, test_label)
#             test_loss = loss.item()
#             test_score = [[], [], [], [], []]
#             outputlists = [test_output1, test_output2, test_output3, test_output4, test_output]
#             for p, v in enumerate(outputlists):
#                 y_one_hot = label_binarize(test_y, classes=np.arange(65))
#                 # test_score[p].append(accuracy_score(torch.argmax(v, dim=1), test_label))
#                 # test_score[p].append(roc_aupr_score(y_one_hot, v.numpy(), average='micro'))
#                 # test_score[p].append(roc_auc_score(y_one_hot, v.numpy(), average='micro'))
#                 # test_score[p].append(f1_score(test_label, torch.argmax(v, dim=1), average='macro'))
#                 # test_score[p].append(precision_score(test_label, torch.argmax(v, dim=1), average='macro'))
#                 # test_score[p].append(recall_score(test_label, torch.argmax(v, dim=1), average='macro'))
#
#                 test_score[p].append(accuracy_score(torch.argmax(v, dim=1).cpu(), test_label.cpu()))
#                 test_score[p].append(roc_aupr_score(y_one_hot, v.cpu().numpy(), average='micro'))
#                 test_score[p].append(roc_auc_score(y_one_hot, v.cpu().numpy(), average='micro'))
#                 test_score[p].append(f1_score(test_label.cpu(),torch.argmax(v, dim=1).cpu(), average='macro'))
#                 test_score[p].append(precision_score(test_label.cpu(),torch.argmax(v, dim=1).cpu() , average='macro'))
#                 test_score[p].append(recall_score(test_label.cpu(),torch.argmax(v, dim=1).cpu() , average='macro'))
#                 for pos, j in enumerate(test_score[p]):
#                     anslists[pos][p].append(j)
#                     if j == max(anslists[pos][p]):
#                         ansdatas[pos][p] = v
#             for layer, i in enumerate(test_score):
#                 print('layer %d, acc: %f, aupr: %f, auc: %f, f1: %f, pre: %f, rec: %f' % (
#                     layer, i[0], i[1], i[2], i[3], i[4], i[5]))
#
#         print('epoch [%d] train_loss: %.6f testing_loss: %.6f train_acc: %.6f' % (
#             epoch + 1, train_l / len(train_y), test_loss / len(test_y), sum(train_a) / len(train_a)))
#     return test_loss / len(test_y), max(anslists[4][0]), train_l / len(train_y), sum(train_a) / len(
#         train_a), ansdatas
#
#
# def find_dif(raw_matrix):
#     sim_matrix4 = np.zeros((572, 572), dtype=float)
#     for i in range(572):
#         for j in range(572):
#             for k in range(20):
#                 if i==j:
#                     sim_matrix4[i,j] = 0
#                     break
#                 else:
#                     if raw_matrix[i,k] == raw_matrix[j,k]:
#                         sim_matrix4[i, j] += 1
#                     else:
#                         sim_matrix4[i, j] += 0
#     return sim_matrix4
#
# def Jaccard(matrix):
#     matrix = np.mat(matrix)
#     numerator = matrix * matrix.T
#     denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
#     return numerator / denominator
#
# def main():
#     conn = sqlite3.connect("./dataset/event.db")
#     df_drug = pd.read_sql('select * from drug;', conn)
#     extraction = pd.read_sql('select * from extraction;', conn)
#     mechanism = extraction['mechanism']
#     action = extraction['action']
#     drugA = extraction['drugA']
#     drugB = extraction['drugB']
#     new_label,event_num = prepare(mechanism, action)
#     new_label = np.array(new_label)
#     dict1 = {}
#     for i in df_drug["name"]:
#         dict1[i] = len(dict1)
#     drug_name = [dict1[i] for i in df_drug["name"]]
#     drugA_id = [dict1[i] for i in drugA]
#     drugB_id = [dict1[i] for i in drugB]
#     dataset1_kg, dataset1_tail_len, dataset1_relation_len = read_dataset(dict1,1)
#     dataset2_kg, dataset2_tail_len, dataset2_relation_len = read_dataset(dict1,2)
#     dataset3_kg, dataset3_tail_len, dataset3_relation_len = read_dataset(dict1,3)
#     dataset4_kg, dataset4_tail_len, dataset4_relation_len = read_dataset(dict1,4)
#     x_datasets = {"drugA": drugA_id, "drugB": drugB_id}
#     x_datasets = pd.DataFrame(data=x_datasets)
#     x_datasets = x_datasets.to_numpy()
#     dataset={}
#     dataset["dataset1"],dataset["dataset2"],dataset["dataset3"],dataset["dataset3"] = dataset1_kg,dataset2_kg,dataset3_kg,dataset4_kg
#     tail_len={}
#     tail_len["dataset1"],tail_len["dataset2"],tail_len["dataset3"],tail_len["dataset3"] = dataset1_tail_len,dataset2_tail_len,dataset3_tail_len,dataset4_tail_len
#     relation_len={}
#     relation_len["dataset1"],relation_len["dataset2"],relation_len["dataset3"],relation_len["dataset3"] = dataset1_relation_len,dataset2_relation_len,dataset3_relation_len,dataset4_relation_len
#     train_sum, test_sum = 0, 0
#
#     temp_kg = [defaultdict(list) for i in range(4)]
#     for p,kg in enumerate(dataset):
#         for i in dataset[kg].keys():
#             for j in dataset[kg][i]:
#                 temp_kg[p][i].append(j[0])
#
#     # feature_matrix3 = np.zeros((572, 572), dtype=float)
#     feature_matrix1 = np.zeros((572, dataset1_tail_len), dtype=float)
#     feature_matrix2 = np.zeros((572, dataset2_tail_len), dtype=float)
#     feature_matrix3 = np.zeros((572, dataset4_tail_len), dtype=float)
#     feature_matrix4 = np.zeros((572, 572), dtype=float)
#
#     for i in dataset4_kg.keys():
#         for p,v in dataset4_kg[i]:
#             feature_matrix3[i][p] = v
#
#     for i in temp_kg[0].keys():
#         for j in temp_kg[0][i]:
#             feature_matrix1[i][j] = 1
#
#     for i in temp_kg[1].keys():
#         for j in temp_kg[1][i]:
#             feature_matrix2[i][j] = 1
#
#     for i in temp_kg[2].keys():
#         for j in temp_kg[2][i]:
#             feature_matrix4[i][j] = 1
#
#
#     drug_sim1 = Jaccard(feature_matrix1)
#     # feature_matrix2 = np.mat(feature_matrix2)
#     drug_sim2 = Jaccard(feature_matrix2)
#     # feature_matrix4 = np.mat(feature_matrix4)
#     drug_sim4 = Jaccard(feature_matrix4)
#
#     drug_sim3 = find_dif(feature_matrix3)
#
#
#     temp_drugA = [[] for i in range(event_num)]
#     temp_drugB = [[] for i in range(event_num)]
#     for i in range(len(new_label)):
#         temp_drugA[new_label[i]].append(drugA_id[i])
#         temp_drugB[new_label[i]].append(drugB_id[i])
#
#     drug_cro_dict = {}
#     for i in range(event_num):
#         for j in range(len(temp_drugA[i])):
#             drug_cro_dict[temp_drugA[i][j]] = j % 5
#             drug_cro_dict[temp_drugB[i][j]] = j % 5
#
#     train_drug = [[] for i in range(5)]
#     test_drug = [[] for i in range(5)]
#     # 将全部数据分为test data 和 train data,
#     for i in range(5):
#         for dr_key in drug_cro_dict.keys():
#             if drug_cro_dict[dr_key] == i:
#                 test_drug[i].append(dr_key)
#             else:
#                 train_drug[i].append(dr_key)
#
#     test_adj = [[defaultdict(list) for _ in range(4)] for _ in range(5)]
#     for i in range(5):
#         for k in range(4):
#             for j in test_drug[i]:
#                 if k == 0:
#                     target_list = drug_sim1[j].tolist()
#                     max_v = 0
#                     current_p = []
#                     for p,v in enumerate(target_list[0]):
#                         if v > max_v and p not in test_drug[i] and p != j:
#                             max_v = v
#                             current_p = []
#                             current_p.append(p)
#                         elif v == max_v and p not in test_drug[i] and p != j:
#                             current_p.append(p)
#                     test_adj[i][k][j].append(current_p)
#                 if k == 1:
#                     target_list = drug_sim2[j].tolist()
#                     max_v = 0
#                     current_p = []
#                     for p,v in enumerate(target_list[0]):
#                         if v > max_v and p not in test_drug[i] and p != j:
#                             max_v = v
#                             current_p = []
#                             current_p.append(p)
#                         elif v == max_v and p not in test_drug[i] and p != j:
#                             current_p.append(p)
#                     test_adj[i][k][j].append(current_p)
#                 if k == 2:
#                     target_list = drug_sim3[j].tolist()
#                     max_v = 0
#                     current_p = []
#                     for p,v in enumerate(target_list):
#                         if v > max_v and p not in test_drug[i] and p != j:
#                             max_v = v
#                             current_p = []
#                             current_p.append(p)
#                         elif v == max_v and p not in test_drug[i] and p != j:
#                             current_p.append(p)
#                     test_adj[i][k][j].append(current_p)
#                 if k == 3:
#                     target_list = drug_sim4[j].tolist()
#                     max_v = 0
#                     current_p = []
#                     for p, v in enumerate(target_list[0]):
#                         if v > max_v and p not in test_drug[i] and p != j:
#                             max_v = v
#                             current_p = []
#                             current_p.append(p)
#                         elif v == max_v and p not in test_drug[i] and p != j:
#                             current_p.append(p)
#                     test_adj[i][k][j].append(current_p)
#     y_true = np.array([])
#     # y_score = np.zeros((0, 65), dtype=float)
#     # y_pred = np.array([])
#
#     y_preds = []
#     y_scores = []
#     for i in range(0,args.layers,1):
#         y_preds.append([np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])])
#         y_scores.append([np.zeros((0, 65), dtype=float),np.zeros((0, 65), dtype=float),np.zeros((0, 65), dtype=float),np.zeros((0, 65), dtype=float),np.zeros((0, 65), dtype=float),np.zeros((0, 65), dtype=float)])
#
#     for cross_ver in range(5):
#         net = nn.Sequential(GNN1(dataset, tail_len, relation_len, args, dict1, drug_name),
#                             GNN2(dataset, tail_len, relation_len, args, dict1,drug_name),
#                             GNN3(dataset, tail_len, relation_len, args, dict1,drug_name),
#                             FusionLayer(args))
#         X_train = []
#         X_test = []
#         y_train = []
#         y_test = []
#         for i in range(len(drugA)):
#             if (drugA_id[i] in np.array(train_drug[cross_ver])) and (drugB_id[i] in np.array(train_drug[cross_ver])):
#                 X_train.append(i)
#                 y_train.append(i)
#             if (drugA_id[i] not in np.array(train_drug[cross_ver])) and (drugB_id[i] in np.array(train_drug[cross_ver])):
#                 X_test.append(i)
#                 y_test.append(i)
#             if (drugA_id[i] in np.array(train_drug[cross_ver])) and (drugB_id[i] not in np.array(train_drug[cross_ver])):
#                 X_test.append(i)
#                 y_test.append(i)
#
#         train_x = x_datasets[X_train]
#         train_y = new_label[y_train]
#         test_x = x_datasets[X_test]
#         test_y = new_label[y_test]
#
#         test_loss, test_acc, train_loss, train_acc, datalists  = train(train_x, train_y, test_x, test_y,
#                                                                                    net,test_adj[cross_ver])
#         # test_loss, test_acc, train_loss, train_acc, datalists = train(train_x, train_y, test_x, test_y, net)
#
#         train_sum += train_acc
#         test_sum += test_acc
#         y_true = np.hstack((y_true, test_y))
#
#         # pred_type = torch.argmax(test_f1_output, dim=1).numpy()
#         # pred_type1 = torch.argmax(test_acc_output, dim=1).numpy()
#         # y_pred = np.hstack((y_pred, pred_type))
#         # y_pred1 = np.hstack((y_pred1, pred_type1))
#         # y_score = np.row_stack((y_score, test_f1_output))
#
#         # datalists为6维,value为5维
#         for pos, value in enumerate(datalists):
#             for p, v in enumerate(value):
#                 y_scores[p][pos] = np.row_stack((y_scores[p][pos], v.cpu()))
#                 pred_type = torch.argmax(v.cpu(), dim=1).numpy()
#                 # y_scores[p][pos] = np.row_stack((y_scores[p][pos], v))
#                 # pred_type = torch.argmax(v, dim=1).numpy()
#                 y_preds[p][pos] = np.hstack((y_preds[p][pos], pred_type))
#
#         print('fold %d, test_loss %f, test_acc %f, train_loss %f, train_acc %f' % (
#             cross_ver, test_loss, test_acc, train_loss, train_acc))
#
#     for j in range(args.layers):
#         result_all, result_eve = evaluate(y_preds[j], y_scores[j], y_true, args.event_num)
#         file1, file2 = "all" + str(j), "each" + str(j)
#         save_result("./result/", file1, result_all)
#         save_result("./result/", file2, result_eve)
#     print('%d-fold validation: avg train acc  %f, avg test acc %f' % (cross_ver, train_sum / 5, test_sum / 5))
#     return
#
#     #     train_sum += train_acc
#     #     test_sum += test_acc
#     #     pred_type = torch.argmax(test_output, dim=1).numpy()
#     #     y_pred = np.hstack((y_pred, pred_type))
#     #     y_score = np.row_stack((y_score, test_output))
#     #     y_true = np.hstack((y_true, test_y))
#     #     print('fold %d, test_loss %f, test_acc %f, train_loss %f, train_acc %f' % (
#     #             cross_ver, test_loss, test_acc, train_loss, train_acc))
#     #
#     # result_all, result_eve = evaluate(y_pred, y_score, y_true, args.event_num)
#     # save_result("../result/", "all", result_all)
#     # save_result("../result/", "each", result_eve)
#     # print('%d-fold validation: avg train acc  %f, avg test acc %f' % (5, train_sum / 5, test_sum / 5))
#     # return
#
# if __name__ == '__main__':
#     main()

import argparse
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sqlite3
import csv
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset,Dataset
from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
# from pytorchtools import EarlyStopping
from task2_layer5 import FusionLayer,GNN1,GNN2,GNN3,GNN4,Tailed_FocalLoss,FocalLoss
from collections import defaultdict
import os
import random

from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GNN based on the whole datas')
parser.add_argument("--epoches",type=int,choices=[73,500,730,2000],default=100)
parser.add_argument("--batch_size",type=int,choices=[2048,1024,512,256,128],default=1024)
parser.add_argument("--weigh_decay",type=float,choices=[1e-1,1e-2,1e-3,1e-4,1e-8],default=1e-4)
parser.add_argument("--lr",type=float,choices=[1e-3,1e-4,1e-5,4*1e-3],default=1*1e-2) #5*1e-3
parser.add_argument("--layers",type=int,choices=[1,2,3,4],default=5)

parser.add_argument("--neighbor_sample_size",choices=[4,6,10,16],type=int,default=6)
parser.add_argument("--event_num",type=int,default=73)

parser.add_argument("--n_drug",type=int,default=846)
parser.add_argument("--seed",type=int,default=1)
parser.add_argument("--dropout",type=float,default=0.4)
parser.add_argument("--embedding_num",type=int,choices=[128,64,256,32],default=384)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# seed = 0

random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

# def findthreshold(df, tail_factor=0.8):
#     df = pd.DataFrame(df)
#     value_counts = df.value_counts().values
#     # print(value_counts)
#     # print(len(value_counts))
#     # print(class_rate)
#     # os.system("pause")
#     sum_num = sum(value_counts)
#     class_rate = value_counts/sum_num
#     for i in range(len(class_rate)):
#         if tail_factor >= sum(class_rate[:i]):
#             continue
#         else:
#             threshold_point = i
#             break
#     return threshold_point

def read_dataset(drug_name_id,num):
    kg = defaultdict(list)
    tails = {}
    relations = {}
    drug_list=[]
    filename = "./dataset3/dataset"+str(num)+".txt"
    print(num)
    with open(filename, encoding="utf8") as reader:
        for line in reader:
            string= line.rstrip().split('//',2)
            head=string[0]
            tail=string[1]
            relation=string[2]
            drug_list.append(drug_name_id[head])
            if tail not in tails:
                tails[tail] = len(tails)
            if relation not in relations:
                relations[relation] = len(relations)
            if num==3:
                kg[drug_name_id[head]].append((drug_name_id[tail], relations[relation]))
                kg[drug_name_id[tail]].append((drug_name_id[head], relations[relation]))
            else:
                kg[drug_name_id[head]].append((tails[tail], relations[relation]))
    return kg,len(tails),len(relations)

def prepare(mechanism_action):
    d_label = {}
    d_event = []
    new_label = []
    for i in range(len(mechanism_action)):
        d_event.append(mechanism_action[i])
    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]] = i
    for i in range(len(d_event)):
        new_label.append(d_label[d_event[i]])
    return new_label,len(count)

def l2_re(parameter):
    reg=0
    for param in parameter:
        reg+=0.5*(param**2).sum()
    return reg

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)
    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)
    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    # label_binarize:返回一个one_hot的类型
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    # pred_one_hot = label_binarize(pred_type,classes= np.arange(event_num))
    # pred_one_hot1 = label_binarize(pred_type[0], classes=np.arange(event_num))
    result_all[0] = accuracy_score(y_test, pred_type[0])
    result_all[1] = roc_aupr_score(y_one_hot, pred_score[1], average='micro')
    result_all[2] = 0.0
    result_all[3] = roc_auc_score(y_one_hot, pred_score[2], average='micro')
    result_all[4] = 0.0
    result_all[5] = 0.0
    result_all[6] = f1_score(y_test, pred_type[3], average='macro')
    result_all[7] = 0.0
    result_all[8] = precision_score(y_test, pred_type[4], average='macro')
    result_all[9] = 0.0
    result_all[10] = recall_score(y_test, pred_type[5], average='macro')
    # for i in range(event_num):
    #     result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), label_binarize(pred_type[0], classes=np.arange(event_num)).take([i], axis=1).ravel())
    #     result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), label_binarize(pred_type[1], classes=np.arange(event_num)).take([i], axis=1).ravel(),
    #                                       average=None)
    #     result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), label_binarize(pred_type[2], classes=np.arange(event_num)).take([i], axis=1).ravel(),
    #                                      average=None)
    #     result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), label_binarize(pred_type[3], classes=np.arange(event_num)).take([i], axis=1).ravel(),
    #                                 average='binary')
    #     result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), label_binarize(pred_type[4], classes=np.arange(event_num)).take([i], axis=1).ravel(),
    #                                        average='binary')
    #     result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), label_binarize(pred_type[5], classes=np.arange(event_num)).take([i], axis=1).ravel(),
    #                                     average='binary')
    return [result_all, result_eve]


def save_result(filepath,result_type,result):
    with open(filepath+result_type +'dataset3task2gc'+ '.csv', "w", newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0

def train(train_x,train_y,test_x,test_y,net,test_adj):
    net = net.to(device)
    loss_function = nn.CrossEntropyLoss()
    opti = torch.optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.weigh_decay)
    test_loss, test_acc, train_l = 0, 0, 0
    train_a = []
    train_x1 = train_x.copy()
    train_x[:,[0,1]] = train_x[:,[1,0]]
    train_x_total = torch.LongTensor(np.concatenate([train_x1, train_x], axis=0))
    train_y = torch.LongTensor(np.concatenate([train_y,train_y]))
    train_data = TensorDataset(train_x_total, train_y)
    train_iter = DataLoader(train_data, args.batch_size, shuffle=True)
    test_x = torch.LongTensor(test_x)
    test_y = torch.LongTensor(test_y)
    test_data = TensorDataset(test_x,test_y)
    test_iter = DataLoader(test_data, args.batch_size, shuffle=False)

    anslists = []
    ansdatas = []
    for i in range(1, 7, 1):
        # 保存5层的数据的容器
        anslists.append([[], [], [], [], []])
        ansdatas.append([torch.zeros((0, 73), dtype=torch.float), torch.zeros((0, 73), dtype=torch.float),
                         torch.zeros((0, 73), dtype=torch.float), torch.zeros((0, 73), dtype=torch.float),
                         torch.zeros((0, 73), dtype=torch.float)])

    # test_x = torch.LongTensor(test_x)
    # test_x = test_x.to(device)
    # f_input_test = list()
    # f_input_test.append(test_x)
    # f_input_test.append(1)
    # f_input_test.append(test_adj)

    for epoch in range(args.epoches):
        test_loss, test_score, train_l = 0, 0, 0
        train_a = []
        net.train()
        for x, y in train_iter:
            opti.zero_grad()
            train_acc = 0
            train_label = torch.LongTensor(y)
            x = torch.LongTensor(x)
            x = x.to(device)
            train_label = train_label.to(device)

            f_input = list()
            f_input.append(x)
            f_input.append(0)
            f_input.append(defaultdict(list))

            output1, output2, output3, output4, output = net(f_input)
            l = 1 * loss_function(output, train_label) + 0.2 * loss_function(output1,
                                                                               train_label) + 0.4 * loss_function(
                output2, train_label) + 0.6 * loss_function(output3, train_label) + 0.8 * loss_function(output4,
                                                                                                      train_label)
            l.backward()

            opti.step()
            train_l += l.item()
            train_acc = accuracy_score(torch.argmax(output, dim=1).cpu(), train_label.cpu())
            # train_acc = accuracy_score(torch.argmax(output,dim=1), train_label)
            train_a.append(train_acc)
        net.eval()
        with torch.no_grad():
            test_output1, test_output2, test_output3, test_output4,test_output = np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float)
            # total_y = []
            for Test_x,Test_y in test_iter:
                Test_x = torch.LongTensor(Test_x)
                Test_x = Test_x.to(device)
                f_input_test = list()
                f_input_test.append(Test_x)
                f_input_test.append(1)
                f_input_test.append(test_adj)
                raw_output1, raw_output2, raw_output3, raw_output4, raw_output5 = net(f_input_test)
                t_output = F.softmax(raw_output5, dim=1)
                t_output1, t_output2, t_output3, t_output4 = F.softmax(raw_output1, dim=1), F.softmax(
                raw_output2, dim=1), F.softmax(raw_output3, dim=1), F.softmax(raw_output4, dim=1)
                test_label = torch.LongTensor(Test_y)
                test_label = test_label.to(device)
                loss = loss_function(t_output, test_label)
                test_loss = loss.item()
                test_output1 = np.vstack((test_output1, t_output1.cpu().numpy()))
                test_output2 = np.vstack((test_output2, t_output2.cpu().numpy()))
                test_output3 = np.vstack((test_output3, t_output3.cpu().numpy()))
                test_output4 = np.vstack((test_output4, t_output4.cpu().numpy()))
                test_output = np.vstack((test_output, t_output.cpu().numpy()))
            test_score = [[], [], [], [], []]
            test_output1, test_output2, test_output3, test_output4,test_output = torch.from_numpy(test_output1),torch.from_numpy(test_output2),torch.from_numpy(test_output3),torch.from_numpy(test_output4),torch.from_numpy(test_output)
            outputlists = [test_output1, test_output2, test_output3, test_output4, test_output]
            test_label = torch.LongTensor(test_y)
            test_label = test_label.to(device)
            for p, v in enumerate(outputlists):
                y_one_hot = label_binarize(test_y, classes=np.arange(73))
                test_score[p].append(accuracy_score(torch.argmax(v, dim=1).cpu(), test_label.cpu()))
                test_score[p].append(roc_aupr_score(y_one_hot, v.cpu().numpy(), average='micro'))
                test_score[p].append(roc_auc_score(y_one_hot, v.cpu().numpy(), average='micro'))
                test_score[p].append(f1_score(test_label.cpu(), torch.argmax(v, dim=1).cpu(), average='macro'))
                test_score[p].append(precision_score(test_label.cpu(), torch.argmax(v, dim=1).cpu(), average='macro'))
                test_score[p].append(recall_score(test_label.cpu(), torch.argmax(v, dim=1).cpu(), average='macro'))
                for pos, j in enumerate(test_score[p]):
                    anslists[pos][p].append(j)
                    if j == max(anslists[pos][p]):
                        ansdatas[pos][p] = v
            for layer, i in enumerate(test_score):
                print('layer %d, acc: %f, aupr: %f, auc: %f, f1: %f, pre: %f, rec: %f' % (
                    layer, i[0], i[1], i[2], i[3], i[4], i[5]))

        print('epoch [%d] train_loss: %.6f testing_loss: %.6f train_acc: %.6f' % (
            epoch + 1, train_l / len(train_y), test_loss / len(test_y), sum(train_a) / len(train_a)))
    return test_loss / len(test_y), max(anslists[4][0]), train_l / len(train_y), sum(train_a) / len(
        train_a), ansdatas

#
# def find_dif(raw_matrix):
#     sim_matrix4 = np.zeros((1258, 1258), dtype=float)
#     for i in range(572):
#         for j in range(572):
#             for k in range(20):
#                 if i==j:
#                     sim_matrix4[i,j] = 0
#                     break
#                 else:
#                     if raw_matrix[i,k] == raw_matrix[j,k]:
#                         sim_matrix4[i, j] += 1
#                     else:
#                         sim_matrix4[i, j] += 0
#     return sim_matrix4

def Jaccard(matrix):
    matrix = np.mat(matrix)
    numerator = matrix * matrix.T
    denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
    return numerator / denominator

def main():
    # conn = sqlite3.connect("./dataset/event.db")
    # df_drug = pd.read_sql('select * from drug;', conn)
    # extraction = pd.read_sql('select * from extraction;', conn)
    # mechanism = extraction['mechanism']
    # action = extraction['action']
    # drugA = extraction['drugA']
    # drugB = extraction['drugB']
    ddi_file_path = './dataset3/new_final_DDI.csv'
    drug_file_path = './dataset3/drug_information.csv'
    df_drug = pd.read_csv(drug_file_path)
    extraction = pd.read_csv(ddi_file_path)
    mechanism_action = extraction['Map']
    # mechanism = extraction['mechanism']
    # action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']

    new_label,event_num = prepare(mechanism_action)
    new_label = np.array(new_label)
    dict1 = {}
    for i in df_drug["id"]:
        dict1[i] = len(dict1)
    drug_name = [dict1[i] for i in df_drug["id"]]
    drugA_id = [dict1[i] for i in drugA]
    drugB_id = [dict1[i] for i in drugB]
    dataset1_kg, dataset1_tail_len, dataset1_relation_len = read_dataset(dict1,1)
    dataset2_kg, dataset2_tail_len, dataset2_relation_len = read_dataset(dict1,2)
    dataset3_kg, dataset3_tail_len, dataset3_relation_len = read_dataset(dict1,3)
    # dataset4_kg, dataset4_tail_len, dataset4_relation_len = read_dataset(dict1,4)
    x_datasets = {"drugA": drugA_id, "drugB": drugB_id}
    x_datasets = pd.DataFrame(data=x_datasets)
    x_datasets = x_datasets.to_numpy()
    dataset = {}
    dataset["dataset1"], dataset["dataset2"], dataset["dataset3"] = dataset1_kg, dataset2_kg, dataset3_kg
    tail_len = {}
    tail_len["dataset1"], tail_len["dataset2"], tail_len[
        "dataset3"] = dataset1_tail_len, dataset2_tail_len, dataset3_tail_len
    relation_len = {}
    relation_len["dataset1"], relation_len["dataset2"], relation_len[
        "dataset3"] = dataset1_relation_len, dataset2_relation_len, dataset3_relation_len
    train_sum, test_sum = 0, 0

    temp_kg = [defaultdict(list) for i in range(4)]
    for p,kg in enumerate(dataset):
        for i in dataset[kg].keys():
            for j in dataset[kg][i]:
                temp_kg[p][i].append(j[0])

    # feature_matrix3 = np.zeros((572, 572), dtype=float)
    feature_matrix1 = np.zeros((846, dataset1_tail_len), dtype=float)
    feature_matrix2 = np.zeros((846, dataset2_tail_len), dtype=float)
    # feature_matrix3 = np.zeros((572, dataset4_tail_len), dtype=float)
    feature_matrix4 = np.zeros((846, 846), dtype=float)

    # for i in dataset4_kg.keys():
    #     for p,v in dataset4_kg[i]:
    #         feature_matrix3[i][p] = v

    for i in temp_kg[0].keys():
        for j in temp_kg[0][i]:
            feature_matrix1[i][j] = 1

    for i in temp_kg[1].keys():
        for j in temp_kg[1][i]:
            feature_matrix2[i][j] = 1

    for i in temp_kg[2].keys():
        for j in temp_kg[2][i]:
            feature_matrix4[i][j] = 1


    drug_sim1 = Jaccard(feature_matrix1)
    # feature_matrix2 = np.mat(feature_matrix2)
    drug_sim2 = Jaccard(feature_matrix2)
    # feature_matrix4 = np.mat(feature_matrix4)
    drug_sim4 = Jaccard(feature_matrix4)

    # drug_sim3 = find_dif(feature_matrix3)

    temp_drugA = [[] for i in range(event_num)]
    temp_drugB = [[] for i in range(event_num)]
    for i in range(len(new_label)):
        temp_drugA[new_label[i]].append(drugA_id[i])
        temp_drugB[new_label[i]].append(drugB_id[i])

    drug_cro_dict = {}
    for i in range(event_num):
        for j in range(len(temp_drugA[i])):
            drug_cro_dict[temp_drugA[i][j]] = j % 5
            drug_cro_dict[temp_drugB[i][j]] = j % 5

    train_drug = [[] for i in range(5)]
    test_drug = [[] for i in range(5)]
    # 将全部数据分为test data 和 train data,
    for i in range(5):
        for dr_key in drug_cro_dict.keys():
            if drug_cro_dict[dr_key] == i:
                test_drug[i].append(dr_key)
            else:
                train_drug[i].append(dr_key)
                train_drug[i].append(dr_key)

    test_adj = [[defaultdict(list) for _ in range(4)] for _ in range(5)]
    for i in range(5):
        for k in range(4):
            for j in test_drug[i]:
                if k == 0:
                    target_list = drug_sim1[j].tolist()
                    max_v = 0
                    current_p = []
                    for p,v in enumerate(target_list[0]):
                        if v > max_v and p not in test_drug[i] and p != j:
                            max_v = v
                            current_p = []
                            current_p.append(p)
                        elif v == max_v and p not in test_drug[i] and p != j:
                            current_p.append(p)
                    test_adj[i][k][j].append(current_p)
                if k == 1:
                    target_list = drug_sim2[j].tolist()
                    max_v = 0
                    current_p = []
                    for p,v in enumerate(target_list[0]):
                        if v > max_v and p not in test_drug[i] and p != j:
                            max_v = v
                            current_p = []
                            current_p.append(p)
                        elif v == max_v and p not in test_drug[i] and p != j:
                            current_p.append(p)
                    test_adj[i][k][j].append(current_p)
                # if k == 2:
                #     target_list = drug_sim3[j].tolist()
                #     max_v = 0
                #     current_p = []
                #     for p,v in enumerate(target_list):
                #         if v > max_v and p not in test_drug[i] and p != j:
                #             max_v = v
                #             current_p = []
                #             current_p.append(p)
                #         elif v == max_v and p not in test_drug[i] and p != j:
                #             current_p.append(p)
                #     test_adj[i][k][j].append(current_p)
                if k == 3:
                    target_list = drug_sim4[j].tolist()
                    max_v = 0
                    current_p = []
                    for p, v in enumerate(target_list[0]):
                        if v > max_v and p not in test_drug[i] and p != j:
                            max_v = v
                            current_p = []
                            current_p.append(p)
                        elif v == max_v and p not in test_drug[i] and p != j:
                            current_p.append(p)
                    test_adj[i][k][j].append(current_p)
    y_true = np.array([])
    # y_score = np.zeros((0, 65), dtype=float)
    # y_pred = np.array([])

    y_preds = []
    y_scores = []
    for i in range(0,args.layers,1):
        y_preds.append([np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])])
        y_scores.append([np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float)])

    for cross_ver in range(5):
        net = nn.Sequential(GNN1(dataset, tail_len, relation_len, args, dict1, drug_name),
                            GNN2(dataset, tail_len, relation_len, args, dict1,drug_name),
                            GNN3(dataset, tail_len, relation_len, args, dict1,drug_name),
                            FusionLayer(args))
        # loss_fn = FocalLoss()
        # threshold = findthreshold(new_label)
        # loss_fn = Tailed_FocalLoss(threshold=threshold)
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        for i in range(len(drugA)):
            if (drugA_id[i] in np.array(train_drug[cross_ver])) and (drugB_id[i] in np.array(train_drug[cross_ver])):
                X_train.append(i)
                y_train.append(i)
            if (drugA_id[i] not in np.array(train_drug[cross_ver])) and (drugB_id[i] in np.array(train_drug[cross_ver])):
                X_test.append(i)
                y_test.append(i)
            if (drugA_id[i] in np.array(train_drug[cross_ver])) and (drugB_id[i] not in np.array(train_drug[cross_ver])):
                X_test.append(i)
                y_test.append(i)

        train_x = x_datasets[X_train]
        train_y = new_label[y_train]
        test_x = x_datasets[X_test]
        test_y = new_label[y_test]

        test_loss, test_acc, train_loss, train_acc, datalists  = train(train_x, train_y, test_x, test_y,
                                                                                   net,test_adj[cross_ver])
        # test_loss, test_acc, train_loss, train_acc, datalists = train(train_x, train_y, test_x, test_y, net)

        train_sum += train_acc
        test_sum += test_acc
        y_true = np.hstack((y_true, test_y))

        # pred_type = torch.argmax(test_f1_output, dim=1).numpy()
        # pred_type1 = torch.argmax(test_acc_output, dim=1).numpy()
        # y_pred = np.hstack((y_pred, pred_type))
        # y_pred1 = np.hstack((y_pred1, pred_type1))
        # y_score = np.row_stack((y_score, test_f1_output))

        # datalists为6维,value为5维
        for pos, value in enumerate(datalists):
            for p, v in enumerate(value):
                y_scores[p][pos] = np.row_stack((y_scores[p][pos], v.cpu()))
                pred_type = torch.argmax(v.cpu(), dim=1).numpy()
                y_preds[p][pos] = np.hstack((y_preds[p][pos], pred_type))

        print('fold %d, test_loss %f, test_acc %f, train_loss %f, train_acc %f' % (
            cross_ver, test_loss, test_acc, train_loss, train_acc))

    for j in range(args.layers):
        result_all, result_eve = evaluate(y_preds[j], y_scores[j], y_true, args.event_num)
        file1, file2 = "all" + str(j), "each" + str(j)
        save_result("./result/", file1, result_all)
        save_result("./result/", file2, result_eve)
    print('%d-fold validation: avg train acc  %f, avg test acc %f' % (cross_ver, train_sum / 5, test_sum / 5))
    return

    #     train_sum += train_acc
    #     test_sum += test_acc
    #     pred_type = torch.argmax(test_output, dim=1).numpy()
    #     y_pred = np.hstack((y_pred, pred_type))
    #     y_score = np.row_stack((y_score, test_output))
    #     y_true = np.hstack((y_true, test_y))
    #     print('fold %d, test_loss %f, test_acc %f, train_loss %f, train_acc %f' % (
    #             cross_ver, test_loss, test_acc, train_loss, train_acc))
    #
    # result_all, result_eve = evaluate(y_pred, y_score, y_true, args.event_num)
    # save_result("../result/", "all", result_all)
    # save_result("../result/", "each", result_eve)
    # print('%d-fold validation: avg train acc  %f, avg test acc %f' % (5, train_sum / 5, test_sum / 5))
    # return

if __name__ == '__main__':
    main()
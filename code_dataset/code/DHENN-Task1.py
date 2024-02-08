# task2+task3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sqlite3
import csv
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset,Dataset
from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
# from pytorchtools import EarlyStopping
# from newmodeltask1 import FusionLayer,GNN1,GNN2,GNN3,GNN4
from newmodeltask1 import FusionLayer,GNN1,GNN2,GNN3,GNN4
from collections import defaultdict
import os
import random
# newest
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='GNN based on the whole datas')
parser.add_argument("--epoches",type=int,choices=[100,500,1000,2000],default=35)
parser.add_argument("--batch_size",type=int,choices=[2048,1024,512,256,128],default=4096)
parser.add_argument("--weigh_decay",type=float,choices=[1e-1,1e-2,1e-3,1e-4,1e-8],default=1e-4)
parser.add_argument("--lr",type=float,choices=[1e-3,1e-4,1e-5,4*1e-3],default=5*1e-4) #4*1e-3
parser.add_argument("--layers",type=int,choices=[1,2,3,4],default=7)

parser.add_argument("--neighbor_sample_size",choices=[4,6,10,16],type=int,default=6)
parser.add_argument("--event_num",type=int,default=73)

parser.add_argument("--n_drug",type=int,default=846)
parser.add_argument("--seed",type=int,default=0)
parser.add_argument("--dropout",type=float,default=0.5)
parser.add_argument("--embedding_num",type=int,choices=[128,64,256,32],default=128)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed():
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

def read_dataset(drug_name_id,num):
    kg = defaultdict(list)
    tails = {}
    relations = {}
    drug_list=[]
    filename = "./dataset3/dataset"+str(num)+".txt"
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
    return new_label

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
    result_all[2] = roc_aupr_score(y_one_hot, pred_score[1], average='macro')
    result_all[3] = roc_auc_score(y_one_hot, pred_score[2], average='micro')
    result_all[4] = roc_auc_score(y_one_hot, pred_score[2], average='macro')
    result_all[5] = f1_score(y_test, pred_type[3], average='micro')
    result_all[6] = f1_score(y_test, pred_type[3], average='macro')
    result_all[7] = precision_score(y_test, pred_type[4], average='micro')
    result_all[8] = precision_score(y_test, pred_type[4], average='macro')
    result_all[9] = recall_score(y_test, pred_type[5], average='micro')
    result_all[10] = recall_score(y_test, pred_type[5], average='macro')
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), label_binarize(pred_type[0], classes=np.arange(event_num)).take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), label_binarize(pred_type[1], classes=np.arange(event_num)).take([i], axis=1).ravel(),
                                          average=None)
        result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), label_binarize(pred_type[2], classes=np.arange(event_num)).take([i], axis=1).ravel(),
                                         average=None)
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), label_binarize(pred_type[3], classes=np.arange(event_num)).take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), label_binarize(pred_type[4], classes=np.arange(event_num)).take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), label_binarize(pred_type[5], classes=np.arange(event_num)).take([i], axis=1).ravel(),
                                        average='binary')
    return [result_all, result_eve]

def save_result(filepath,result_type,result):
    with open(filepath+result_type +'task1+test_4_7'+'.csv', "w", newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0

# def save_pic(test_acc_list,test_acc_lists,file_name):
#     x = list(range(0,120,1))
#     color_list = ['y','c','m']
#     for i,y in enumerate(test_acc_lists):
#         plt.plot(x,y,color=color_list[i],label=str(max(y)))
#     plt.plot(x,test_acc_list,color='r')
#     plt.xlabel("epoches")
#     plt.ylabel(file_name)
#     plt.savefig("../result/pic"+file_name)
#     plt.show()
def train(train_x,train_y,test_x,test_y,net):
    net = net.to(device)
    loss_function=nn.CrossEntropyLoss()
    opti = torch.optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.weigh_decay)
    test_loss, test_acc, train_l = 0, 0, 0
    train_a = []
    train_x1 = train_x.copy()
    train_x[:,[0,1]] = train_x[:,[1,0]]
    train_x_total = torch.LongTensor(np.concatenate([train_x1, train_x], axis=0))
    train_y = torch.LongTensor(np.concatenate([train_y,train_y]))
    train_data = TensorDataset(train_x_total, train_y)
    train_iter = DataLoader(train_data, args.batch_size, shuffle=True)
    # test_f1_list = []
    # test_acc_list = []
    # max_f1_output = torch.zeros((0,73),dtype=torch.float)
    # max_acc_output = torch.zeros((0, 73), dtype=torch.float)
    # for i in range()
    # max_f1_outputs = [torch.zeros((0, 73), dtype=torch.float), torch.zeros((0, 73), dtype=torch.float),
    #                   torch.zeros((0, 73), dtype=torch.float), torch.zeros((0, 73), dtype=torch.float)]
    # max_acc_outputs = [torch.zeros((0, 73), dtype=torch.float), torch.zeros((0, 73), dtype=torch.float),
    #                    torch.zeros((0, 73), dtype=torch.float), torch.zeros((0, 73), dtype=torch.float)]
    # max_aupr_outputs = [torch.zeros((0, 73), dtype=torch.float), torch.zeros((0, 73), dtype=torch.float),
    #                   torch.zeros((0, 73), dtype=torch.float), torch.zeros((0, 73), dtype=torch.float)]
    # max_auc_outputs = [torch.zeros((0, 73), dtype=torch.float), torch.zeros((0, 73), dtype=torch.float),
    #                    torch.zeros((0, 73), dtype=torch.float), torch.zeros((0, 73), dtype=torch.float)]
    # max_pre_outputs = [torch.zeros((0, 73), dtype=torch.float), torch.zeros((0, 73), dtype=torch.float),
    #                   torch.zeros((0, 73), dtype=torch.float), torch.zeros((0, 73), dtype=torch.float)]
    # max_rec_outputs = [torch.zeros((0, 73), dtype=torch.float), torch.zeros((0, 73), dtype=torch.float),
    #                    torch.zeros((0, 73), dtype=torch.float), torch.zeros((0, 73), dtype=torch.float)]
    anslists = []
    ansdatas = []
    for i in range(1,7,1):
        #保存5层的数据的容器
        anslists.append([[],[],[],[],[],[],[]])
        ansdatas.append([torch.zeros((0, 73), dtype=torch.float),torch.zeros((0, 73), dtype=torch.float),torch.zeros((0, 73), dtype=torch.float), torch.zeros((0, 73), dtype=torch.float),
                      torch.zeros((0, 73), dtype=torch.float), torch.zeros((0, 73), dtype=torch.float), torch.zeros((0, 73), dtype=torch.float)])

    test_x = torch.LongTensor(test_x)
    test_x = test_x.to(device)
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
            output1, output2, output3, output4, output5, output6, output7 = net(x)
            # print(set(train_label))
            l =0.4* loss_function(output1, train_label) + 0.5* loss_function(output2, train_label) \
                + 0.6* loss_function(output3, train_label) + 0.7 * loss_function(output4, train_label) \
                + 0.8* loss_function(output5, train_label) + 0.9* loss_function(output6,train_label) + 1* loss_function(output7, train_label)

            # output1,output2,output3,output4,output = net(x)
            # l = 1*loss_function(output, train_label)+0.2*loss_function(output1, train_label)+0.4*loss_function(output2, train_label)+0.6*loss_function(output3, train_label)+0.8*loss_function(output4, train_label)
            l.backward()
            opti.step()
            train_l += l.item()
            train_acc = accuracy_score(torch.argmax(output7,dim=1).cpu(), train_label.cpu())
            train_a.append(train_acc)
        net.eval()
        with torch.no_grad():
            # test_x = torch.LongTensor(test_x)
            # raw_output1,raw_output2,raw_output3,raw_output4,raw_output5 = net(test_x)
            raw_output1, raw_output2, raw_output3, raw_output4, raw_output5, raw_output6, raw_output7 = net(test_x)
            test_output = F.softmax(raw_output7, dim=1)
            test_output1, test_output2, test_output3, test_output4, test_output5, test_output6 = \
                F.softmax(raw_output1, dim=1), F.softmax(raw_output2, dim=1), F.softmax(raw_output3, dim=1), \
                F.softmax(raw_output4, dim=1), F.softmax(raw_output5, dim=1), F.softmax(raw_output6, dim=1)
            # for name,parameters in net.named_parameters():
            #     print(name," : ",parameters)
            # test_output1,test_output2,test_output3,test_output4 = F.softmax(raw_output1, dim=1),F.softmax(raw_output2, dim=1),F.softmax(raw_output3, dim=1),F.softmax(raw_output4, dim=1)
            test_label = torch.LongTensor(test_y)
            test_label = test_label.to(device)
            loss = loss_function(test_output, test_label)
            test_loss = loss.item()
            test_score = [[], [], [], [], [], [], []]
            outputlists = [test_output1, test_output2, test_output3, test_output4, test_output5, test_output6,
                           test_output]
            for p,v in enumerate(outputlists):
                y_one_hot = label_binarize(test_y, classes=np.arange(73))
                # print(y_one_hot.dtype)
                # y_one_hot = torch.LongTensor(y_one_hot)
                test_score[p].append(accuracy_score(torch.argmax(v, dim=1).cpu(), test_label.cpu()))
                test_score[p].append(roc_aupr_score(y_one_hot, v.cpu().numpy(), average='micro'))
                test_score[p].append(roc_auc_score(y_one_hot, v.cpu().numpy(), average='micro'))
                test_score[p].append(f1_score(test_label.cpu(),torch.argmax(v, dim=1).cpu(), average='macro'))
                test_score[p].append(precision_score(test_label.cpu(),torch.argmax(v, dim=1).cpu() , average='macro'))
                test_score[p].append(recall_score(test_label.cpu(),torch.argmax(v, dim=1).cpu() , average='macro'))
                for pos,j in enumerate(test_score[p]):
                    anslists[pos][p].append(j)
                    if j==max(anslists[pos][p]):
                        ansdatas[pos][p] = v
            for layer,i in enumerate(test_score):
                print('layer %d, acc: %f, aupr: %f, auc: %f, f1: %f, pre: %f, rec: %f' % (
                    layer, i[0], i[1], i[2], i[3],i[4],i[5]))
            # test_f1_score = f1_score(torch.argmax(test_output,dim=1), test_label, average='macro')
            # test_acc_score = accuracy_score(torch.argmax(test_output,dim=1), test_label)
            #
            # test_f1_score1,test_f1_score2,test_f1_score3,test_f1_score4 = f1_score(torch.argmax(test_output1, dim=1), test_label, average='macro'),f1_score(torch.argmax(test_output2, dim=1), test_label, average='macro'),f1_score(torch.argmax(test_output3, dim=1), test_label, average='macro'),f1_score(torch.argmax(test_output4, dim=1), test_label, average='macro')
            # test_acc_score1,test_acc_score2,test_acc_score3,test_acc_score4 = accuracy_score(torch.argmax(test_output1, dim=1), test_label),accuracy_score(torch.argmax(test_output2, dim=1), test_label),accuracy_score(torch.argmax(test_output3, dim=1), test_label),accuracy_score(torch.argmax(test_output4, dim=1), test_label)
            #
            # test_f1_lists[0].append(test_f1_score1)
            # test_acc_lists[0].append(test_acc_score1)
            # test_f1_lists[1].append(test_f1_score2)
            # test_acc_lists[1].append(test_acc_score2)
            # test_f1_lists[2].append(test_f1_score3)
            # test_acc_lists[2].append(test_acc_score3)
            # test_f1_lists[3].append(test_f1_score4)
            # test_acc_lists[3].append(test_acc_score4)

            # if test_f1_score1 == max(test_f1_lists[0]):
            #     max_f1_outputs[0] = test_output1
            # if test_acc_score1 == max(test_acc_lists[0]):
            #     max_acc_outputs[0] = test_output1
            # if test_f1_score2 == max(test_f1_lists[1]):
            #     max_f1_outputs[1] = test_output2
            # if test_acc_score2 == max(test_acc_lists[1]):
            #     max_acc_outputs[1] = test_output2
            # if test_f1_score3 == max(test_f1_lists[2]):
            #     max_f1_outputs[2] = test_output3
            # if test_acc_score3 == max(test_acc_lists[2]):
            #     max_acc_outputs[2] = test_output3
            # if test_f1_score4 == max(test_f1_lists[3]):
            #     max_f1_outputs[3] = test_output4
            # if test_acc_score4 == max(test_acc_lists[3]):
            #     max_acc_outputs[3] = test_output4

            # test_f1_list.append(test_f1_score)
            # test_acc_list.append(test_acc_score)
            # if test_f1_score == max(test_f1_list):
            #     max_f1_output = test_output
            # if test_acc_score == max(test_acc_list):
            #     max_acc_output = test_output
        print('epoch [%d] train_loss: %.6f testing_loss: %.6f train_acc: %.6f' % (
                epoch + 1, train_l / len(train_y), test_loss / len(test_y), sum(train_a) / len(train_a)))
    # save_pic(test_acc_list,test_acc_lists,"acc")
    # save_pic(test_f1_list,test_f1_lists,"f1")
    # print(" ********************************************************** ")
    # evaluate_list = ["acc","aupr","auc","f1","pre","reca"]
    # for p,v in enumerate(evaluate_list):
    #     print("layer",p,)


    # print(" layer1_accvalue ", " : ", max(test_acc_lists[0]))
    # print(" layer2_accvalue ", " : ", max(test_acc_lists[1]))
    # print(" layer3_accvalue ", " : ", max(test_acc_lists[2]))
    # print(" layer4_accvalue ", " : ", max(test_acc_lists[3]))
    # print(" layer5_accvalue ", " : ", max(test_acc_list))
    # print(" layer1_f1value ", " : ", max(test_f1_lists[0]))
    # print(" layer2_f1value ", " : ", max(test_f1_lists[1]))
    # print(" layer3_f1value ", " : ", max(test_f1_lists[2]))
    # print(" layer4_f1value ", " : ", max(test_f1_lists[3]))
    # print(" layer5_f1value ", " : ", max(test_f1_list))
    # print(" ********************************************************** ")
    return test_loss / len(test_y), max(anslists[4][0]), train_l / len(train_y), sum(train_a) / len(
        train_a), ansdatas

def main():
    ddi_file_path = './dataset3/new_final_DDI.csv'
    drug_file_path = './dataset3/two_drugs.csv'
    df_drug = pd.read_csv(drug_file_path)
    extraction = pd.read_csv(ddi_file_path)
    mechanism_action = extraction['Map']
    drugA = extraction['drugA']
    drugB = extraction['drugB']
    # conn = sqlite3.connect("./dataset/event.db")
    # df_drug = pd.read_sql('select * from drug;', conn)
    # extraction = pd.read_sql('select * from extraction;', conn)
    # mechanism = extraction['mechanism']
    # action = extraction['action']
    # drugA = extraction['drugA']
    # drugB = extraction['drugB']
    new_label = prepare(mechanism_action)
    new_label = np.array(new_label)
    dict1 = {}
    for i in df_drug["name"]:
        dict1[i] = len(dict1)
    drug_name = [dict1[i] for i in df_drug["name"]]
    drugA_id = [dict1[i] for i in drugA]
    drugB_id = [dict1[i] for i in drugB]
    dataset1_kg, dataset1_tail_len, dataset1_relation_len = read_dataset(dict1,1)
    dataset2_kg, dataset2_tail_len, dataset2_relation_len = read_dataset(dict1,2)
    dataset3_kg, dataset3_tail_len, dataset3_relation_len = read_dataset(dict1,3)
    # dataset4_kg, dataset4_tail_len, dataset4_relation_len = read_dataset(dict1, 4) ,dataset["dataset3"] ,dataset4_kg ,tail_len["dataset3"] ,,dataset4_relation_len
    train_sum, test_sum = 0, 0
    # relation_len["dataset3"] ,dataset4_tail_len
    x_datasets = {"drugA": drugA_id, "drugB": drugB_id}
    x_datasets = pd.DataFrame(data=x_datasets)
    x_datasets = x_datasets.to_numpy()
    dataset={}
    dataset["dataset1"],dataset["dataset2"],dataset["dataset3"] = dataset1_kg,dataset2_kg,dataset3_kg
    tail_len={}
    tail_len["dataset1"],tail_len["dataset2"],tail_len["dataset3"] = dataset1_tail_len,dataset2_tail_len,dataset3_tail_len
    relation_len={}
    relation_len["dataset1"],relation_len["dataset2"],relation_len["dataset3"]= dataset1_relation_len,dataset2_relation_len,dataset3_relation_len

    y_true = np.array([])
    # y_score = np.zeros((0, 73), dtype=float)
    # y_pred = np.array([])
    # y_pred1 = np.array([])
    # y_scores = [np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float)]
    #
    # f1_preds = [np.array([]),np.array([]),np.array([]),np.array([])]
    # acc_preds = [np.array([]),np.array([]),np.array([]),np.array([])]

    y_preds = []
    y_scores = []
    for i in range(0,args.layers,1):
        y_preds.append([np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])])
        y_scores.append([np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float),np.zeros((0, 73), dtype=float)])

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    kfold = kf.split(x_datasets, new_label)
    for i, (train_idx, test_idx) in enumerate(kfold):
        net = nn.Sequential(GNN1(dataset, tail_len, relation_len, args, dict1, drug_name),
                            GNN2(dataset, tail_len, relation_len, args, dict1, drug_name),
                            GNN3(dataset, tail_len, relation_len, args, dict1, drug_name),
                            FusionLayer(args))
        # net = nn.Sequential(FusionLayer(args,drug_embedding))
        # GNN4(dataset, tail_len, relation_len, args, dict1, drug_name),
        train_x = x_datasets[train_idx]
        train_y = new_label[train_idx]
        test_x = x_datasets[test_idx]
        test_y = new_label[test_idx]
        test_loss, test_acc, train_loss, train_acc,datalists = train(train_x,train_y,test_x,test_y,net)
        train_sum += train_acc
        test_sum += test_acc
        y_true = np.hstack((y_true, test_y))

        # pred_type = torch.argmax(test_f1_output, dim=1).numpy()
        # pred_type1 = torch.argmax(test_acc_output, dim=1).numpy()
        # y_pred = np.hstack((y_pred, pred_type))
        # y_pred1 = np.hstack((y_pred1, pred_type1))
        # y_score = np.row_stack((y_score, test_f1_output))

        # datalists为6维,value为5维
        for pos,value in enumerate(datalists):
            for p,v in enumerate(value):
                y_scores[p][pos] = np.row_stack((y_scores[p][pos], v.cpu()))
                pred_type = torch.argmax(v.cpu(), dim=1).numpy()
                y_preds[p][pos] = np.hstack((y_preds[p][pos], pred_type))
        # for s,f1_output in enumerate(test_f1_outputs):
        #     f1_pred = torch.argmax(f1_output, dim=1).numpy()
        #     f1_preds[s] = np.hstack((f1_preds[s], f1_pred))
        # for s,acc_output in enumerate(test_acc_outputs):
        #     acc_pred = torch.argmax(acc_output, dim=1).numpy()
        #     acc_preds[s] = np.hstack((acc_preds[s], acc_pred))
        # for s,f1_output in enumerate(test_f1_outputs):
        #     y_scores[s] = np.row_stack((y_scores[s], f1_output))

        print('fold %d, test_loss %f, test_acc %f, train_loss %f, train_acc %f' % (
            i, test_loss, test_acc, train_loss, train_acc))
        # if i== 0:
        #     break
    # result_all, result_eve = evaluate(y_pred, y_score, y_true, args.event_num,y_pred1)
    # save_result("../result/", "all", result_all)
    # save_result("../result/", "each", result_eve)

    for j in range(args.layers):
        result_all, result_eve = evaluate(y_preds[j], y_scores[j], y_true, args.event_num)
        file1,file2 = "all"+str(j),"each"+str(j)
        save_result("./result/", file1, result_all)
        save_result("./result/", file2, result_eve)


    print('%d-fold validation: avg train acc  %f, avg test acc %f' % (i, train_sum / 5, test_sum / 5))
    return

if __name__ == '__main__':
    setup_seed()
    main()
# -*- coding: utf-8 -*-

import  torch
import torch.nn as nn
import torch.utils.data as Data
from datasets import DatasetDict
from transformers import AutoTokenizer,AutoModel,BertForSequenceClassification,BertTokenizer,BertModel
from torch.nn.utils.weight_norm import weight_norm
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_curve,roc_auc_score,matthews_corrcoef, confusion_matrix,precision_recall_curve,auc
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("start")
print("build model")
device = torch.device("cuda", 0)
print("device done")
model_name_1 = "/pretrained_model/DNABERT_6"
model_name_2 = "/pretrained_model/gena_lm_bert_base_t2t"
batch_SIZE = 8
epoch_NUM = 100
patience_NUM =10

def calc_metrics(y_true,y_pred,y_score):
    precision = precision_score(y_true, y_pred, pos_label=1, average="binary")
    recall = recall_score(y_true, y_pred, pos_label=1, average="binary")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sp = cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])
    ACC = accuracy_score(y_true, y_pred)
    AUC = roc_auc_score(y_true, y_score)
    FPR, TPR, _ = roc_curve(y_true, y_score, pos_label=1)
    ROC = auc(FPR, TPR)
    pre ,rec, _ = precision_recall_curve(y_true,y_score,pos_label=1)
    AUPRC = auc(rec,pre)
    results = [ACC,AUC,AUPRC,precision,recall,sp]
    roc_data = [FPR,TPR,ROC]
    auprc_data = [pre,rec,AUPRC]
    return results,roc_data,auprc_data

def evaluate(data_iter, net, criterion):
    pred_prob = []
    label_pred = []
    label_real = []
    total_loss = 0.0
    num_batches = 0
    for j, (data, labels) in enumerate(data_iter, 0):
        labels = labels.to(device)
        output,attn= net(data)
        loss = criterion(output, labels)
        total_loss += loss.item()
        num_batches += 1
        outputs_cpu = output.cpu()
        y_cpu = labels.cpu()
        pred_prob_positive = outputs_cpu[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + output.argmax(dim=1).tolist()
        label_real = label_real + y_cpu.tolist()
    performance, roc_data, prc_data = calc_metrics(label_real, label_pred, pred_prob)
    average_loss = total_loss / num_batches
    return performance, roc_data, prc_data, average_loss

def save_metrics(metrics, roc_data, prc_data, fold, epoch, data_type):
    os.makedirs(f'results/{fold}/metrics', exist_ok=True)
    os.makedirs(f'results/{fold}/roc', exist_ok=True)
    os.makedirs(f'results/{fold}/prc', exist_ok=True)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'results/{fold}/metrics/{data_type}_metrics.csv', index=False)
    roc_df = pd.DataFrame({
        "FPR": roc_data[0],
        "TPR": roc_data[1],
        "ROC": [roc_data[2]] * len(roc_data[0])# AUC is a single value, we repeat it to match the length of FPR and TPR
    })
    roc_df.to_csv(f'results/{fold}/roc/{data_type}_roc_epoch_{epoch + 1}.csv', index=False)
    prc_df = pd.DataFrame({
        "Precision": prc_data[0],
        "Recall": prc_data[1],
        "AUPRC": [prc_data[2]] * len(prc_data[0])# AUPRC is a single value, we repeat it to match the length of Precision and Recall
    })
    prc_df.to_csv(f'results/{fold}/prc/{data_type}_prc_epoch_{epoch + 1}.csv', index=False)

class EarlyStopping:
    def __init__(self, patience=patience_NUM, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class RNADataset(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = torch.tensor(label)
    def __getitem__(self, i):
        return self.data[i], self.label[i]
    def __len__(self):
        return len(self.label)

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    Modified from https://github.com/jnhwkim/ban-vqa/blob/master/fc.py
    """
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()
        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3): #k是最后sumpooling时的stride=3
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out
        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps

class HybridBertClssificationModel(nn.Module):
    def __init__(self, device='cuda'):
        super(HybridBertClssificationModel, self).__init__()
        self.device = device
        self.tokenizer_1 = BertTokenizer.from_pretrained(model_name_1)
        self.bert_1 = BertModel.from_pretrained(model_name_1, trust_remote_code = True).to(device)
        self.tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)
        self.bert_2 = AutoModel.from_pretrained(model_name_2, trust_remote_code = True).to(device)
        self.bcn = weight_norm(BANLayer(v_dim=768, q_dim=768, h_dim=768, h_out=2),name = 'h_mat',dim=None)
        self.fc = nn.Sequential(nn.Linear(768,128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Linear(128,64),
                                nn.BatchNorm1d(64),
                                nn.ReLU(),
                                nn.Linear(64,2)
                                )

    def kmer_trans(self,sentence,k):
        kmer = [sentence[i:i+k] for i in range(len(sentence)-k+1)]
        kmers = ' '.join(kmer)
        return kmers

    def forward(self, batch_seqs):
        batch_seqs = list(batch_seqs)
        k = 6
        batch_seq_kmer = [self.kmer_trans(seq,k) for seq in batch_seqs]
        token_seq_kmer = self.tokenizer_1(batch_seq_kmer,truncation=True,return_tensors='pt',max_length=512)
        input_ids_kmer, token_type_ids_kmer, attention_mask_kmer = token_seq_kmer['input_ids'], token_seq_kmer['token_type_ids'], token_seq_kmer['attention_mask']

        rna_code_kmer = self.bert_1(input_ids=input_ids_kmer.to(self.device),
                             token_type_ids=token_type_ids_kmer.to(self.device),
                             attention_mask=attention_mask_kmer.to(self.device))['last_hidden_state']
        kmer_cls_code = rna_code_kmer
        batch_seq_gena = [seq for seq in batch_seqs]
        token_seq_gena = self.tokenizer_2(batch_seq_gena, add_special_tokens=True, padding=True, return_tensors='pt')
        input_ids_gena, token_type_ids_gena, attention_mask_gena = token_seq_gena['input_ids'], token_seq_gena['token_type_ids'], token_seq_gena['attention_mask']
        rna_code_gena = self.bert_2(input_ids=input_ids_gena.to(self.device),
                             token_type_ids = token_type_ids_gena.to(self.device),
                             attention_mask=attention_mask_gena.to(self.device),output_hidden_states=True)['hidden_states']
        gena_last_hidden_state_code = rna_code_gena[-1]
        gena_cls_code = gena_last_hidden_state_code
        hybrid_code, atten = self.bcn(kmer_cls_code, gena_cls_code)
        pred = self.fc(hybrid_code)

        return pred, atten

def main():
    print("loading data...")
    for k in range(5):
        print("-" * 30 + "k-fold: " + f'{k + 1}' + "-" * 30)
        train_dataset = DatasetDict.from_csv({"train": f"cls_data/cls_train_{k + 1}.csv"})
        train_dataset = train_dataset['train']
        valid_dataset = DatasetDict.from_csv({"valid": f"cls_data/cls_val_{k + 1}.csv"})
        valid_dataset = valid_dataset['valid']
        test_dataset = DatasetDict.from_csv({"test": f"cls_data/cls_test.csv"})
        test_dataset = test_dataset['test']
        dataset = DatasetDict()
        dataset["train"] = train_dataset
        dataset["valid"] = valid_dataset
        dataset["test"] = test_dataset
        traindataset = RNADataset(train_dataset['seq'], train_dataset['label'])
        validdataset = RNADataset(valid_dataset['seq'], valid_dataset['label'])
        testdataset = RNADataset(test_dataset['seq'], test_dataset['label'])
        train_loader = Data.DataLoader(traindataset, batch_size=batch_SIZE, shuffle=True)
        valid_loader = Data.DataLoader(validdataset, batch_size=batch_SIZE, shuffle=False)
        test_loader = Data.DataLoader(testdataset, batch_size=batch_SIZE, shuffle=False)
        epoch_num = epoch_NUM
        model = HybridBertClssificationModel(device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.000001, weight_decay=0.0001)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=0)
        early_stopping = EarlyStopping()
        print('model loading done!')
        print('start training...')
        best_valid_auc = 0
        lr_list = []
        train_metrics = []
        valid_metrics = []
        test_metrics = []
        for epoch in range(epoch_num):
            train_loss_ls = []
            t0 = time.time()
            model.train()
            for i, (data, labels) in enumerate(train_loader):
                if len(data) <= 1:
                    continue
                labels = labels.to(device)
                output,att = model(data)
                optimizer.zero_grad()
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                train_loss_ls.append(loss.item())
            scheduler.step()
            lr_list.append({'LR': optimizer.param_groups[0]['lr']})
            model.eval()
            with torch.no_grad():
                valid_results, valid_roc_curve, valid_prc_curve, valid_loss = evaluate(valid_loader, model, criterion)
                test_results, test_roc_curve, test_prc_curve, test_loss = evaluate(test_loader, model, criterion)

            print(f'\nEpoch:{epoch + 1}, loss:{np.mean(train_loss_ls):.5f}, time:{time.time() - t0:.2f}\n',
                  f'Valid_ACC:{valid_results[0]:.4f}|Valid_AUC:{valid_results[1]:.4f}|Valid_AUPRC:{valid_results[2]:.4f}|Valid_precision:{valid_results[3]:.4f}|Valid_recall:{valid_results[4]:.4f}|Valid_sp:{valid_results[5]:.4f}|Valid_loss:{valid_loss:.4f}\n'
                  f'Test_ACC:{test_results[0]:.4f}|Test_AUC:{test_results[1]:.4f}|Test_AUPRC:{test_results[2]:.4f}|Test_precision:{test_results[3]:.4f}|Test_recall:{test_results[4]:.4f}|Test_sp:{test_results[5]:.4f}|Test_loss:{test_loss:.4f}\n'
                  )
            # save metrics of 5-cv and test dataset
            valid_metrics.append({'epoch': epoch + 1,
                                  'ACC': valid_results[0],
                                  'AUC': valid_results[1],
                                  'AUPRC': valid_results[2],
                                  'precision': valid_results[3],
                                  'recall': valid_results[4],
                                  'sp': valid_results[5],
                                  'loss': f'{valid_loss:.4f}'})
            test_metrics.append({'epoch': epoch + 1,
                                 'ACC': test_results[0],
                                 'AUC': test_results[1],
                                 'AUPRC': test_results[2],
                                 'precision': test_results[3],
                                 'recall': test_results[4],
                                 'sp': test_results[5],
                                 'loss': f'{test_loss:.4f}'})
            valid_auc = valid_results[1]
            if valid_auc > best_valid_auc:
                best_valid_auc = valid_auc
                os.makedirs(f'Saved_Best_Models/{k + 1}', exist_ok=True)
                save_path_pt = f'Saved_Best_Models/{k + 1}/best_model.pt'
                print(f'Saving model: {k + 1}fold {epoch + 1}epoch')
                torch.save(model.state_dict(), save_path_pt, _use_new_zipfile_serialization=False)
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        save_metrics(valid_metrics, valid_roc_curve, valid_prc_curve, k + 1, epoch, "valid")
        save_metrics(test_metrics, test_roc_curve, test_prc_curve, k + 1, epoch, "test")
if __name__ == '__main__':
    main()



















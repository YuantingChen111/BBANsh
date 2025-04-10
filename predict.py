# -*- coding: utf-8 -*-

from train import *
import torch
import torch.nn as nn
import torch.utils.data as Data
from datasets import DatasetDict
import pandas as pd
import numpy as np

def load_model(filepath,model_class,device='cuda'):
    model = model_class(device=device)
    model.load_state_dict(torch.load(filepath,map_location=device))
    model.to(device)
    return model

def external_evaluate(data_iter,net):
    results = []
    softmax = nn.Softmax(dim=1)
    for j, (data, labels) in enumerate(data_iter, 0):
        output, attn = net(data)
        probabilities = softmax(output).detach().cpu().numpy()
        predicted_labels = np.argmax(probabilities, axis=1)
        for prob, label in zip(probabilities, predicted_labels):
            results.append({'pred_label': label})
    return results

def main():
    print("loading data...")
    model = load_model('/Saved_Best_Models/best_model.pt',HybridBertClssificationModel,device='cuda')
    print('loading model done')
    model.eval()
    external_dataset = DatasetDict.from_csv({'train': f'external_data/external.csv'})
    external_dataset = external_dataset['external']
    dataset = DatasetDict()
    dataset['external'] = external_dataset
    external_Dataset = RNADataset(external_dataset['seq'], external_dataset['label'])
    external_loader = Data.DataLoader(external_Dataset,batch_size=batch_SIZE, shuffle=False)
    with torch.no_grad():
        pred= external_evaluate(external_loader,model)
    df = pd.DataFrame(pred)
    df.to_csv('external_pred.csv',index=False)
    print('predict done!')
if __name__ == '__main__':
    main()

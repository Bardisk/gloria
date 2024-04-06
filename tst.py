#!/bin/env -S python -i
import time
basetime = time.time()
def report_time(str=''):
  print(str + '\n' + f'time elapsed: {time.time() - basetime:.2f} s')

import torch,gloria
import numpy as np
from sys import argv
from datasets.general import CXR_dataset
from torch.utils.data import DataLoader

report_time('Import finished!')

if len(argv) < 4:
  raise ValueError('Usage ./tst.py NAME DEVICE BATCHSIZE [THRESHOLD]')

name = argv[1]
device = argv[2]
batch_size = int(argv[3])

threshold = 0.5
if len(argv) >= 5:
  threshold = float(argv[4])

model = gloria.load_gloria(device=device)
dataset = CXR_dataset(name, device=device)

report_time('Model and dataset loaded!')

# init classes and text embedding
prompts = [f'Findings suggesting {class_name}.' for class_name in dataset.classnames]\
        + [f'No evidence of {class_name}.' for class_name in dataset.classnames]
processed_txt = model.process_text(prompts, device)
cls_nr = len(dataset.classnames)

# init dataloader
loader = DataLoader(dataset, batch_size=batch_size)

# for output
y_pred = []
y_label = []
y_true = []

batch_id = 0
similarities_bak = None
softmax = torch.nn.Softmax(dim=1)

for img, _, labels in loader:
  batch_id += 1
  report_time(f'Batch {batch_id}...')
  sample_nr = img.shape[0]

  # stack gts
  y_true.append(labels)
  
  # infer the results with the model
  similarities_bak = similarities = gloria.get_similarities(
    model, img, processed_txt
  ).view(sample_nr, 2, cls_nr)
  probs = softmax(similarities)[:, 0, :]
  labels_pred = (probs > threshold).type(torch.int)

  # stack the results
  y_pred.append(probs)
  y_label.append(labels_pred)
  
  break

y_pred = torch.vstack(y_pred)
y_label = torch.vstack(y_label)
y_true = torch.vstack(y_true)

report_time('done')
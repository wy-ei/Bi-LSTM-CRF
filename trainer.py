import os
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from metric import NER_Metric

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def train_one_epoch(model, optimizer, train_dl, epoch, device):
    steps = 0
    total_loss = 0.
    batch_size = train_dl.batch_size
    
    total_steps = int(len(train_dl.dataset) / batch_size)

    for sequence, tags in train_dl:
        optimizer.zero_grad()

        sequence_cuda = sequence.to(device)
        tags_cuda = tags.to(device)
        mask_cuda = sequence_cuda > 0

        loss = model(sequence_cuda, tags_cuda, mask_cuda)

        loss.backward()

        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()
        
        total_loss += loss.item()

        steps += 1

        if steps % 400 == 0:
            print('epoch: {} - progress: {:.2f}'.format(epoch, steps / total_steps))

    train_loss = total_loss / (steps * batch_size)

    return train_loss


def train(model, optimizer, train_dl, val_dl, device=None, epochs=10, early_stop=False, early_stop_epochs=3, save_every_n_epochs=1):
        
    history = {
        'acc': [],
        'loss': [],
        'val_acc': [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        
        train_loss = train_one_epoch(model, optimizer, train_dl, epoch, device)
        
        with torch.no_grad():
            val_metric = evaluate(model, val_dl, device=device)
            train_metric = evaluate(model, train_dl, device=device)
            
            train_acc = train_metric.global_precision
            val_acc = val_metric.global_precision

        history['acc'].append(train_acc)
        history['loss'].append(train_loss)
        history['val_acc'].append(val_acc)

        
        logger.info("epoch {} - loss: {:.2f} acc: {:.2f} - val_acc: {:.2f}"\
              .format(epoch, train_loss, train_acc, val_acc))
        
        if (epoch + 1) % save_every_n_epochs == 0:
            save_model(model, epoch, val_acc)

        if early_stop and epoch > early_stop_epochs:
            if history['val_acc'][-1] < history['val_acc'][-early_stop_epochs]:
                logger.info('early stop')
                save_model(model, epoch, val_acc)
                break
            
    return history


def predict(model, dl, device=None):
    tags_pred_list = []
    with torch.no_grad():
        for sequence, tags in dl:
            sequence_cuda = sequence.to(device)
            mask_cuda = sequence_cuda > 0

            tags_pred = model.predict(sequence_cuda, mask_cuda)
            tags_pred.extend(tags_pred_list)

    return tags_pred_list

def evaluate(model, dl, device=None):
    model.eval()
    
    metric = NER_Metric()

    with torch.no_grad():
        for sequence, tags in dl:
            sequence_cuda = sequence.to(device)
            mask_cuda = sequence_cuda > 0

            tags_pred = model.predict(sequence_cuda, mask_cuda)
            
            for i in range(len(tags)):
                metric.update(tags[i].numpy(), tags_pred[i])
     
    return metric

def save_model(model, epoch, acc):
    model_file_name = 'model_{}_epoch_{}_acc_{:.2f}'.format(model.name, epoch, acc)
    model_dir = './model/'
    save_path = os.path.join(model_dir, '{}.tar'.format(model_file_name))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({
        'model': model.state_dict(),
    }, save_path)

def load_model(model, model_file_name):
    model_dir = './model/'
    model_file = os.path.join(model_dir, model_file_name)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model'])
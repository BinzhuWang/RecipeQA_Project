import os
import logging
import pickle
import torch
import torch.nn as nn
import sys
import json
import torch.nn.functional as F

import numpy as np


from collections import defaultdict
from torch.serialization import default_restore_location


def move_to_cuda(sample):
    if torch.is_tensor(sample):
        return sample.cuda()
    elif isinstance(sample, list):
        return [move_to_cuda(x) for x in sample]
    elif isinstance(sample, dict):
        return {key: move_to_cuda(value) for key, value in sample.items()}
    else:
        return sample

def save_checkpoint(args, model, optimizer, epoch, valid_loss):
    os.makedirs(args.save_dir, exist_ok=True)
    last_epoch = getattr(save_checkpoint, 'last_epoch', -1)
    save_checkpoint.last_epoch = max(last_epoch, epoch)
    prev_best = getattr(save_checkpoint, 'best_loss', float('inf'))
    save_checkpoint.best_loss = min(prev_best, valid_loss)

    state_dict = {
        'epoch': epoch,
        'val_loss': valid_loss,
        'best_loss': save_checkpoint.best_loss,
        'last_epoch': save_checkpoint.last_epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': args,
    }

    if args.epoch_checkpoints and epoch % args.save_interval == 0:
        torch.save(state_dict, os.path.join(args.save_dir, 'checkpoint{}_{:.3f}.pt'.format(epoch, valid_loss)))
    if valid_loss < prev_best:
        torch.save(state_dict, os.path.join(args.save_dir, 'checkpoint_best.pt'))
    if last_epoch < epoch:
        torch.save(state_dict, os.path.join(args.save_dir, 'checkpoint_last.pt'))

def accuracy(preds,y):
    preds = F.softmax(preds,dim=1)
    correct = 0
    pred = preds.max(1,keepdim = True)[1]
    correct += pred.eq(y.view_as(pred)).sum().item()
    acc = correct / len(y)

    return acc


def shuffle_data(recipe_context,recipe_question,recipe_choice,recipe_answer):
    #shuffle
    combine = list(zip(recipe_context,recipe_question,recipe_choice,recipe_answer))
    np.random.shuffle(combine)
    recipe_context_shuffled,recipe_question_shuffled, recipe_choice_shuffled, recipe_answer_shuffled = zip(*combine)
    recipe_context_shuffled = list(recipe_context_shuffled)
    recipe_question_shuffled = list(recipe_question_shuffled)
    recipe_choice_shuffled = list(recipe_choice_shuffled)
    recipe_answer_shuffled = list(recipe_answer_shuffled)
    return recipe_context_shuffled,recipe_question_shuffled, recipe_choice_shuffled, recipe_answer_shuffled


def save_model(model,epoch, accuracy,saved_path):
    torch.save(model.state_dict(),
               '%s/hasty_student_epoch_%d_%f_acc.pth' % (saved_path, epoch,accuracy))
    print('Save model with accuracy:', accuracy)


def log_data(log_path, train_loss, train_accuracy, val_loss, val_accuracy):
    file = open(log_path, 'a')
    if torch.cuda.is_available():
        data = str(train_loss) + ' ' + str(f'{train_accuracy:.2f}') \
               + ' ' + str(val_loss.cpu().numpy()) + ' ' + str(f'{val_accuracy:.2f}')
    else:
        data = str(train_loss) + ' ' + str(f'{train_accuracy:.2f}') \
               + ' ' + str(val_loss.numpy()) + ' ' + str(f'{val_accuracy:.2f}')
    file.write(data)
    file.write('\n')
    file.close()


def load_cleaned_data(file = 'train_cleaned.json'):
    file = open(file,'r',encoding='utf8').read()
    recipe = json.loads(file)

    recipe_context = recipe['context']
    recipe_answer = recipe['answer']
    recipe_choice = recipe['choice']
    recipe_question = recipe['question']
    recipe_images = recipe['images']

    return recipe_context, recipe_images, recipe_question, recipe_choice, recipe_answer

def split_batch(batch_size, recipe_context_new, recipe_question_new, recipe_choice_new, recipe_answer_new):
    train_context = []
    train_question = []
    train_answer = []
    train_choice = []
    for i in range(0, len(recipe_question_new), batch_size):
        train_context.append(recipe_context_new[i : i + batch_size])
        train_question.append(recipe_question_new[i : i + batch_size])
        train_choice.append(recipe_choice_new[i : i + batch_size])
        actual_scores = recipe_answer_new[i : i + batch_size]
        if torch.cuda.is_available():
            actual_scores = torch.LongTensor(actual_scores).cuda()
        else:
            actual_scores = torch.LongTensor(actual_scores)
        train_answer.append(actual_scores)
    return train_context, train_question, train_choice, train_answer


def transport_1_0_2(a):
        max_step = 0
        for i in a:
            if max_step < len(i):
                max_step = len(i)
        new = []
        for i in range(max_step):
            step = []
            for j in a:
                if len(j) <= i:
                    step.append(['0','0'])
                else:
                    step.append(j[i])
            new.append(step)
        return new
import torch
import torch.nn as nn
from torch.autograd import Variable
from Elmo import elmo_embedding
from utils import transport_1_0_2
import numpy as np
import torch.nn.functional as F
import  torch.optim as optim
import argparse
from tqdm import tqdm
from QaNet import *
from utils import shuffle_data, save_model, log_data, load_cleaned_data, accuracy, split_batch
import time

def get_args():

    parser = argparse.ArgumentParser('QaNet_Model')
    parser.add_argument('--cuda',default=False,help= 'Use a GPU')

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--word_hidden_size", type=int, default=256)
    parser.add_argument("--sent_hidden_size", type=int, default=256)
    parser.add_argument("--log_path", type=str, default="/Users/wangbinzhu/Desktop/Final/log.txt")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--max_len", type = int, default = 20)
    args = parser.parse_args()
    return args


def train(model, train_context,train_question,train_answer,train_choice,optimizer,criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch_context,batch_question,batch_answer,batch_choice in tqdm(zip(train_context,train_question,train_answer,train_choice)):
        optimizer.zero_grad()
        output = model(batch_context,batch_question,batch_choice)
        output = torch.cat(output,0).view(-1,len(batch_question))
        output = output.permute(1,0)
        loss = criterion(output,batch_answer)
        acc = accuracy(output,batch_answer)
        # print('loss backward')
        # startt = time.clock()
        loss.backward()
        optimizer.step()
        # end = time.clock()
        # print(str(end-startt))
        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss/ len(train_question) , epoch_acc/ len(train_question)

def evaluate(model,val_context,val_question,val_answer,val_choice,criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch_context, batch_question, batch_choice, batch_answer in tqdm(zip(val_context, val_question, val_choice, val_answer)):
            output = model(batch_context,batch_question,batch_choice)
            output = torch.cat(output, 0).view(-1, len(batch_context))
            output = output.permute(1, 0)
            loss = criterion(output, batch_answer)
            acc = accuracy(output, batch_answer)
            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(val_context), epoch_acc / len(val_context)

def main(args):

    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.num_epochs
    max_len = args.max_len
    word_hidden_size_ = args.word_hidden_size
    sent_hidden_size_ = args.sent_hidden_size
    embed_size = 1024

    print("load data")
    startt = time.clock()

    recipe_context, recipe_images, recipe_question, recipe_choice, recipe_answer = load_cleaned_data(
        'train_cleaned.json')
    recipe_context_valid, recipe_images_valid, recipe_question_valid, recipe_choice_valid, recipe_answer_valid = load_cleaned_data(
        'val_cleaned.json')
    recipe_context_test , recipe_images_test, recipe_question_test , recipe_choice_test , recipe_answer_test = load_cleaned_data(
        'test_cleaned.json')

    recipe_context = recipe_context[:7824]
    recipe_images = recipe_images[:7824]
    recipe_question = recipe_question[:7824]
    recipe_choice = recipe_choice[:7824]
    recipe_answer = recipe_answer[:7824]

    recipe_context_valid = recipe_context_valid[:960]
    recipe_images_valid = recipe_images_valid[:960]
    recipe_question_valid = recipe_question_valid[:960]
    recipe_choice_valid= recipe_choice_valid[:960]
    recipe_answer_valid = recipe_answer_valid[:960]

    end  = time.clock()
    print(str(end-startt))

    model = QAnet(max_len,model_dim = 5122)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
        model.eval()
        print("LOAD Model:", args.load_model)

    optimizer = optim.Adam(model.parameters(),lr = lr, weight_decay=0.0001)

    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = criterion.to(device)

    max_val_acc = 0.0


    for epoch in tqdm(range(num_epochs)):

        print(epoch)
        print("shuffle data")
        # startt = time.clock()


        train_context_new, train_question_new, train_choice_new, train_answer_new = shuffle_data(recipe_context,
                                                                                                 recipe_question,
                                                                                                 recipe_choice,
                                                                                                 recipe_answer)
        train_context, train_question, train_choice, train_answer = split_batch(batch_size, train_context_new,
                                                                                train_question_new, train_choice_new,
                                                                              train_answer_new)

        val_context_new, val_question_new, val_choice_new, val_answer_new = shuffle_data(recipe_context_valid,
                                                                                         recipe_question_valid,
                                                                                         recipe_choice_valid,
                                                                                         recipe_answer_valid)

        val_context, val_question, val_choice, val_answer = split_batch(batch_size, val_context_new, val_question_new,
                                                                        val_choice_new, val_answer_new)

        test_context_new, test_question_new, test_choice_new, test_answer_new = shuffle_data(recipe_context_test,
                                                                                         recipe_question_test,
                                                                                         recipe_choice_test,
                                                                                         recipe_answer_test)

        test_context, test_question, test_choice, test_answer = split_batch(batch_size,test_context_new, test_question_new,
                                                                        test_choice_new, test_answer_new)

        # end = time.clock()
        # print(str(end - startt))

        train_loss, train_acc = train(model,train_context,train_question,train_answer,train_choice,optimizer,criterion)
        val_loss ,val_acc = evaluate(model,val_context,val_question,val_answer, val_choice, criterion)
        test_loss, test_acc = evaluate(model, test_context, test_question, test_answer, test_choice, criterion)
        # log_data(args.log_path,train_loss,train_acc,val_loss,val_acc)

        print(f'| Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Val. Loss: {val_loss:.3f} | Test. Acc: {test_acc * 100:.2f}%| Test. Loss:{test_loss:.3f} | test. Acc: {test_loss * 100:.2f}%')
        # if val_acc > max_val_acc:
            # max_val_acc = val_acc
            # save_model(model,epoch,val_acc,args.saved_path)

        
if __name__ == '__main__':
    args = get_args()
    main(args)






# recipe_context, recipe_images, recipe_question, recipe_choice, recipe_answer = load_cleaned_data('train_cleaned.json')
#
# train_context, train_question, train_choice, train_answer = split_batch(3, recipe_context,
#                                                                                 recipe_question, recipe_choice,
#                                                                                 recipe_answer)
#
# input = train_context[0]
# query = train_question[0]
# choice = train_choice[0]
# answer = train_answer[0]
#
# encoder = QAnet(3,20,model_dim = 128)
# result = encoder(input,query,choice)
# print(result)
# print(1)
# result = torch.cat(result,0).view(-1,3)
# result = result.transpose(0,1)
# print(result)
# cri =nn.CrossEntropyLoss()
# loss = cri(result,answer)
# print(loss)
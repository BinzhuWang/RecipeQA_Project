import torch
import torch.nn as nn
from torch.autograd import Variable
from Elmo import elmo_embedding
from utils import transport_1_0_2
import numpy as np
import torch.nn.functional as F
import time
from allennlp.modules.elmo import Elmo, batch_to_ids


import math

class WordRNN(nn.Module):

    def __init__(self,word_hidden_size=256,embed_size =1024, bidirectional = True):
        super(WordRNN, self).__init__()
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0.2, requires_grad=False)

        if bidirectional == True:
            self.word_lstm = nn.LSTM(embed_size,word_hidden_size,bidirectional = True, dropout=0.2)
        else:
            self.word_lstm = nn.LSTM(embed_size,word_hidden_size,bidirectional = False, dropout=0.2)

    def forward(self,sentences):
        if torch.cuda.is_available():
            character_ids = batch_to_ids(sentences).cuda()
        else:
            character_ids = batch_to_ids(sentences)

        embeddings = self.elmo(character_ids)['elmo_representations'][0]

        embedded = embeddings.permute(1,0,2)
        output_word, (state_word,_) = self.word_lstm(embedded)
        last_hidden = torch.cat((state_word[-2,:,:],state_word[-1,:,:]),dim = 1)
        return  output_word, last_hidden
        # output with size [(batch size,step, hidden_size)]
        # last_hidden represent the last output for two direction lstm and then concatnate them together

class SentRNN(nn.Module):
    #  to extract feature from question
    def __init__(self, sent_hidden_size = 256, word_hidden_size =256, bidirectional=True):

        super(SentRNN, self).__init__()
        self.sent_hidden_size = sent_hidden_size
        self.word_hidden_size = word_hidden_size
        self.bidirectional = bidirectional

        if self.bidirectional == True:
            self.sent_lstm = nn.LSTM(2 * word_hidden_size, sent_hidden_size,bidirectional=True)
            self.final_linear = nn.Linear(2 * sent_hidden_size, 256)
        else:
            self.sent_lstm = nn.LSTM(word_hidden_size, sent_hidden_size, bidirectional=True)
            self.final_linear = nn.Linear(sent_hidden_size, 256)

    def forward(self, word_vectors):
        word_vectors = word_vectors
        output_sent, (state_sent,_) = self.sent_lstm(word_vectors)
        final_map = torch.cat((state_sent[-2,:,:],state_sent[-1,:,:]),dim=1)
        return final_map

class PosEncoder(nn.Module):
    def __init__(self,length,d_model):
        super().__init__()
        # D_model is the length of the maximum length of sentence. I may set 10
        # length is the demension of hidden state
        freqs = torch.Tensor(
            [10000 ** (-i / d_model) if i % 2 == 0 else -10000 ** ((1 - i) / d_model) for i in range(d_model)]).unsqueeze(dim=1)
        phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(d_model)]).unsqueeze(dim=1)
        pos = torch.arange(length).repeat(d_model, 1).to(torch.float)
        self.pos_encoding = torch.sin(torch.add(torch.mul(pos, freqs), phases))

    def forward(self, x):
        # print(x.size())
        # print(self.pos_encoding.size())
        if torch.cuda.is_available():
            result = x.cuda() + self.pos_encoding.transpose(0,1).cuda()
        else:
            result = x + self.pos_encoding.transpose(0,1)

        return result


class Highway(nn.Module):
    def __init__(self, layer_num: int, size: int):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(self.n)])

    def forward(self, x):
        #         x = x.transpose(1,2)
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        #         x = x.transpose(1,2)
        return x


class DepthwiseSeparableConv(nn.Module):
    # in ch and out ch is the demension of the input length. the length of the step
    def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception("Wrong dimension for Depthwise Separable Convolution!")
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class ScaledDotproductAttention(nn.Module):

    def __init__(self,attention_dropout = 0.0):
        super(ScaledDotproductAttention,self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, q,k,v,scale = None, attn_mask = None):

        attention = torch.bmm(q,k.transpose(1,2))
        if scale:
            attention = attention*scale
        if torch.cuda.is_available():
            attention = attention.masked_fill_(attn_mask.cuda(), -1e6).cuda()
        else:
            attention = attention.masked_fill_(attn_mask, -1e6)
        attention = self.softmax(attention)
        #attention[attention!=attention] = 0
        #attention = attention.masked_fill_(attn_mask,0)

        attention = self.dropout(attention)
        context = torch.bmm(attention,v) # * mask


        return context,attention # batch size, step size, dim (batch size, step size) [1, 1, 1, 1, 0, ...]

class MultiheadAttention(nn.Module):
    def __init__(self,model_dim,num_heads,dropout = 0.0):
        super(MultiheadAttention,self).__init__()

        self.dim_per_head  = model_dim//num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim,self.dim_per_head*self.num_heads)
        self.linear_v = nn.Linear(model_dim,self.dim_per_head*self.num_heads)
        self.linear_q = nn.Linear(model_dim,self.dim_per_head*self.num_heads)

        self.dot_product_attention = ScaledDotproductAttention(dropout)
        self.linear_final = nn.Linear(model_dim,model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, atten_mask = None):
        residual = query

        batch_size = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        key = key.view(batch_size*self.num_heads,-1,self.dim_per_head)
        value = value.view(batch_size*self.num_heads,-1,self.dim_per_head)
        query = query.view(batch_size*self.num_heads,-1,self.dim_per_head)


        atten_mask = atten_mask.repeat(self.num_heads,1,1)

        scale = (key.size(-1)//self.num_heads)** (-0.5)

        context, attention = self.dot_product_attention(query,key,value,scale,atten_mask)

        context = context.view(batch_size,-1,self.dim_per_head*self.num_heads)

        output = self.linear_final(context)

        output = self.dropout(output)

        output = self.layer_norm(residual+output)

        return output ,attention

class Embedding(nn.Module):
    def __init__(self,d_word):
        super(Embedding).__init__()
        self.high = Highway(2, d_word)

    def forward(self, wd_emb):
        emb = self.high(wd_emb,512)
        return emb

class CqAttention(nn.Module):

    def __init__(self):
        super(CqAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, context,query,mask):
        query = query.unsqueeze(1).transpose(1,2)
        c_q_attention = torch.bmm(context,query)
        if torch.cuda.is_available():
            c_q_attention = c_q_attention.masked_fill_(mask.cuda(),-1e6).cuda()
        else:
            c_q_attention = c_q_attention.masked_fill_(mask, -1e6)

        c_q_attention = self.softmax(c_q_attention)
        # c_q_attention[c_q_attention!=c_q_attention] = 0
        result = torch.bmm(c_q_attention.transpose(1,2),context)

        return result


class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, ch_num: int, k: int, length: int,d_model):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = MultiheadAttention(d_model,8)
        self.fc = nn.Linear(ch_num, ch_num, bias=True)
        self.pos = PosEncoder(length,d_model)
        # self.norm = nn.LayerNorm([d_model, length])
        self.normb = nn.LayerNorm([length,d_model])
        self.norms = nn.ModuleList([nn.LayerNorm([length,d_model]) for _ in range(conv_num)])
        self.norme = nn.LayerNorm([length,d_model])
        self.L = conv_num
        self.high = Highway(2,512)

    def forward(self, x, mask,dropout = 0.1):
        x = self.high(x)
        out = self.pos(x)
        res = out
        out = self.normb(out)
        for i, conv in enumerate(self.convs):
            out = conv(out)
            out = F.relu(out)[:,:,:512]

            out = out + res
            if (i + 1) % 2 == 0:
                p_drop = dropout * (i + 1) / self.L
                out = F.dropout(out, p=p_drop, training=self.training)
            res = out
            out = self.norms[i](out)
        # print("Before attention: {}".format(out.size()))
        out ,attention = self.self_att(out,out,out,mask)
        # print("After attention: {}".format(out.size()))
        out = out + res
        out = F.dropout(out, p=dropout, training=self.training)
        res = out
        out = self.norme(out)
        out = self.fc(out.transpose(1, 2)).transpose(1, 2)
        out = F.relu(out)
        out = out + res
        out = F.dropout(out, p=dropout, training=self.training)
        return out


class Pointer(nn.Module):
    def __init__(self,input_len,output_len,model_dim):
        super(Pointer,self).__init__()
        self.fc_1 = nn.Linear(input_len,output_len,bias=True)
        self.fc_2 = nn.Linear(output_len,1,bias=True)
        self.model_dim = model_dim
        self.encoderblock = EncoderBlock(1,20,4,20,self.model_dim)



    def forward(self, result):
        result = self.fc_1(result.transpose(1,2)).transpose(1,2)
        # mask = torch.zeros(result.size(0),20,20).byte()
        # result = self.encoderblock(result,mask)
        result = self.fc_2(result.transpose(1,2))
        result = result.squeeze(2) #( batchsize,model_dim)
        return result


class MakeChoice(nn.Module):

    def __init__(self,word_hidden_size=256,embed_size =1024, bidirectional = True):

        super(MakeChoice, self).__init__()
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0.2, requires_grad=False)

        if bidirectional == True:
            self.word_lstm = nn.LSTM(embed_size, word_hidden_size, bidirectional=True, dropout=0.2)
        else:
            self.word_lstm = nn.LSTM(embed_size, word_hidden_size, bidirectional=False, dropout=0.2)
    def forward(self,sentences):
        if torch.cuda.is_available():
            character_ids = batch_to_ids(sentences).cuda()
        else:
            character_ids = batch_to_ids(sentences)

        embeddings = self.elmo(character_ids)['elmo_representations'][0]

        embedded = embeddings.permute(1,0,2)
        output_word, (state_word,_) = self.word_lstm(embedded)
        last_hidden = torch.cat((state_word[-2,:,:],state_word[-1,:,:]),dim = 1)
        return  output_word, last_hidden

class QAnet(nn.Module):

    def __init__(self,max_len,model_dim = 512):
        super(QAnet,self).__init__()
        self.wordrnn = WordRNN()
        self.max_len = max_len
        self.model_dim = model_dim
        self.encoder_block = EncoderBlock(1,20,4,20,self.model_dim)
        self.srnn = SentRNN()
        self.q_c_attention = CqAttention()
        self.pointer = Pointer(25,20,self.model_dim)
        self.choicenet = MakeChoice(256, 1024, bidirectional=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(512 *4, 512)
        self.fc4 = nn.Linear(512, 1)
        self.fc1 = nn.Linear(160,16)
        # self.cosine_similarity = nn.CosineSimilarity()

    def padding_mask(self,batch_size, batch_len, max_len=20):

        mask = np.ones((int(batch_size), int(max_len), max_len))
        for i in range(mask.shape[0]):
            for j in range(batch_len[i]):
                mask[i][:batch_len[i]][j][:batch_len[i]] = 0
        return torch.from_numpy(mask).byte()

    def cq_mask(self,batch_size,batch_len,max_len):
        mask = np.ones((batch_size,max_len,1))
        for i in range(mask.shape[0]):
            for j in range(batch_len[i]):
                mask[i][j] = 0
        return torch.from_numpy(mask).byte()

    def Infersent(self, x1, x2):
        return torch.cat((x1, x2, torch.abs(x1 - x2), x1 * x2), 1)


    def forward(self, train_context,train_query,choice_list):

        # context part
        train_query = transport_1_0_2(train_query)
        choice_list = transport_1_0_2(choice_list)
        train_context = transport_1_0_2(train_context)

        '''
        outlist = []
        for sample in train_context:
            if torch.cuda.is_available():
                outlist.append(self.wordrnn(sample)[1].cpu().detach().numpy())# extract each step as a vector, outlist: list [batch,each steps,model_dim
            else:
                outlist.append(self.wordrnn(sample)[1].detach().numpy())

        if torch.cuda.is_available():
            output_word = torch.FloatTensor(outlist).cuda()
        else:
            output_word = torch.FloatTensor(outlist) # (step, batch_size,dim)


        output_word = output_word.transpose(0,1) # (batch_size, step, dim)



        batch_len = [] # count len in order to form the mask

        if torch.cuda.is_available():
            start = torch.zeros(output_word.size(0),self.max_len,self.model_dim).cuda() # start is the tensor as the first input to highway
        else:
            start = torch.zeros(output_word.size(0), self.max_len, self.model_dim)

        for i in range(output_word.size(0)):
            # padding part. make the maximum length as 20,use 0 *128 as the padding element
            start[i,:output_word[i].size(0),:] = start[i,:output_word[i].size(0),:].clone() + output_word[i]
            batch_len.append(output_word[i].size(0))


        pad_mask = self.padding_mask(output_word.size(0), batch_len)  #padmask, [batch, max len, model_dim], byte type
        # the padding part be 1 other part be 0,

        context_out = self.encoder_block(start,pad_mask)
        '''

        # query part

        outlist_query = []
        for sample in train_query:
            outlist_query.append(self.wordrnn(sample)[1])
        start = torch.stack(outlist_query, dim=0) # (4, batchsize, model_dim)

        query = self.srnn(start) # (batch_size,model_dim)

        '''
        qcmask = self.cq_mask(output_word.size(0),batch_len,self.max_len)

        qc_result = self.q_c_attention(context_out,query,qcmask)
        result = torch.cat((context_out,start.transpose(0,1),qc_result),dim = 1) # (8,25,128)
        # print("kkb")
        # print(context_out.size())
        # print(start.size())
        # print(qc_result.size())


        # 　pointer part

        result = self.pointer(result)
        result = torch.cat((result,query),dim = 0)
        '''
        result = query


        # print(result.size())
        # maske a choice
        output_choice_list = []
        for i in range(len(choice_list)):
            output_choice, hidden_state = self.choicenet(choice_list[i])
            # print(hidden_state.size())
            # print(result[i].size())
            similarity_scores = self.Infersent(result,hidden_state)
            similarity_scores = self.dropout(torch.tanh(self.fc3(similarity_scores)))
            similarity_scores = self.fc4(similarity_scores)


            output_choice_list.append(similarity_scores)

        return output_choice_list

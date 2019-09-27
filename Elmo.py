from allennlp.modules.elmo import Elmo,batch_to_ids
import torch
#reference : https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py
# https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
def elmo_embedding(sentence): # input sentence should be list and split with 2d . the first demension is batichsize
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    # we can  change the parameters by through change the options_file and weight_file
    if torch.cuda.is_available():
        elmo = Elmo(options_file, weight_file, 1, dropout=0.2, requires_grad=False).cuda()
        character_ids = batch_to_ids(sentence).cuda()
    else:
        elmo = Elmo(options_file, weight_file, 1, dropout=0.2, requires_grad=False)
        character_ids = batch_to_ids(sentence)
    embedding= elmo(character_ids)['elmo_representations'][0] # output size [batch_size,max_length_in_sentence,1024] 1024 is default vector length
    embedding_mask =  elmo(character_ids)['mask'][0]

    return embedding,embedding_mask

'''
input example:
x = ["I am a student ".split(),"please do me a favor".split()]
we can get the embedding through elmo_embedding(x)[0]
output size is [batch_size,sentence length,hidden_size]

'''
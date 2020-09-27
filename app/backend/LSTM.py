import torch
import torch.nn as nn
import json
import numpy as np

# read json for dictionary mapping
# open from json file
with open('char2idx_m2.json', 'r', encoding = 'UTF-8') as json_file:
    char2idx = json.load(json_file)

# open from json file
with open('img2idx.json', 'r', encoding = 'UTF-8') as json_file:
    img2idx = json.load(json_file)

idx2char = {value:key for key, value in char2idx.items()}
idx2img = {value:key for key, value in img2idx.items()}



class MemeGeneratorLSTM(nn.Module):
    def __init__(self):
        super(MemeGeneratorLSTM, self).__init__()
        self.embedding_dim = 128
        self.img_embedding = 32
        self.seqlen = 1
        self.num_classes = len(char2idx)

        self.lstm_hidden_size = 1024
        self.lstm_layer_size = 1
        self.lstm_num_directions = 1
        
        # Embedding Layer for Images
        self.embedding_img = nn.Embedding(len(img2idx), self.img_embedding)
        # Embedding Layer for character embeddings
        self.embedding_layer = nn.Embedding(len(char2idx), self.embedding_dim, padding_idx = char2idx['<pad>'])
        
        # project to embedding dim
        self.project_down = nn.Linear(self.img_embedding + self.embedding_dim, self.embedding_dim)

        # LSTM layer
        self.lstm_layer = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.lstm_hidden_size, num_layers=self.lstm_layer_size, bidirectional=False, batch_first = True)
        # fc layer
        self.fc = nn.Linear(self.lstm_hidden_size, self.num_classes) 


    def forward(self, input_img, x, prev_state_h, prev_state_c):
        # input_img (batch_size)
        # x is the decoder input (batch_size, 1) where 1 is seqlen
        # prev_state_h (num_layers_dec * num_directions_dec, batch_size, hidden_size_dec)
        # prev_state_c (num_layers_dec * num_directions_dec, batch_size, hidden_size_dec)
        batch_size = x.size()[0]

        # input_img (batch_size, 1)
        input_img = torch.unsqueeze(input_img, dim=1)
        
        # repeat for replication (batch_size, 1)
        input_img = input_img.repeat(1,1)

        # image embeddings (batch_size, 1, img_embedding_dim)
        img_out = self.embedding_img(input_img)

        # embedding shape (batch_size, 1, embedding_dim) where 1 is seqlen
        text_out = self.embedding_layer(x)

        # concatenate between image and caption embeddings
        # (batch_size, 1, text_emb + img_emb) where 1 is seqlen
        cat = torch.cat((img_out, text_out), dim=2)

        # project down to (batch_size, 1, 128) where 1 is seqlen
        embedding_out = self.project_down(cat)

        # apply LSTM layer
        # HERE IN THE DECODER WE PASS IN SEQ_LEN = 1 to force feed decoder
        # input = batchsize x seq_len x input_size -> Here input_size = 128
        # lstm_out = (batch, seq_len, num_directions * hidden_size)
        # hn = hidden at t=seq_len  (numdirection x num_layers, batchsize, hidden_size)
        # cn = cell at t=seq_len (numdirection x num_layers, batchsize, hidden_size)
        lstm_out, (hn, cn) = self.lstm_layer(embedding_out, (prev_state_h, prev_state_c))

        # output shape before squeeze == (batch_size, 1, hidden_size)
        # output shape after squeeze == (batch_size, hidden_size)
        output = torch.squeeze(lstm_out, dim = 1)
        
        # output shape == (batch_size, vocab)
        out = self.fc(output)
        return out, hn, cn #, attention_weights


    def init_state(self, batch_size):
        # first one is layer size * num_directions
        return (torch.zeros(1, batch_size, self.lstm_hidden_size),
                torch.zeros(1, batch_size, self.lstm_hidden_size))


class MemeGeneratorM2(nn.Module):
    def __init__(self):
        super(MemeGeneratorM2, self).__init__()
        self.memegeneratorlstm = MemeGeneratorLSTM()

    def forward(self, input_img, x, label, device, prediction_mode = False):
        # input_img (batch_size)
        # x is the lstm input (batch_size, 199) where 199 is seqlen
        # label is the label (batch_size, 199) where 199 is seqlen
        # device = "cpu"/"cuda:0"
        # prediction_mode = True/False

        batch_size = input_img.size()[0]

        # LSTM hidden state initialization
        prev_state_h, prev_state_c = self.memegeneratorlstm.init_state(batch_size)
        prev_state_h = prev_state_h.to(device)
        prev_state_c = prev_state_c.to(device)

        # store predictions and outputs
        predictions_arr = []
        output_tensor = torch.zeros((batch_size, 1, len(char2idx)))
        output_tensor = output_tensor.to(device)

        lstm_input = torch.unsqueeze(x[:,0], dim = 1)

        # for prediction only
        first_nonzero = (x == 0).nonzero(as_tuple=False)[0][1].item()

        # Teacher forcing - feeding the target as the next input
        for t in range(0, x.size()[1]): # iterate until len of sequence
             # prediction size (batchsize, num_vocab)
             predictions, prev_state_h, prev_state_c = self.memegeneratorlstm(input_img, lstm_input, prev_state_h, prev_state_c)

             # store lstm_output_tensor
             output_tensor = torch.cat([output_tensor, torch.unsqueeze(predictions, dim=1)], dim = 1)            
             # get one prediction
             one_prediction = torch.max(predictions, dim = 1).indices
             # save the prediction in list
             predictions_arr.append(one_prediction.detach().cpu().numpy())

             if prediction_mode == False:
                 # using teacher forcing
                 lstm_input = torch.unsqueeze(label[:, t], dim = 1)
             else:
                 # only for batchsize = 1
                 # use teacher forcing for initial starter string
                 if t < first_nonzero - 1:
                    lstm_input = torch.zeros((1, 1)).long()
                    lstm_input[0] = x[0][t+1]
                 else:
                    # use prediction as previous output previous input
                    one_prediction = torch.unsqueeze(one_prediction, dim = 1)
                    lstm_input = one_prediction
        
        # remove the original zeros for output_tensor
        # output_tensor shape (batch_size, seqlen, numclasses)
        # predictions_arr shape (seqlen, batch_size)
        output_tensor = output_tensor[:,1:,:]
        predictions_arr = np.array(predictions_arr)
        predictions_arr = np.transpose(predictions_arr)

        return output_tensor, predictions_arr
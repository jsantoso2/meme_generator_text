import torch
import torch.nn as nn
import json

# read json for dictionary mapping
# open from json file
with open('char2idx.json', 'r', encoding = 'UTF-8') as json_file:
    char2idx = json.load(json_file)

# open from json file
with open('img2idx.json', 'r', encoding = 'UTF-8') as json_file:
    img2idx = json.load(json_file)

idx2char = {value:key for key, value in char2idx.items()}
idx2img = {value:key for key, value in img2idx.items()}


class MemeGeneratorCNN(nn.Module):
    def __init__(self):
        super(MemeGeneratorCNN, self).__init__()
        self.embedding_dim = 16
        self.img_embedding = 8
        self.num_classes = len(char2idx)
        
        # Embedding Layer for Images
        self.embedding_img = nn.Embedding(len(img2idx), self.img_embedding)
        # Embedding Layer for character embeddings
        self.embedding_layer = nn.Embedding(len(char2idx), self.embedding_dim, padding_idx = char2idx['<pad>'])
        
        # project to embedding dim
        self.project_down = nn.Linear(self.img_embedding + self.embedding_dim, self.embedding_dim)

        # convolution block
        self.conv1 = nn.Conv1d(in_channels = 16, out_channels = 1024, kernel_size = 5, padding=2)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=0.25)

        # convolution block
        self.conv2 = nn.Conv1d(in_channels = 1024, out_channels= 1024, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)  # default value of stride = kernel_size
        self.dropout2 = nn.Dropout(p=0.25)

        # convolution block
        self.conv3 = nn.Conv1d(in_channels = 1024, out_channels = 1024, kernel_size = 5, padding=2)
        self.relu3 = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm1d(1024)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(p=0.25)
        
        # convolution block
        self.conv4 = nn.Conv1d(in_channels = 1024, out_channels = 1024, kernel_size = 5, padding=2)
        self.relu4 = nn.ReLU()
        self.batchnorm4 = nn.BatchNorm1d(1024)
        self.maxpool4 = nn.MaxPool1d(kernel_size=2)
        self.dropout4 = nn.Dropout(p=0.25)

        # convolution block
        self.conv5 = nn.Conv1d(in_channels = 1024, out_channels = 1024, kernel_size = 5, padding=2)
        self.relu5 = nn.ReLU()
        self.batchnorm5 = nn.BatchNorm1d(1024)

        # final layers
        self.dropout5 = nn.Dropout(p=0.25)
        self.Linear1 = nn.Linear(1024, 1024)
        self.LinearRelu1 = nn.ReLU()
        self.batchnorm6 = nn.BatchNorm1d(1024)
        self.dropout6 = nn.Dropout(p=0.25)
        self.fc = nn.Linear(1024, self.num_classes)


    def forward(self, input_img, x):
        # input_img (batch_size)
        # x is the decoder input (batch_size, 128) where 128 is seqlen

        # input_img (batch_size, 1)
        input_img = torch.unsqueeze(input_img, dim=1)

        # need to repeat for concat (batch_size, 128) 
        input_img = input_img.repeat(1,128)
        # image embeddings (batch_size, 128, img_embedding_dim)
        img_out = self.embedding_img(input_img)

        # embedding shape (batch_size, 128, embedding_dim)
        text_out = self.embedding_layer(x)
        
        # concatenate between image and caption embeddings
        # (batch_size, 128, text_emb + img_emb)
        cat = torch.cat((img_out, text_out), dim=2)

        # project down to (batch_size, 128, 16)
        embedding_out = self.project_down(cat)

        # need to permute in order to match convnets
        # embedding shape (batch_size, embedding_dim, 128) -> (batch_size, 16, 128)
        embedding_out = embedding_out.permute(0, 2, 1)

        # apply convolution (batch_size, out_channels, 128) -> (batch_size, 1024, 128)
        conv1_out = self.conv1(embedding_out)
        conv1_out = self.relu1(conv1_out)
        # apply batchnorm -> (batch_size, 1024, 128)
        batchnorm1_out = self.batchnorm1(conv1_out)
        # apply maxpooling1 -> (batch_size, 1024, 64) kernel is 2 here
        maxpool1_out = self.maxpool1(batchnorm1_out)
        # apply dropout 1 -> (batch_size, 1024, 64)
        dropout1_out = self.dropout1(maxpool1_out)

        # apply convolution (batch_size, 1024, out_channels) -> (batch_size, 1024, 64)
        conv2_out = self.conv2(dropout1_out)
        conv2_out = self.relu2(conv2_out)
        # apply batchnorm -> (batch_size, 1024, 64)
        batchnorm2_out = self.batchnorm2(conv2_out)
        # apply maxpooling2 -> (batch_size, 1024, 32) kernel is 2 here
        maxpool2_out = self.maxpool2(batchnorm2_out)
        # apply dropout 1 -> (batch_size, 1024, 32)
        dropout2_out = self.dropout2(maxpool2_out)

        # apply convolution (batch_size, 1024, out_channels) -> (batch_size, 1024, 32)
        conv3_out = self.conv3(dropout2_out)
        conv3_out = self.relu3(conv3_out)
        # apply batchnorm -> (batch_size, 1024, 32)
        batchnorm3_out = self.batchnorm3(conv3_out)
        # apply maxpooling3 -> (batch_size, 1024, 16) kernel is 2 here
        maxpool3_out = self.maxpool3(batchnorm3_out)
        # apply dropout 1 -> (batch_size, 1024, 16)
        dropout3_out = self.dropout3(maxpool3_out)

        # apply convolution (batch_size, 1024, out_channels) -> (batch_size, 1024, 32)
        conv4_out = self.conv4(dropout3_out)
        conv4_out = self.relu4(conv4_out)
        # apply batchnorm -> (batch_size, 1024, 16)
        batchnorm4_out = self.batchnorm4(conv4_out)
        # apply maxpooling4 -> (batch_size, 1024, 8) kernel is 2 here
        maxpool4_out = self.maxpool4(batchnorm4_out)
        # apply dropout 1 -> (batch_size, 1024, 8)
        dropout4_out = self.dropout4(maxpool4_out)

        # apply convolution (batch_size, 1024, out_channels) -> (batch_size, 1024, 8)
        conv5_out = self.conv5(dropout4_out)
        conv5_out = self.relu5(conv5_out)
        # apply batchnorm -> (batch_size, 1024, 8)
        batchnorm5_out = self.batchnorm5(conv5_out)

        # Global MaxPooling1d shape (batch_size, 1024)
        # this takes maximum among all channels
        gmaxpool_out = torch.max(batchnorm5_out, dim=2)
        gmaxpool_out = gmaxpool_out.values
        gmaxpool_out = self.dropout5(gmaxpool_out)

        # apply dense layer (batch_size, 1024)
        linear_out = self.Linear1(gmaxpool_out)
        linear_out = self.LinearRelu1(linear_out)
        batchnorm6_out = self.batchnorm6(linear_out)
        dropout6_out = self.dropout6(batchnorm6_out)

        final = self.fc(dropout6_out)

        return final
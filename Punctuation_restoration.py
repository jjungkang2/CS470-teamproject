import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm.notebook as tq
# from pathlib import Path
import os
import pickle

from easydict import EasyDict as edict

vocab_size = 11176
vector_size = 100

trg_ntoken = 6
pad_id = 0
src_ntoken = vocab_size

with open(os.getcwd() + '/Resources/word_dict_small', 'rb') as file:
    # word_dict = file.read()
    word_dict = pickle.load(file)

with open(os.getcwd() + '/Resources/rev_dict', 'rb') as file:
    # rev_word_dict = file.read()
    rev_word_dict = pickle.load(file)

with open(os.getcwd() + '/Resources/GloVe_matrix', 'rb') as file:
    # emb_matrix = file.read()
    emb_matrix = pickle.load(file)

rev_punc_dict = {0: ' ', 1: ' ', 2: '. ', 3: ', ', 4: '? ', 5: ' '}

# Hyperparameters
args = edict()
args.batch_size = 32
args.lr = 0.001
args.epochs = 20
args.clip = 1
args.ninp = vector_size

args.dropout = 0.2

args.nlayers = 2
args.nhid = 512
args.nhead = 8
args.attn_pdrop = 0.1  # 0.1
args.resid_pdrop = 0.1  # 0.1
args.embd_pdrop = 0.1  # 0.1
args.nff = 4 * args.nhid

args.gpu = True

# Basic settings
torch.manual_seed(470)
torch.cuda.manual_seed(470)
device = 'cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu'

class PuncApplyDataset(Dataset):
    """Punctuation restoration dataset for real use"""

    def __init__(self, data_file_path):
        self.raw_input = []
        new_lst = []
        for i in range(len(input_list)):
            new_lst.append(input_list[i])
            if i % 100 == 99:
                self.raw_input.append(new_lst)
                new_lst = []

        if len(input_list) % 100 != 0:
            addition = 100 - len(input_list) % 100
            for j in range(addition):
                new_lst.append('<pad>')
            self.raw_input.append(new_lst)

        real_input = []
        for hund_words in self.raw_input:
            hund_nums = []
            for word in hund_words:
                hund_nums.append(word_dict.get(word, word_dict['<unk>']))
            real_input.append(hund_nums)
        self.input = torch.tensor(real_input, dtype=torch.long)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        sample = {'input': self.input[idx], 'raw_input': self.raw_input[idx]}
        return sample

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_input = nn.Linear(input_size, 4 * hidden_size)
        self.linear_hidden = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, state):
        hx, cx = state
        hx_out = self.linear_hidden(hx)
        x_out = self.linear_input(x)
        before_chunk = x_out + hx_out

        chunk_forgetgate, chunk_ingate, chunk_cellgate, chunk_outgate = torch.chunk(before_chunk,chunks=4, dim=1)
        fx = torch.sigmoid(chunk_forgetgate)
        ix = torch.sigmoid(chunk_ingate)
        cty = torch.tanh(chunk_cellgate)
        ox = torch.sigmoid(chunk_outgate)

        out_cy = (cx * fx) + (ix * cty)

        out_hy = torch.tanh(out_cy) * ox

        return out_hy, (out_hy, out_cy)


class LSTMLayer(nn.Module):
    def __init__(self, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = LSTMCell(*cell_args)

    def forward(self, x, state, length_x=None):
        inputs = x.unbind(0)
        assert (length_x is None) or torch.all(length_x == length_x.sort(descending=True)[0])
        outputs = []
        out_hidden_state = []
        out_cell_state = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
            if length_x is not None:
                if torch.any(i + 1 == length_x):
                    out_hidden_state = [state[0][i + 1 == length_x]] + out_hidden_state
                    out_cell_state = [state[1][i + 1 == length_x]] + out_cell_state
        if length_x is not None:
            state = (torch.cat(out_hidden_state, dim=0), torch.cat(out_cell_state, dim=0))
        return torch.stack(outputs), state


class LSTM(nn.Module):
    def __init__(self, ninp, nhid, num_layers, dropout):
        super(LSTM, self).__init__()
        self.layers = []
        self.dropout = nn.Dropout(dropout)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(LSTMLayer(ninp, nhid))
            else:
                self.layers.append(LSTMLayer(nhid, nhid))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, states, length_x=None):
        output_states = []

        input = x
        for i in range(len(states)):
            output_tensor, output_state = self.layers[i](input, states[i], length_x)
            output_states.append(output_state)

            input = self.dropout(output_tensor)
        return output_tensor, output_states


class LSTMModule(nn.Module):
    def __init__(self):
        super(LSTMModule, self).__init__()
        ninp = args.ninp
        nhid = args.nhid
        nlayers = args.nlayers
        dropout = args.dropout

        self.dropout = nn.Dropout(dropout)
        self.lstm = LSTM(ninp, nhid, nlayers, dropout)

    def forward(self, x, states, length_x=None):
        input = self.dropout(x)

        return self.lstm(input, states, length_x)


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.LSTM_LtoR = LSTMModule()
        self.LSTM_RtoL = LSTMModule()
        self.ff1 = nn.Linear(args.nhid, args.nhid)

        self.ff2 = nn.Linear(args.nhid, trg_ntoken)

        self.seq = nn.Sequential(nn.Linear(args.nhid * 2, args.nhid),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(args.nhid, trg_ntoken),
                                 nn.Softmax(dim=-1))
        self.softmax = nn.Softmax(dim=-1)
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vector_size)
        self.embed.weight = nn.Parameter(torch.tensor(emb_matrix, dtype=torch.float32))

    def forward(self, x, y, length, max_len=None, teacher_forcing=True):
        B = int(x.shape[1])
        x = self.embed(x)
        zero_tensor = torch.zeros(B, args.nhid).to(device)
        init_states = [(zero_tensor, zero_tensor)] * args.nlayers

        reverse_x = torch.flip(x, [1])

        output1, _ = self.LSTM_LtoR(x, init_states, length)
        output2, _ = self.LSTM_RtoL(reverse_x, init_states, length)

        output2 = torch.flip(output2, [1])

        output = torch.cat((output1, output2), dim=-1)

        output = output.transpose(0, 1)
        output = self.seq(output)

        return output

def sort_batch(x, y):
    lengths = (x!=pad_id).long().sum(0)
    length, idx = lengths.sort(dim = 0, descending= True)
    x = torch.index_select(x, 1, idx)
    y = torch.index_select(y, 1, idx)
    return x, y, length

def load_model(model):
    model.load_state_dict(torch.load(os.getcwd() + '/Resources/BiLSTM/BiLSTM_glove_best.ckpt', map_location = device))

def run_test(model, input):
    global input_list
    input_list = input
    with torch.no_grad():
        # model.eval()
        load_model(model)

        total_loss = 0
        n_correct = 0
        n_total = 0

        confusion_matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        total_counts = [0, 0, 0, 0]

        percent_matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        final_output_words = []
        final_output_puncs = []

        # test dataset
        path_to_dir = ''
        speech_dataset = PuncApplyDataset(path_to_dir)
        dataloader = DataLoader(speech_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

        for batch in dataloader:
            batch_words = []
            batch_puncs = []
            print_string = ''

            raw_input = batch['raw_input']
            raw_input = np.array(raw_input).T

            x = batch['input'].to(device)
            y = torch.rand(x.size())
            size_x, size_y = x.size()

            x = x.transpose(0, 1)
            y = y.transpose(0, 1)
            x, y, length = sort_batch(x, y)
            source = x.transpose(0, 1)
            pred = model(x, y, length)

            prediction = pred.argmax(-1)

            for i in range(size_x):
                words = []
                pred_punc = []
                for j in range(100):
                    try:
                        words.append(rev_word_dict[int(source[i][j])])
                        pred_punc.append(rev_punc_dict[int(prediction[i, j])])
                    except:
                        continue

                if len(words) == 0: continue
                batch_words.append(words)
                batch_puncs.append(pred_punc)

            for sent_num in range(len(raw_input)):
                sents = raw_input[sent_num]
                for word_num in range(len(batch_words)):
                    words = batch_words[word_num]
                    if (sents[0] == words[0] or words[0] == '<unk>') \
                            and (sents[1] == words[1] or words[1] == '<unk>') \
                            and (sents[2] == words[2] or words[2] == '<unk>') \
                            and (sents[3] == words[3] or words[3] == '<unk>') \
                            and (sents[4] == words[4] or words[4] == '<unk>') \
                            and (sents[-1] == words[-1] or words[-1] == '<unk>') \
                            and (sents[-2] == words[-2] or words[-2] == '<unk>'):
                        final_output_words += sents.tolist()
                        final_output_puncs += batch_puncs[word_num]

        i = 0
        print_sentence = ''
        while (final_output_words[i] != '<pad>'):
            print_sentence += final_output_words[i]
            print_sentence += final_output_puncs[i]
            i += 1
            if i == len(final_output_words): break
        return print_sentence

def execute_punctuator(input):
    model = BiLSTM().to(device)
    output = run_test(model, input)
    return output
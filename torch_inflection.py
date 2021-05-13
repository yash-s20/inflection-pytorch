import argparse
import codecs
import torch
import matplotlib
from flask import request, render_template, Flask, redirect, url_for

matplotlib.use('agg')
import matplotlib.pyplot as plt
import myutil
import numpy as np
from operator import itemgetter
import os, sys
from random import shuffle
import random
import torch.utils.data as data

parser = argparse.ArgumentParser()
parser.add_argument("--datapath", help="path to data", type=str, required=True)
parser.add_argument("--L1", help="transfer languages (split with comma for multiple ones)", type=str, required=False)
parser.add_argument("--L2", help="test languages", type=str, required=True)
parser.add_argument("--mode", help="usage mode", type=str,
                    choices=['train', 'test', 'test-dev', 'demo',
                             #'draw-dev', 'test-dev-ensemble', 'test-ensemble',
                             # 'test-two-ensemble', 'test-three-ensemble',
                             # 'test-all-ensemble'
                             ],
                    default='', required=True)
parser.add_argument("--setting", help="data setting", type=str, choices=['original', 'swap', 'low', ], default='original')
parser.add_argument("--modelpath", help="path to store the models", type=str, default='./models')
parser.add_argument("--figurepath", help="path to store the output attention figures", type=str, default='./figures')
parser.add_argument("--outputpath", help="path to store the inflected outputs on the test set", type=str,
                    default='./outputs')
parser.add_argument("--notest", help="do not use the test set at all", action="store_true")
parser.add_argument("--testset", help="path to different test set", type=str, required=False)
parser.add_argument("--outputfile", help="path to store the inflected outputs", type=str, required=False)
parser.add_argument("--use_hall", help="whether to use a hallucinated dataset (def: False)", action="store_true")
parser.add_argument("--only_hall", help="only use the hallucinated dataset to train (def: False)", action="store_true")
parser.add_argument("--predict_lang", help="use the language discriminator auxiliary task (def: False)", action="store_true")
parser.add_argument("--use_att_reg", help="use attention regularization on the lemma attention (def: False)",
                    action="store_true")
parser.add_argument("--use_tag_att_reg", help="use attention regularization on the tag attention (def: False)",
                    action="store_true")
parser.add_argument("--seed", help="random seed", default=42, type=int, required=False)
parser.add_argument("--optimizer", help="one of torch.optim's optimizers",
                    choices=list(filter(lambda x: x[0] == x[0].upper() and x[0] != "_",
                                        [optimizer for optimizer in dir(torch.optim)])), required=True)
parser.add_argument("--lr", type=float, default=0.01)

args = parser.parse_args()
random.seed(args.seed)

if args.L1:
    L1 = args.L1
    L1s = L1.split(',')
else:
    L1 = ''
    L1s = []

L2 = args.L2
DATA_PATH = args.datapath

if not os.path.isdir(DATA_PATH):
    print("Wrong data path, the provided one does not exist")
    sys.exit()
LOW_PATH = os.path.join(DATA_PATH, L2 + "-train")
DEV_PATH = os.path.join(DATA_PATH, L2 + "-dev")
HALL_PATH = os.path.join(DATA_PATH, L2 + "-hall")
TEST_PATH = os.path.join(DATA_PATH, L2 + "-test-covered")
if args.testset:
    TEST_PATH = args.testset

if not os.path.isdir(args.modelpath):
    os.mkdir(args.modelpath)
if not os.path.isdir(args.figurepath):
    os.mkdir(args.figurepath)
if not os.path.isdir(args.outputpath):
    os.mkdir(args.outputpath)

if L1:
    exp_dir = L1 + "-" + L2
else:
    exp_dir = L2

MODEL_DIR = os.path.join(args.modelpath, exp_dir)
if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)

FIGURE_DIR = os.path.join(args.figurepath, exp_dir)
if not os.path.isdir(FIGURE_DIR):
    os.mkdir(FIGURE_DIR)

OUTPUT_DIR = os.path.join(args.outputpath, exp_dir)
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

TRAIN = False
TEST = False
TEST_ENSEMBLE = False
TEST_TWO_ENSEMBLE = False
TEST_THREE_ENSEMBLE = False
TEST_ALL_ENSEMBLE = False
TEST_DEV = False
DRAW_DEV = False
TEST_DEV_ENSEMBLE = False
DEMO = False

if args.mode == "train":
    TRAIN = True
elif args.mode == "test":
    TEST = True
elif args.mode == "test-dev":
    TEST_DEV = True
elif args.mode == "demo":
    DEMO = True

USE_HALL = False
if args.use_hall:
    USE_HALL = True

ONLY_HALL = False
if args.only_hall:
    ONLY_HALL = True

if args.setting == "original":
    ORIGINAL = True
    SWAP = False
    LOW = False

MODEL_NAME = "orig."
if SWAP:
    MODEL_NAME = "swap."
elif LOW:
    MODEL_NAME = "low."

if USE_HALL:
    MODEL_NAME += "hall."
if ONLY_HALL:
    MODEL_NAME += "hallonly."

MAX_PREDICTION_LEN_DEF = 20
if L2 == "kabardian":
    MAX_PREDICTION_LEN_DEF = 25
elif L2 == "tatar":
    MAX_PREDICTION_LEN_DEF = 23
elif L2 == "greek":
    MAX_PREDICTION_LEN_DEF = 30
elif L2 == "latin":
    MAX_PREDICTION_LEN_DEF = 25
elif L2 == "livonian":
    MAX_PREDICTION_LEN_DEF = 22
elif L2 == "bengali":
    MAX_PREDICTION_LEN_DEF = 23
elif L2 == "czech":
    MAX_PREDICTION_LEN_DEF = 30
elif L2 == "lithuanian":
    MAX_PREDICTION_LEN_DEF = 33
elif L2 == "russian":
    MAX_PREDICTION_LEN_DEF = 50
elif L2 == "irish":
    MAX_PREDICTION_LEN_DEF = 37
elif L2 == "quechua":
    MAX_PREDICTION_LEN_DEF = 30
elif L2 == "azeri":
    MAX_PREDICTION_LEN_DEF = 22
elif L2 == "yiddish":
    MAX_PREDICTION_LEN_DEF = 22


LENGTH_NORM_WEIGHT = 0.1
EXTRA_WEIGHT = 0.3
USE_ATT_REG = False
USE_TAG_ATT_REG = False
PREDICT_LANG = False

if args.predict_lang:
    PREDICT_LANG = True
    MODEL_NAME += "lang."
if args.use_att_reg:
    USE_ATT_REG = True
if args.use_tag_att_reg:
    USE_TAG_ATT_REG = True

if USE_HALL:
    low_i, low_o, low_t = myutil.read_data(LOW_PATH)
    dev_i, dev_o, dev_t = myutil.read_data(DEV_PATH)
    if args.notest:
        test_i, test_t = dev_i, dev_t
    else:
        test_i, test_t = myutil.read_test_data(TEST_PATH)
    hall_i, hall_o, hall_t = myutil.read_data(HALL_PATH)
    low_i += hall_i
    low_o += hall_o
    low_t += hall_t
    lids_1 = [0] * len(low_i)
    high_i, high_o, high_t = [], [], []
    for j, L1 in enumerate(L1s):
        HIGH_PATH = os.path.join(DATA_PATH, L1 + "-train")
        ti, to, tt = myutil.read_data(HIGH_PATH)
        high_i += ti
        high_o += to
        high_t += tt
        lids_1 += [j + 1] * len(ti)
    NUM_LANG = len(L1s) + 1
elif ONLY_HALL:
    high_i, high_o, high_t = [], [], []
    low_i, low_o, low_t = myutil.read_data(LOW_PATH)
    dev_i, dev_o, dev_t = myutil.read_data(DEV_PATH)
    if args.notest:
        test_i, test_t = dev_i, dev_t
    else:
        test_i, test_t = myutil.read_test_data(TEST_PATH)
    hall_i, hall_o, hall_t = myutil.read_data(HALL_PATH)
    low_i += hall_i
    low_o += hall_o
    low_t += hall_t
else:
    low_i, low_o, low_t = myutil.read_data(LOW_PATH)
    dev_i, dev_o, dev_t = myutil.read_data(DEV_PATH)
    if args.notest:
        test_i, test_t = dev_i, dev_t
    else:
        test_i, test_t = myutil.read_test_data(TEST_PATH)
    _, test_o, _ = myutil.read_data(TEST_PATH[:-len("-covered")])
    high_i, high_o, high_t = [], [], []
    lids_1 = [0] * len(low_i)
    for j, L1 in enumerate(L1s):
        HIGH_PATH = os.path.join(DATA_PATH, L1 + "-train")
        ti, to, tt = myutil.read_data(HIGH_PATH)
        high_i += ti
        high_o += to
        high_t += tt
        lids_1 += [j + 1] * len(ti)
    NUM_LANG = len(L1s) + 1

if SWAP:
    if len(dev_i) < len(low_o):
        N = len(dev_i)
        tmp1, tmp2, tmp3 = list(low_i[-N:]), list(low_o[-N:]), list(low_t[-N:])
        low_i = list(low_i[:-N] + dev_i)
        low_o = list(low_o[:-N] + dev_o)
        low_t = list(low_t[:-N] + dev_t)
        dev_i, dev_o, dev_t = tmp1, tmp2, tmp3
    else:
        tmp1, tmp2, tmp3 = list(low_i), list(low_o), list(low_t)
        low_i, low_o, low_t = list(dev_i), list(dev_o), list(dev_t)
        dev_i, dev_o, dev_t = tmp1, tmp2, tmp3

print("Data lengths")
print("transfer-language", len(high_i), len(high_o), len(high_t))
print("test-language", len(low_i), len(low_o), len(low_t))
print("dev", len(dev_i), len(dev_o), len(dev_t))
print("test", len(test_i), len(test_t))


def compute_mixing_weights(l):
    if l == 3:
        K = float(len(high_i))
        N = float(len(low_i))
        M = float(len(dev_i))
        denom = 2 * N + M + 2 * K
        return [(K + N) / denom, (M + K) / denom, N / denom]
    elif l == 2:
        K = float(len(high_i))
        N = float(len(low_i))
        M = float(len(dev_i))
        denom = N + M + 2 * K
        return [(K + N) / denom, (M + K) / denom]


COPY_THRESHOLD = 0.9
COPY_TASK_PROB = 0.2
STARTING_LEARNING_RATE = args.lr
EPOCHS_TO_HALVE = 6

MULTIPLY = 1
if len(high_i) + len(low_i) < 5000:
    MULTIPLY = 1
    STARTING_LEARNING_RATE = 2 * args.lr
    COPY_THRESHOLD = 0.6
    COPY_TASK_PROB = 0.4
    EPOCHS_TO_HALVE = 12


def get_chars(l):
    flat_list = [char for word in l for char in word]
    return list(set(flat_list))


def get_tags(l):
    flat_list = [tag for sublist in l for tag in sublist]
    return list(set(flat_list))


EOS = "<EOS>"
NULL = "<NULL>"

if TRAIN:
    # SOS = "<SOS>"
    characters = get_chars(high_i + high_o + low_i + low_o + dev_i + dev_o + test_i)
    # characters.append(SOS)
    characters.append(EOS)
    if u' ' not in characters:
        characters.append(u' ')

    tags = get_tags(high_t + low_t + dev_t + test_t)
    tags.append(NULL)

    # Store vocabularies for future reference
    myutil.write_vocab(characters, os.path.join(MODEL_DIR, MODEL_NAME + "char.vocab"))
    myutil.write_vocab(tags, os.path.join(MODEL_DIR, MODEL_NAME + "tag.vocab"))
else:
    characters = myutil.read_vocab(os.path.join(MODEL_DIR, MODEL_NAME + "char.vocab"))
    if u' ' not in characters:
        characters.append(u' ')
    tags = myutil.read_vocab(os.path.join(MODEL_DIR, MODEL_NAME + "tag.vocab"))

# TODO: we could put the PAD character here
int2char = list(characters)
char2int = {c: i for i, c in enumerate(characters)}

int2tag = list(tags)
tag2int = {c: i for i, c in enumerate(tags)}

VOCAB_SIZE = len(characters)
TAG_VOCAB_SIZE = len(tags)

LSTM_NUM_OF_LAYERS = 1
EMBEDDINGS_SIZE = 32
STATE_SIZE = 100
ATTENTION_SIZE = 100
MINIBATCH_SIZE = 1
COPY_WEIGHT = 0.8
DROPOUT_PROB = 0.2

print("Characters:", characters)
print("Vocab size:", VOCAB_SIZE)
print("All tags:", tags)
print("Tag vocab size:", TAG_VOCAB_SIZE)


def run_lstm(init_state, input_vecs):
    s = init_state
    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


optimizer_class = getattr(torch.optim, args.optimizer)


class InflectionModule(torch.nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, tag_vocab_size=TAG_VOCAB_SIZE):
        super().__init__()
        self.enc_blstm = torch.nn.LSTM(input_size=EMBEDDINGS_SIZE, hidden_size=STATE_SIZE,
                                       num_layers=LSTM_NUM_OF_LAYERS, bidirectional=True)
        self.dec_lstm = torch.nn.LSTM(input_size=STATE_SIZE * 3 + EMBEDDINGS_SIZE, hidden_size=STATE_SIZE,
                                      num_layers=LSTM_NUM_OF_LAYERS, bidirectional=False)
        # TODO: embeddings have to add padding_idx if it's added
        self.input_lookup = torch.nn.Embedding(vocab_size, EMBEDDINGS_SIZE)
        self.tag_input_lookup = torch.nn.Embedding(tag_vocab_size, EMBEDDINGS_SIZE)
        self.attn_w1 = torch.nn.Linear(STATE_SIZE * 2, ATTENTION_SIZE, bias=False)
        self.attn_w2 = torch.nn.Linear(STATE_SIZE * 2 * LSTM_NUM_OF_LAYERS, ATTENTION_SIZE, bias=False)
        self.attn_w3 = torch.nn.Linear(5, ATTENTION_SIZE, bias=False)
        self.attn_v = torch.nn.Linear(ATTENTION_SIZE, 1, bias=False)

        self.decoder = torch.nn.Linear(STATE_SIZE, vocab_size)
        self.output_lookup = self.input_lookup

        self.enc_tag_lstm = torch.nn.LSTM(input_size=EMBEDDINGS_SIZE, hidden_size=STATE_SIZE,
                                          num_layers=LSTM_NUM_OF_LAYERS, bidirectional=False)
        self.tag_attn_w1 = torch.nn.Linear(STATE_SIZE, ATTENTION_SIZE, bias=False)
        self.tag_attn_w2 = torch.nn.Linear(STATE_SIZE * 2 * LSTM_NUM_OF_LAYERS, ATTENTION_SIZE, bias=False)
        self.tag_attn_v = torch.nn.Linear(ATTENTION_SIZE, 1, bias=False)

        if PREDICT_LANG:
            self.lang_class_w = torch.nn.Linear(NUM_LANG, 2 * STATE_SIZE, bias=False)

    def embed_tags(self, tags_list):
        """
        param:
        tags_list: is a list of list of tags. batched basically.
        return (B, L, D)
        """
        int_tags = torch.stack([torch.tensor([tag2int[t] for t in tags]) for tags in tags_list])
        # print(int_tags)
        return self.tag_input_lookup(int_tags)

    def embed_sentence(self, sentences):
        """
        return (B, L, D) sentences. currently (1, L, D)
        """
        # TODO: need to pad for bigger batch size than 1. currently no support for padding
        sentences = [[EOS] + list(sentence) + [EOS] for sentence in sentences]
        sentences = torch.stack([torch.tensor([char2int[c] for c in sentence]) for sentence in sentences])
        # print(sentences)
        return self.input_lookup(sentences)

    def self_encode_tags(self, tags):
        """
        tags: (B, L, D)
        """
        vectors, _ = self.enc_tag_lstm(torch.transpose(tags, 0, 1))  # needs (L, B, D)
        # (L, B, S * 1)
        vectors = vectors.transpose(0, 1)
        # (B, L, S)
        unnormalized = torch.matmul(vectors, vectors.transpose(1, 2))
        # (B, L, L)
        self_att_weights = torch.softmax(unnormalized, dim=-1)
        # (B, L, L)
        to_add = torch.matmul(vectors.transpose(1, 2), self_att_weights).transpose(1, 2)
        # (B, L, S)
        return vectors + to_add

    def encode_sentence(self, sentence):
        """
        sentence: (B, L, D) hopefully a packed padded sequence
        """
        vectors, _ = self.enc_blstm(sentence.transpose(0, 1))
        return vectors.transpose(0, 1)
        # (B, L, 2S)

    def attend_tags(self, state, w1dt):
        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim x attdim)
        # state: (B, D)
        # w2dt = self.tag_attention_w2 * state
        # print(state.shape)
        # print(self.tag_attn_w2)
        w2dt = self.tag_attn_w2(state)  # (B, A)
        # w1dt : (B, L, A)
        # att_weights: (seqlen,) row vector
        # print(w1dt.shape)
        # print(w2dt.shape)
        temp = torch.tanh(w1dt + w2dt)
        # print(temp.shape)
        unnormalized = self.tag_attn_v(temp)
        # (B, L, 1)
        att_weights = torch.softmax(unnormalized, dim=1)
        # print(att_weights.shape)
        # context: (encoder_state)
        return att_weights

    def attend(self, state, w1dt):
        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim x attdim)
        w2dt = self.attn_w2(state)
        # att_weights: (seqlen,) row vector
        unnormalized = self.attn_v(torch.tanh(w1dt + w2dt).transpose(1, 2))
        att_weights = torch.softmax(unnormalized, dim=1)
        # (B, L, 1)
        return att_weights

    def attend_with_prev(self, state, w1dt, prev_att):
        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim x attdim)
        # print(state.shape)
        w2dt = self.attn_w2(state)
        w3dt = self.attn_w3(prev_att)
        # att_weights: (seqlen,) row vector
        unnormalized = self.attn_v(torch.tanh(w1dt + w2dt + w3dt))
        att_weights = torch.softmax(unnormalized, dim=1)
        # (B, L, 1)
        return att_weights

    def decode(self, vectors: torch.Tensor, tag_vectors, outputs, lang_ids, weight, teacher_prob=1.0):
        # vectors is hopefully (B, L, D)
        outputs = [[EOS] + list(output) + [EOS] for output in outputs]
        outputs = torch.stack([torch.tensor([char2int[c] for c in output], dtype=torch.int) for output in outputs])
        batch_size, N, _ = vectors.shape
        _, tag_N, _ = tag_vectors.shape
        input_mat = vectors

        w1dt = None
        input_mat = torch.dropout(input_mat, DROPOUT_PROB, train=True)

        tag_input_mat = tag_vectors
        # (B, L, D) both
        tag_w1dt = None
        last_output_embeddings = self.output_lookup(torch.tensor([char2int[EOS] for _ in range(batch_size)]))
        temp = torch.cat([vectors[:, -1, :], tag_vectors[:, -1, :], last_output_embeddings], dim=1).unsqueeze(0)
        # print(temp.shape)
        _, (h_n, c_n) = self.dec_lstm(temp)
        # this is hacky because we're taking the last index even for the reverse direction

        loss = []
        prev_att = torch.zeros((batch_size, 5))

        if USE_ATT_REG:
            total_att = torch.zeros((batch_size, N, 1))
        if USE_TAG_ATT_REG:
            total_tag_att = torch.zeros((batch_size, tag_N, 1))
        assert batch_size == 1
        for char in outputs[0]:
            w1dt = w1dt if w1dt is not None else self.attn_w1(input_mat)
            tag_w1dt = tag_w1dt if tag_w1dt is not None else self.tag_attn_w1(tag_input_mat)
            state = torch.cat((h_n, c_n), dim=-1).squeeze(0)
            tag_att_weights = self.attend_tags(state, tag_w1dt)
            tag_context = torch.matmul(tag_input_mat.transpose(1, 2), tag_att_weights).squeeze(-1)
            # this was (B, D, L) x (B, L, 1)
            # (B, D)
            tag_context2 = torch.cat([tag_context, tag_context], dim=-1)
            # (B, 2D)
            new_state = state + tag_context2

            att_weights = self.attend_with_prev(new_state, w1dt, prev_att)

            context = torch.matmul(input_mat.transpose(1, 2), att_weights).squeeze(-1)
            # (B, 2D)

            best_ic = torch.argmax(att_weights, dim=1).squeeze().item()
            startt = min(best_ic - 2, N - 6)
            if startt < 0:
                startt = 0
            endd = startt + 5
            if N < 5:
                prev_att = torch.cat([att_weights] + [torch.zeros((batch_size, 1, 1))] * (5 - N), dim=1)
            else:
                prev_att = att_weights[:, startt:endd]
            prev_att = prev_att.squeeze(-1)
            assert prev_att.shape[1] == 5

            if USE_ATT_REG:
                total_att = total_att + att_weights
            if USE_TAG_ATT_REG:
                total_tag_att = total_tag_att + tag_att_weights

            vector = torch.cat([context, tag_context, last_output_embeddings], dim=1)
            s_out, (h_n, c_n) = self.dec_lstm(vector.unsqueeze(0), (h_n, c_n))
            s_out = s_out.squeeze(0)
            # (B, STATE_SIZE)
            s_out = torch.dropout(s_out, DROPOUT_PROB, train=True)

            out_vector = self.decoder(s_out)
            # (B, VOCAB_SIZE)
            probs = torch.softmax(out_vector, dim=-1).squeeze()
            if teacher_prob == 1.:
                last_output_embeddings = self.output_lookup(torch.tensor([char]))
            else:
                if random.random() > teacher_prob:
                    out_char = torch.argmax(probs)
                    last_output_embeddings = self.output_lookup(torch.tensor([out_char]))
                else:
                    last_output_embeddings = self.output_lookup(torch.tensor([char]))
            loss.append(-torch.log(probs[char]))
        loss = torch.sum(torch.stack(loss) * torch.tensor(weight, dtype=torch.float))
        if PREDICT_LANG:
            raise NotImplementedError()

        if USE_ATT_REG:
            loss += torch.nn.SmoothL1Loss()(torch.ones((batch_size, N, 1)), total_att)

        if USE_TAG_ATT_REG:
            loss += torch.nn.SmoothL1Loss()(torch.ones((batch_size, tag_N, 1)), total_tag_att)
        return loss

    @torch.no_grad()
    def generate(self, in_seq, tag_seq, show_att=False, show_tag_att=False, fn=None):
        embedded = self.embed_sentence([in_seq])
        encoded = self.encode_sentence(embedded)

        embedded_tags = self.embed_tags([tag_seq])
        # encoded_tags = self.encode_tags(embedded_tags)
        encoded_tags = self.self_encode_tags(embedded_tags)

        input_mat = encoded
        tag_input_mat = encoded_tags
        w1dt = None
        tag_w1dt = None

        prev_att = torch.zeros((1, 5))

        tmpinseq = [EOS] + list(in_seq) + [EOS]
        N = len(tmpinseq)

        last_output_embeddings = self.output_lookup(torch.tensor([char2int[EOS]]))
        _, (h_n, c_n) = self.dec_lstm(torch.cat([encoded[:, -1, :],
                                                encoded_tags[:, -1, :],
                                                last_output_embeddings], dim=1).unsqueeze(0))
        out = ''
        batch_size = 1
        count_EOS = 0
        if show_att:
            attt_weights = []
        if show_tag_att:
            ttt_weights = []
        for i in range(len(in_seq) * 2):
            w1dt = w1dt if w1dt is not None else self.attn_w1(input_mat)
            tag_w1dt = tag_w1dt if tag_w1dt is not None else self.tag_attn_w1(tag_input_mat)
            state = torch.cat((h_n, c_n), dim=-1).squeeze(0)
            tag_att_weights = self.attend_tags(state, tag_w1dt)
            tag_context = torch.matmul(tag_input_mat.transpose(1, 2), tag_att_weights).squeeze(-1)
            # this was (B, D, L) x (B, L, 1)
            # (B, D)
            tag_context2 = torch.cat([tag_context, tag_context], dim=-1)
            # (B, 2D)
            new_state = state + tag_context2

            att_weights = self.attend_with_prev(new_state, w1dt, prev_att)

            context = torch.matmul(input_mat.transpose(1, 2), att_weights).squeeze(-1)
            # (B, 2D)

            best_ic = torch.argmax(att_weights, dim=1).squeeze().item()
            startt = min(best_ic - 2, N - 6)
            if startt < 0:
                startt = 0
            end = startt + 5
            if N < 5:
                prev_att = torch.cat([att_weights] + [torch.zeros((batch_size, 1, 1))] * (5 - N), dim=1)
            else:
                prev_att = att_weights[:, startt:end]
            prev_att = prev_att.squeeze(-1)
            assert prev_att.shape[1] == 5

            if show_att:
                attt_weights.append(att_weights)
            if show_tag_att:
                ttt_weights.append(tag_att_weights)

            vector = torch.cat([context, tag_context, last_output_embeddings], dim=1)
            s_out, (h_n, c_n) = self.dec_lstm(vector.unsqueeze(0), (h_n, c_n))
            s_out = s_out.squeeze(0)
            # (B, STATE_SIZE)
            s_out = torch.dropout(s_out, DROPOUT_PROB, train=True)

            out_vector = self.decoder(s_out)
            # (B, VOCAB_SIZE)
            probs = torch.softmax(out_vector, dim=-1)
            next_char = np.argmax(probs)
            last_output_embeddings = self.output_lookup(torch.tensor([next_char]))
            if int2char[next_char] == EOS:
                count_EOS += 1
                continue

            out += int2char[next_char]

        if (show_att) and len(out) and fn is not None:
            arr = np.squeeze(np.array(attt_weights))[1:-1, 1:-1]
            fig, ax = plt.subplots()
            ax = plt.imshow(arr)
            x_positions = np.arange(0, len(attt_weights[0]) - 2)
            y_positions = np.arange(0, len(out))
            plt.xticks(x_positions, list(in_seq))
            plt.yticks(y_positions, list(out))
            plt.savefig(fn + '-char.png')
            plt.clf()
            plt.close()

        if (show_tag_att) and len(out) and fn is not None:
            arr = np.squeeze(np.array(ttt_weights))[1:-1, :]
            fig, ax = plt.subplots()
            ax = plt.imshow(arr)
            x_positions = np.arange(0, len(ttt_weights[0]))
            y_positions = np.arange(0, len(out))
            plt.xticks(x_positions, list(tag_seq))
            plt.yticks(y_positions, list(out))
            plt.savefig(fn + '-tag.png')
            plt.clf()
            plt.close()

        return out

    @torch.no_grad()
    def generate_nbest(self, in_seq, tag_seq, beam_size=4, show_att=False, show_tag_att=False, fn=None):
        embedded = self.embed_sentence([in_seq])
        encoded = self.encode_sentence(embedded)

        embedded_tags = self.embed_tags([tag_seq])
        # encoded_tags = self.encode_tags(embedded_tags)
        encoded_tags = self.self_encode_tags(embedded_tags)

        input_mat = encoded
        tag_input_mat = encoded_tags
        prev_att = torch.zeros((1, 5))

        tmpinseq = [EOS] + list(in_seq) + [EOS]
        N = len(tmpinseq)

        last_output_embeddings = self.output_lookup(torch.tensor([char2int[EOS]]))
        init_vector = torch.cat([encoded[:, -1, :], encoded_tags[:, -1, :], last_output_embeddings], dim=1).unsqueeze(0)
        s_0, (h_n, c_n) = self.dec_lstm(init_vector)
        w1dt = self.attn_w1(input_mat)

        tag_w1dt = self.tag_attn_w1(tag_input_mat)

        beam = {0: [(0, (h_n, c_n), [], prev_att)]}
        i = 1

        nbest = []
        # we'll need this
        last_states = {}

        MAX_PREDICTION_LEN = max(len(in_seq) * 1.5, MAX_PREDICTION_LEN_DEF)

        # expand another step if didn't reach max length and there's still beams to expand
        while i < MAX_PREDICTION_LEN and len(beam[i - 1]) > 0:
            # create all expansions from the previous beam:
            next_beam_id = []
            for hyp_id, hypothesis in enumerate(beam[i - 1]):
                # expand hypothesis tuple
                # prefix_seq, prefix_prob, prefix_decoder, prefix_context, prefix_tag_context = hypothesis
                prefix_prob, prefix_decoder, prefix_seq, prefix_att = hypothesis

                if i > 1:
                    last_hypo_symbol = prefix_seq[-1]
                else:
                    last_hypo_symbol = EOS

                # cant expand finished sequences
                if last_hypo_symbol == EOS and i > 3:
                    continue
                # expand from the last symbol of the hypothesis
                last_output_embeddings = self.output_lookup(torch.tensor([char2int[last_hypo_symbol] for _ in range(1)]))

                # Perform the forward step on the decoder
                # First, set the decoder's parameters to what they were in the previous step
                # if (i == 1):
                #     s = self.dec_lstm.initial_state().add_input(init_vector)
                # else:
                #     s = self.dec_lstm.initial_state(prefix_decoder)

                state = torch.cat(prefix_decoder, dim=-1).squeeze(0)
                tag_att_weights = self.attend_tags(state, tag_w1dt)
                tag_context = torch.matmul(tag_input_mat.transpose(1, 2), tag_att_weights).squeeze(-1)
                tag_context2 = torch.cat([tag_context, tag_context], dim=-1)
                new_state = state + tag_context2

                att_weights = self.attend_with_prev(new_state, w1dt, prefix_att)
                best_ic = torch.argmax(att_weights, dim=1).squeeze().item()
                startt = min(best_ic - 2, N - 6)
                if startt < 0:
                    startt = 0
                endd = startt + 5
                if N < 5:
                    prev_att = torch.cat([att_weights] + [torch.zeros((1, 1, 1))] * (5 - N), dim=1)
                else:
                    prev_att = att_weights[:, startt:endd]
                prev_att = prev_att.squeeze(-1)
                assert prev_att.shape[1] == 5
                context = torch.matmul(input_mat.transpose(1, 2), att_weights).squeeze(-1)

                vector = torch.cat([context, tag_context, last_output_embeddings], dim=1)
                s_0, new_prefix_decoder = self.dec_lstm(vector.unsqueeze(0), prefix_decoder)
                out_vector = self.decoder(s_0.squeeze(0))
                probs = torch.softmax(out_vector, dim=-1).squeeze(0)

                # Add length norm
                length_norm = torch.pow(torch.tensor([5 + i]),
                                        LENGTH_NORM_WEIGHT) / (np.power(torch.tensor([6]),
                                                                        LENGTH_NORM_WEIGHT))
                probs = probs / length_norm

                last_states[hyp_id] = new_prefix_decoder

                # find best candidate outputs
                n_best_indices = argmaxk(probs, beam_size)
                for index in n_best_indices:
                    this_score = prefix_prob + np.log(probs[index])
                    next_beam_id.append((this_score, hyp_id, index, prev_att))
                next_beam_id.sort(key=itemgetter(0), reverse=True)
                next_beam_id = next_beam_id[:beam_size]

            # Create the most probable hypotheses
            # add the most probable expansions from all hypotheses to the beam
            new_hypos = []
            for item in next_beam_id:
                hypid = item[1]
                this_prob = item[0]
                char_id = item[2]
                next_sentence = beam[i - 1][hypid][2] + [int2char[char_id]]
                new_hyp = (this_prob, last_states[hypid], next_sentence, item[3])
                new_hypos.append(new_hyp)
                if next_sentence[-1] == EOS or i == MAX_PREDICTION_LEN - 1:
                    if ''.join(next_sentence) != "<EOS>" and ''.join(next_sentence) != "<EOS><EOS>" and ''.join(
                            next_sentence) != "<EOS><EOS><EOS>":
                        nbest.append(new_hyp)

            beam[i] = new_hypos
            i += 1
            if len(nbest) > 0:
                nbest.sort(key=itemgetter(0), reverse=True)
                nbest = nbest[:beam_size]
            if len(nbest) == beam_size and (len(new_hypos) == 0 or (nbest[-1][0] >= new_hypos[0][0])):
                break

        return nbest

    def get_loss(self, input_sentences, input_tags, output_sentences, lang_ids, weight=1, tf_prob=1.0):
        embedded = self.embed_sentence(input_sentences)
        encoded = self.encode_sentence(embedded)

        embedded_tags = self.embed_tags(input_tags)

        encoded_tags = self.self_encode_tags(embedded_tags)

        return self.decode(encoded, encoded_tags, output_sentences, lang_ids, weight, tf_prob)

    def forward(self, input_sentences, input_tags, output_sentences, lang_ids, weight=1, tf_prob=1.0):
        return self.get_loss(input_sentences, input_tags, output_sentences, lang_ids, weight, tf_prob)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def populate(self, path):
        self.load_state_dict(torch.load(path))


def test_beam(inf_model, beam_size=4, fn=None):
    K = len(test_i)
    ks = list(range(K))
    correct = 0.0
    outs = []
    levs = []
    with codecs.open(fn, 'w', 'utf-8') as outf:
        for j, k in enumerate(ks):
            out = inf_model.generate_nbest(test_i[k], test_t[k], beam_size)
            if len(out):
                word = ''.join([c for c in out[0][2] if c != EOS])
            elif out:
                word = ''.join([c for c in out[0][2] if c != EOS])
            else:
                word = ''.join(test_i[k])
            outf.write(''.join(test_i[k]) + '\t' + word + '\t' + ';'.join(test_t[k]) + '\n')
            outs.append(word)
            lev = myutil.edit_distance(word, test_o[k])
            levs.append(lev)
            if list(word) == test_o[k]:
                correct += 1

    accuracy = correct/float(K)
    avg_edit = np.average(np.array(levs))
    return accuracy, avg_edit


def argmaxk(arr, k):
    k = min(k, len(arr))
    indices = torch.argsort(arr)
    # get k best indices
    indices = indices.flip((0, ))[:k]
    return indices


def eval_dev_greedy(inf_model, K=100, epoch=0):
    if K == "all":
        K = len(dev_i)
    ks = list(range(len(dev_i)))
    shuffle(ks)
    ks = ks[:K]
    outs = []
    levs = []
    correct = 0.0
    for j, k in enumerate(ks):
        out = inf_model.generate(dev_i[k], dev_t[k])
        outs.append(out)
        lev = myutil.edit_distance(out, dev_o[k])
        levs.append(lev)
        if list(out) == dev_o[k]:
            correct += 1

    accuracy = correct / float(K)
    avg_edit = np.average(np.array(levs))
    return accuracy, avg_edit


def eval_dev_copy_greedy(inf_model, K=40, epoch=0):
    if K == "all":
        K = len(dev_i)
    ks = list(range(len(dev_i)))
    shuffle(ks)
    ks = ks[:K]
    outs = []
    levs = []
    correct = 0.0
    for j, k in enumerate(ks):
        out = inf_model.generate(dev_i[k], [NULL])
        outs.append(out)
        lev = myutil.edit_distance(out, dev_i[k])
        levs.append(lev)
        if list(out) == dev_i[k]:
            correct += 1

    accuracy = correct / float(K)
    avg_edit = np.average(np.array(levs))
    return accuracy, avg_edit


def eval_dev_beam(inf_model, beam_size=4, K=100, epoch=0):
    if K == "all":
        K = len(dev_i)
    ks = list(range(len(dev_i)))
    shuffle(ks)
    ks = ks[:K]
    outs = []
    levs = []
    correct = 0.0
    for j,k in enumerate(ks):
        out = inf_model.generate_nbest(dev_i[k], dev_t[k], beam_size)
        if len(out):
            word = ''.join([c for c in out[0][2] if c != EOS])
        elif out:
            word = ''.join([c for c in out[0][2] if c != EOS])
        else:
            word = ''.join(dev_i[k])
        outs.append(word)
        lev = myutil.edit_distance(word, dev_o[k])
        levs.append(lev)
        if list(word) == dev_o[k]:
            correct += 1

    accuracy = correct/float(K)
    avg_edit = np.average(np.array(levs))
    return accuracy, avg_edit


def train_simple_attention_with_tags(inf_model, inputs, tags, outputs, lang_ids=None, finetune=False, trainer=None,
                                     prev_acc=None, prev_edd=None):
    indexes = list(range(len(inputs)))
    tasks = [0, 1, 2]
    burnin_pairs = [(j, t) for j in indexes for t in tasks[:2]]
    total_burnin_pairs = len(burnin_pairs)
    train_pairs = [(j, t) for j in indexes for t in tasks]
    total_train_pairs = len(train_pairs)
    finetune_pairs = [(j, t) for j in indexes for t in tasks[1:]]
    total_finetune_pairs = len(finetune_pairs)
    final_finetune_pairs = [(j, t) for j in indexes for t in [1]]
    total_final_finetune_pairs = len(finetune_pairs)

    learning_rate = STARTING_LEARNING_RATE
    trainer = trainer or optimizer_class(inf_model.parameters(), learning_rate)
    epochs_since_improv = 0
    halvings = 0
    # trainer.set_clip_threshold(-1.0)
    # trainer.set_sparse_updates(True if args.SPARSE == 1 else False)

    prev_acc = prev_acc or 0.0
    prev_edd = prev_edd or 100
    if lang_ids == None:
        lang_ids = np.zeros(len(burnin_pairs))
    # Stage 1
    if not finetune:
        # Learn to copy -- burn in
        MINIBATCH_SIZE = 64
        for i in range(100):
            shuffle(burnin_pairs)
            total_loss = 0.0
            batch = []
            trainer.zero_grad()

            def index_task_to_io(j, t):
                if t == 0:
                    return inputs[j], [NULL], inputs[j], lang_ids[j]
                else:
                    return outputs[j], tags[j], outputs[j], lang_ids[j]

            pairs_io = list(map(lambda x: index_task_to_io(*x), burnin_pairs))
            n = 0
            for example in data.BatchSampler(pairs_io, 1, drop_last=False):
                # task 0 is copy input
                # loss = inf_model.get_loss(inp, tag, otpt, lang_id)
                example = (list(map(lambda x: x[0], example)),  # input
                           list(map(lambda x: x[1], example)),  # tag
                           list(map(lambda x: x[2], example)),  # output
                           list(map(lambda x: x[3], example)))  # lang_id
                # print(example)
                loss = inf_model(*example)
                batch.append(loss)
                if len(batch) == MINIBATCH_SIZE:
                    loss = sum(batch) / len(batch)
                    total_loss += loss.item()
                    n += MINIBATCH_SIZE
                    print(f"Loss: {total_loss / n}", end="\r")
                    loss.backward()
                    trainer.step()
                    batch = []
                    trainer.zero_grad()
            if batch:
                loss = sum(batch) / len(batch)
                total_loss += loss.item()
                loss.backward()
                trainer.step()
                trainer.zero_grad()
            print()
            if i % 1 == 0:
                # trainer.status()
                print("Epoch " + str(i) + " : " + str(total_loss))
                acc, edd = eval_dev_copy_greedy(inf_model, 'all', i)
                print("\t COPY Accuracy: " + str(acc) + " average edit distance: " + str(edd))
            if edd < prev_edd:
                inf_model.save(os.path.join(MODEL_DIR, MODEL_NAME + "edd.model"))
            if (acc > prev_acc and edd < prev_edd) or (acc >= prev_acc and edd < prev_edd) or (
                    acc > prev_acc and edd <= prev_edd):
                inf_model.save(os.path.join(MODEL_DIR, MODEL_NAME + "both.model"))
                epochs_since_improv = 0
            else:
                epochs_since_improv += 1
            if acc > prev_acc:
                inf_model.save(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))
                epochs_since_improv = 0
            if acc > prev_acc:
                prev_acc = acc
            if edd < prev_edd:
                prev_edd = edd
            if epochs_since_improv > EPOCHS_TO_HALVE:
                print("Restarting the trainer with half the learning rate!")
                learning_rate = learning_rate / 2
                halvings += 1
                if halvings == 1:
                    break
                trainer = optimizer_class(inf_model.parameters(), learning_rate)
                epochs_since_improv = 0
                inf_model.populate(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))
            if acc > COPY_THRESHOLD:
                print("Accuracy good enough, breaking")
                break
        # Stage 2
        # epochs
        # We don't care for the performance on the copy burnin task
        halvings = 0
        prev_acc = 0.0
        prev_edd = 100
        MINIBATCH_SIZE = 1
        for i in range(40):
            shuffle(train_pairs)
            total_loss = 0.0
            batch = []
            weight = 0.0
            trainer.zero_grad()

            def index_task_to_io(j, t):
                if t == 0 or t == 1:
                    if random.random() > COPY_TASK_PROB:
                        return ()
                if t == 0:
                    return inputs[j], [NULL], inputs[j], lang_ids[j], COPY_WEIGHT
                elif t == 1:
                    return outputs[j], tags[j], outputs[j], lang_ids[j], COPY_WEIGHT
                elif t == 2:
                    return inputs[j], tags[j], outputs[j], lang_ids[j], 1.
                else:
                    raise NotImplementedError()

            pairs_io = list(filter(lambda x: x != (), map(lambda x: index_task_to_io(*x), train_pairs)))
            n = 0
            for example in data.BatchSampler(pairs_io, 1, drop_last=False):
                # task 0 is copy input
                # loss = inf_model.get_loss(inp, tag, otpt, lang_id)
                # print(example)
                example = (list(map(lambda x: x[0], example)),  # input
                           list(map(lambda x: x[1], example)),  # tag
                           list(map(lambda x: x[2], example)),  # output
                           list(map(lambda x: x[3], example)),  # lang_id
                           list(map(lambda x: x[4], example)),  # weight
                           )
                loss = inf_model(*example, tf_prob=0.8)
                weight += example[4][0]

                batch.append(loss)
                if len(batch) == MINIBATCH_SIZE:
                    loss = sum(batch) / weight
                    total_loss += loss.item()
                    n += MINIBATCH_SIZE
                    print(f"Loss: {total_loss / n}", end="\r")
                    loss.backward()
                    trainer.step()
                    batch = []
                    trainer.zero_grad()
                    weight = 0.0
            if batch:
                loss = sum(batch) / weight
                total_loss += loss.item()
                loss.backward()
                trainer.step()
                trainer.zero_grad()
            if i % 1 == 0:
                # trainer.status()
                print("Epoch ", i, " : ", total_loss)
                acc, edd = eval_dev_copy_greedy(inf_model, 20, 100 + i)
                print("\t COPY Accuracy: ", acc, " average edit distance: ", edd)
                acc, edd = eval_dev_greedy(inf_model, 100, 100 + i)
                print("\t TASK Accuracy: ", acc, " average edit distance: ", edd)
            if acc > prev_acc:
                inf_model.save(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))
                epochs_since_improv = 0
            if edd < prev_edd:
                inf_model.save(os.path.join(MODEL_DIR, MODEL_NAME + "edd.model"))
            if (acc > prev_acc and edd < prev_edd) or (acc >= prev_acc and edd < prev_edd) or (
                    acc > prev_acc and edd <= prev_edd):
                inf_model.save(os.path.join(MODEL_DIR, MODEL_NAME + "both.model"))
                epochs_since_improv = 0
            else:
                epochs_since_improv += 1
            if acc > prev_acc:
                inf_model.save(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))
                epochs_since_improv = 0
            if acc > prev_acc:
                prev_acc = acc
            if edd < prev_edd:
                prev_edd = edd
            if epochs_since_improv > EPOCHS_TO_HALVE:
                print("Restarting the trainer with half the learning rate!")
                halvings += 1
                if halvings == 2:
                    break
                learning_rate = learning_rate / 2
                trainer = optimizer_class(inf_model.parameters(), learning_rate)
                epochs_since_improv = 0
                inf_model.populate(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))
            if acc > 0.9 and epochs_since_improv == 4:
                print("Accuracy good enough, breaking")
                break


    else:
        MINIBATCH_SIZE = 1
        if learning_rate < 0.05:
            learning_rate = 0.05
        trainer = optimizer_class(inf_model.parameters(), learning_rate)
        halvings = 0
        for i in range(40):
            shuffle(finetune_pairs)
            total_loss = 0.0
            weight = 0.0
            batch = []
            trainer.zero_grad()

            def index_task_to_io(j, t):
                if t == 1:
                    if random.random() > COPY_TASK_PROB:
                        return ()
                    return outputs[j], tags[j], outputs[j], lang_ids[j], COPY_WEIGHT
                elif t == 2:
                    return inputs[j], tags[j], outputs[j], lang_ids[j], 1.
                else:
                    raise NotImplementedError()

            pairs_io = list(filter(lambda x: x != (), map(lambda x: index_task_to_io(*x), finetune_pairs)))
            n = 0
            for example in data.BatchSampler(pairs_io, 1, drop_last=False):
                # task 0 is copy input
                # loss = inf_model.get_loss(inp, tag, otpt, lang_id)
                # print(example)
                example = (list(map(lambda x: x[0], example)),  # input
                           list(map(lambda x: x[1], example)),  # tag
                           list(map(lambda x: x[2], example)),  # output
                           list(map(lambda x: x[3], example)),  # lang_id
                           list(map(lambda x: x[4], example)),  # weight
                           )
                loss = inf_model(*example, tf_prob=0.5)
                weight += example[4][0]
                batch.append(loss)
                if len(batch) == MINIBATCH_SIZE:
                    loss = sum(batch) / weight
                    total_loss += loss.item()
                    n += MINIBATCH_SIZE
                    print(f"Loss: {total_loss / n}", end="\r")
                    loss.backward()
                    trainer.step()
                    batch = []
                    trainer.zero_grad()
                    weight = 0.0
            if batch:
                loss = sum(batch) / weight
                total_loss += loss.item()
                loss.backward()
                trainer.step()
                trainer.zero_grad()
            if i % 1 == 0:
                print("Epoch ", i, " : ", total_loss)
                # trainer.status()
                acc, edd = eval_dev_copy_greedy(inf_model, 20, 140 + i)
                print("\t COPY Accuracy: ", acc, " average edit distance: ", edd)
                acc, edd = eval_dev_greedy(inf_model, "all", 140 + i)
                print("\t TASK Accuracy: ", acc, " average edit distance: ", edd)
            if acc > prev_acc:
                inf_model.save(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))
            if edd < prev_edd:
                inf_model.save(os.path.join(MODEL_DIR, MODEL_NAME + "edd.model"))
            if (acc > prev_acc and edd < prev_edd) or (acc >= prev_acc and edd < prev_edd) or (
                    acc > prev_acc and edd <= prev_edd):
                inf_model.save(os.path.join(MODEL_DIR, MODEL_NAME + "both.model"))
                epochs_since_improv = 0
            else:
                epochs_since_improv += 1
            if acc > prev_acc:
                prev_acc = acc
                epochs_since_improv = 0
            if edd < prev_edd:
                prev_edd = edd
            if epochs_since_improv > EPOCHS_TO_HALVE:
                print("Restarting the trainer with half the learning rate!")
                halvings += 1
                if halvings == 2:
                    break
                learning_rate = learning_rate / 2
                trainer = optimizer_class(inf_model.parameters(), learning_rate)
                epochs_since_improv = 0
                inf_model.populate(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))

        halvings = 0
        for i in range(40):
            shuffle(final_finetune_pairs)
            total_loss = 0.0
            batch = []
            trainer.zero_grad()

            def index_task_to_io(j, t):
                return inputs[j], tags[j], outputs[j], lang_ids[j]

            pairs_io = list(filter(lambda x: x != (), map(lambda x: index_task_to_io(*x), final_finetune_pairs)))
            n = 0
            for example in data.BatchSampler(pairs_io, 1, drop_last=False):
                # task 0 is copy input
                # loss = inf_model.get_loss(inp, tag, otpt, lang_id)
                # print(example)
                example = (list(map(lambda x: x[0], example)),  # input
                           list(map(lambda x: x[1], example)),  # tag
                           list(map(lambda x: x[2], example)),  # output
                           list(map(lambda x: x[3], example)),  # lang_id
                           )
                loss = inf_model(*example, tf_prob=0.5)
                batch.append(loss)
                if len(batch) == MINIBATCH_SIZE:
                    loss = sum(batch) / len(batch)
                    total_loss += loss.item()
                    n += MINIBATCH_SIZE
                    print(f"Loss: {total_loss / n}", end="\r")
                    loss.backward()
                    trainer.step()
                    batch = []
                    trainer.zero_grad()
            if batch:
                loss = sum(batch) / len(batch)
                total_loss += loss.item()
                loss.backward()
                trainer.step()
                trainer.zero_grad()
            if i % 1 == 0:
                print("Epoch ", i, " : ", total_loss)
                # trainer.status()
                acc, edd = eval_dev_copy_greedy(inf_model, 20, 160 + i)
                print("\t COPY Accuracy: ", acc, " average edit distance: ", edd)
                acc, edd = eval_dev_greedy(inf_model, "all", 160 + i)
                print("\t TASK Accuracy: ", acc, " average edit distance: ", edd)
            if acc > prev_acc:
                inf_model.save(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))
            if edd < prev_edd:
                inf_model.save(os.path.join(MODEL_DIR, MODEL_NAME + "edd.model"))
            if (acc > prev_acc and edd < prev_edd) or (acc >= prev_acc and edd < prev_edd) or (
                    acc > prev_acc and edd <= prev_edd):
                inf_model.save(os.path.join(MODEL_DIR, MODEL_NAME + "both.model"))
                epochs_since_improv = 0
            else:
                epochs_since_improv += 1
            if acc > prev_acc:
                prev_acc = acc
                epochs_since_improv = 0
            if edd < prev_edd:
                prev_edd = edd
            if epochs_since_improv > EPOCHS_TO_HALVE:
                print("Restarting the trainer with half the learning rate!")
                halvings += 1
                if halvings == 3:
                    break
                learning_rate = learning_rate / 2
                trainer = optimizer_class(inf_model.parameters(), learning_rate)
                epochs_since_improv = 0
                inf_model.populate(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))

    return trainer, prev_acc, prev_edd

inflection_model = None
app = Flask(__name__)


@app.route('/morph', methods=['POST', 'GET'])
def morph():
    if request.method == "POST":
        text = request.form['input_text']
        text = list(text)
        print(text)
        tags = request.form['input_tag']
        tags = tags.split(';')
        print(tags)
        out = inflection_model.generate_nbest(text, tags, beam_size=8)
        word = ''.join([c for c in out[0][2] if c != EOS])
        return render_template('morph.html', morph_word=word, input_text=''.join(text),
                               input_tag=";".join(tags))
    else:
        return render_template('morph.html')


@app.route('/', methods=['GET', 'POST'])
def welcome():
    if request.method == "POST":
        lp = request.form['lang_pair']
        # print(lp)
        characters = myutil.read_vocab(os.path.join(args.modelpath, lp, MODEL_NAME + "char.vocab"))
        if u' ' not in characters:
            characters.append(u' ')
        tags = myutil.read_vocab(os.path.join(args.modelpath, lp, MODEL_NAME + "tag.vocab"))
        global int2tag, int2char, tag2int, char2int
        int2char = list(characters)
        char2int = {c: i for i, c in enumerate(characters)}

        int2tag = list(tags)
        tag2int = {c: i for i, c in enumerate(tags)}
        global inflection_model
        inflection_model = InflectionModule(vocab_size=len(characters), tag_vocab_size=len(tags))
        inflection_model.populate(os.path.join(args.modelpath, lp, MODEL_NAME + "acc.model"))
        # print(inflection_model)
        return redirect(url_for('morph'))
    list_models = os.listdir(args.modelpath)
    # print(list_models)
    return render_template('index.html', lang_pairs=list_models)

# equivalent of main
if TRAIN:
    inflection_model = InflectionModule()
    if PREDICT_LANG:
        if ORIGINAL or SWAP:
            # lids_1 = [0]*MULTIPLY*len(low_i) + [1]*len(high_i)
            trainer, best_acc, best_edd = train_simple_attention_with_tags(inflection_model, MULTIPLY * low_i + high_i,
                                                                           MULTIPLY * low_t + high_t,
                                                                           MULTIPLY * low_o + high_o, lids_1)
            print("Best dev accuracy after pre-training: ", best_acc)
            print("Best dev lev distance after pre-training: ", best_edd)
            lids_2 = [0] * len(low_i)
            trainer2, best_acc, best_edd = train_simple_attention_with_tags(inflection_model, low_i, low_t, low_o, lids_2,
                                                                            True, trainer, best_acc, best_edd)
            print("Best dev accuracy after finetuning: ", best_acc)
            print("Best dev lev distance after finetuning: ", best_edd)
        elif LOW:
            lids_1 = [0] * len(low_i)
            trainer, best_acc, best_edd = train_simple_attention_with_tags(inflection_model, low_i, low_t, low_o, lids_1)
            print("Best dev accuracy after pre-training: ", best_acc)
            print("Best dev lev distance after pre-training: ", best_edd)
            lids_2 = [0] * len(low_i)
            trainer2, best_acc, best_edd = train_simple_attention_with_tags(inflection_model, low_i, low_t, low_o, lids_2,
                                                                            True, trainer, best_acc, best_edd)
            print("Best dev accuracy after finetuning: ", best_acc)
            print("Best dev lev distance after finetuning: ", best_edd)

    else:
        if ORIGINAL or SWAP:
            trainer, best_acc, best_edd = train_simple_attention_with_tags(inflection_model, MULTIPLY * low_i + high_i,
                                                                           MULTIPLY * low_t + high_t,
                                                                           MULTIPLY * low_o + high_o)
            print("Best dev accuracy after pre-training: ", best_acc)
            print("Best dev lev distance after pre-training: ", best_edd)
            trainer2, best_acc, best_edd = train_simple_attention_with_tags(inflection_model, low_i, low_t, low_o, None,
                                                                            True, trainer, best_acc, best_edd)
            print("Best dev accuracy after finetuning: ", best_acc)
            print("Best dev lev distance after finetuning: ", best_edd)
        elif LOW:
            trainer, best_acc, best_edd = train_simple_attention_with_tags(inflection_model, low_i, low_t, low_o)
            print("Best dev accuracy after pre-training: ", best_acc)
            print("Best dev lev distance after pre-training: ", best_edd)
            trainer2, best_acc, best_edd = train_simple_attention_with_tags(inflection_model, low_i, low_t, low_o, None,
                                                                            True, trainer, best_acc, best_edd)
            print("Best dev accuracy after finetuning: ", best_acc)
            print("Best dev lev distance after finetuning: ", best_edd)

elif TEST_DEV:
    inflection_model = InflectionModule()
    inflection_model.populate(os.path.join(MODEL_DIR, MODEL_NAME+"acc.model"))
    #acc, edd = eval_dev_greedy(enc_fwd_lstm, enc_bwd_lstm, dec_lstm, "all", "test")
    acc, edd = eval_dev_beam(inflection_model, 8, "all", "test") # it was 8 beams
    print("Best dev accuracy at test: ", acc)
    print("Best dev lev distance at test: ", edd)

elif TEST:
    inflection_model = InflectionModule()
    inflection_model.populate(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))
    if args.outputfile:
        acc, edd = test_beam(inflection_model, 8, args.outputfile)
    else:
        acc, edd = test_beam(inflection_model, 8, os.path.join(OUTPUT_DIR, MODEL_NAME + "test.output"))
    print("Best test accuracy at test: ", acc)
    print("Best test lev distance at test: ", edd)
elif DEMO:
    inflection_model = InflectionModule()
    app.run()

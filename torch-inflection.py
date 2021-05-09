import argparse
import codecs
import matplotlib
import dynet as dy
matplotlib.use('agg')
import matplotlib.pyplot as plt
import myutil
import numpy as np
from operator import itemgetter
import os, sys
from random import random, shuffle
import logging
import torch
import torch.utils.data as data

parser = argparse.ArgumentParser()
parser.add_argument("--datapath", help="path to data", type=str, required=True)
parser.add_argument("--L1", help="transfer languages (split with comma for multiple ones)", type=str, required=False)
parser.add_argument("--L2", help="test languages", type=str, required=True)
parser.add_argument("--mode", help="usage mode", type=str,
                    choices=['train', 'test',
                             # 'test-dev', 'draw-dev', 'test-dev-ensemble', 'test-ensemble',
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
parser.add_argument("--dynet-mem", help="set dynet memory", default=800, type=int, required=False)
parser.add_argument("--dynet-autobatch", help="use dynet autobatching (def: 1)", default=1, type=int, required=False)
args = parser.parse_args()


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

if args.mode == "train":
    TRAIN = True
elif args.mode == "test":
    TEST = True

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
STARTING_LEARNING_RATE = 0.1
EPOCHS_TO_HALVE = 6

MULTIPLY = 1
if len(high_i) + len(low_i) < 5000:
    MULTIPLY = 1
    STARTING_LEARNING_RATE = 0.2
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


class InflectionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_blstm = torch.nn.LSTM(input_size=EMBEDDINGS_SIZE, hidden_size=STATE_SIZE,
                                       num_layers=LSTM_NUM_OF_LAYERS, bidirectional=True)
        self.dec_lstm = torch.nn.LSTM(input_size=STATE_SIZE * 3 + EMBEDDINGS_SIZE, hidden_size=STATE_SIZE,
                                      num_layers=LSTM_NUM_OF_LAYERS, bidirectional=False)
        # TODO: embeddings have to add padding_idx if it's added
        self.input_lookup = torch.nn.Embedding(VOCAB_SIZE, EMBEDDINGS_SIZE)
        self.tag_input_lookup = torch.nn.Embedding(TAG_VOCAB_SIZE, EMBEDDINGS_SIZE)
        self.attn_w1 = torch.nn.Linear(STATE_SIZE * 2, ATTENTION_SIZE, bias=False)
        self.attn_w2 = torch.nn.Linear(STATE_SIZE * 2 * LSTM_NUM_OF_LAYERS, ATTENTION_SIZE, bias=False)
        self.attn_w3 = torch.nn.Linear(5, ATTENTION_SIZE, bias=False)
        self.attn_v = torch.nn.Linear(ATTENTION_SIZE, 1, bias=False)

        self.decoder = self.nn.Linear(STATE_SIZE, VOCAB_SIZE)
        self.output_lookup = self.input_lookup

        self.enc_tag_lstm = torch.nn.LSTM(input_size=EMBEDDINGS_SIZE, hidden_size=STATE_SIZE,
                                          num_layers=LSTM_NUM_OF_LAYERS, bidirectional=False)
        self.tag_attn_w1 = torch.nn.Linear(STATE_SIZE, ATTENTION_SIZE, bias=False)
        self.tag_attn_w2 = torch.nn.Linear(STATE_SIZE * 2 * LSTM_NUM_OF_LAYERS, ATTENTION_SIZE, bias=False)
        self.tag_attn_v = torch.nn.Linear(STATE_SIZE, ATTENTION_SIZE, bias=False)

        if PREDICT_LANG:
            self.lang_class_w = torch.nn.Linear(NUM_LANG, 2 * STATE_SIZE, bias=False)

    def embed_tags(self, tags_list):
        """
        param:
        tags_list: is a list of list of tags. batched basically.
        return (B, L, D)
        """
        int_tags = [[tag2int[t] for t in tags] for tags in tags_list]
        return torch.stack([torch.stack([self.tag_input_lookup[tag] for tag in tags]) for tags in int_tags])

    def embed_sentence(self, sentences):
        """
        return (B, L, D) sentences. currently (1, L, D)
        """
        # TODO: need to pad for bigger batch size than 1. currently no support for padding
        sentences = [[EOS] + list(sentence) + [EOS] for sentence in sentences]
        sentences = [[char2int[c] for c in sentence] for sentence in sentences]
        return torch.stack([torch.stack([self.input_lookup[char] for char in sentence]) for sentence in sentences])

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
        w2dt = self.tag_attn_w2(state)  # (B, A)
        # w1dt : (B, A, L)
        # att_weights: (seqlen,) row vector
        unnormalized = self.tag_attn_v(torch.tanh(w1dt + w2dt).transpose(1, 2))
        # (B, L, 1)
        att_weights = torch.softmax(unnormalized, dim=1)
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
        w2dt = self.attn_w2(state)
        w3dt = self.attn_w3(prev_att)
        # att_weights: (seqlen,) row vector
        unnormalized = self.attn_v(torch.tanh(w1dt + w2dt + w3dt).transpose(1, 2))
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
        last_output_embeddings = torch.stack([self.output_lookup[char2int[EOS]] for _ in range(batch_size)])
        temp_output, (h_n, c_n) = self.dec_lstm(torch.cat([vectors[:, -1, :],
                                                           tag_vectors[:, -1, :],
                                                           last_output_embeddings]).unsqueeze(0))
        temp_output = temp_output.squeeze(0)
        # this is hacky because we're taking the last index even for the reverse direction

        loss = []
        prev_att = torch.zeros((batch_size, 5))

        if USE_ATT_REG:
            total_att = torch.zeros((batch_size, N, 1))
        if USE_TAG_ATT_REG:
            total_tag_att = torch.zeros((batch_size, tag_N, 1))
        assert batch_size == 1
        for char in outputs[0]:
            w1dt = w1dt or self.attn_w1(input_mat)
            tag_w1dt = tag_w1dt or self.tag_attn_w1(tag_input_mat)
            state = h_n
            tag_att_weights = self.attend_tags(state, tag_w1dt)
            tag_context = torch.matmul(tag_input_mat.transpose(1, 2), tag_att_weights).squeeze(-1)
            # this was (B, D, L) x (B, L, 1)
            # (B, D)
            tag_context2 = torch.cat([tag_context, tag_context], dim=-1)
            # (B, 2D)
            new_state = state + tag_context

            att_weights = self.attend_with_prev(new_state, w1dt, prev_att)

            context = torch.matmul(input_mat.transpose(1, 2), att_weights).squeeze(-1)
            # (B, 2D)

            best_ic = torch.argmax(att_weights, dim=1).squeeze().item()
            startt = min(best_ic - 1, N - 6)
            if startt < 0:
                startt = 0
            end = startt + 5
            if N < 5:
                prev_att = torch.cat([att_weights] + [torch.zeros((batch_size, 1, 1))] * (5 - N), dim=1)
            else:
                prev_att = att_weights[:, startt:end]
            prev_att = prev_att.squeeze(-1)
            assert prev_att.shape[1] == 5

            if USE_ATT_REG:
                total_att = total_att + att_weights
            if USE_TAG_ATT_REG:
                total_tag_att = total_tag_att + tag_att_weights

            vector = torch.cat([context, tag_context, last_output_embeddings])
            s_out, (h_n, c_n) = self.dec_lstm(vector.unsqueeze(0), (h_n, c_n))
            s_out = s_out.squeeze(0)
            # (B, STATE_SIZE)
            s_out = torch.dropout(s_out, DROPOUT_PROB, train=True)

            out_vector = self.decoder(s_out)
            # (B, VOCAB_SIZE)
            probs = torch.softmax(out_vector, dim=-1)
            if teacher_prob == 1.:
                last_output_embeddings = self.output_lookup[char]
            else:
                raise NotImplementedError()
            loss.append(-torch.log(probs[char]))
        loss = sum(loss) * weight
        if PREDICT_LANG:
            raise NotImplementedError()

        if USE_ATT_REG:
            loss += torch.nn.SmoothL1Loss()(torch.ones((batch_size, N, 1)), total_att)

        if USE_TAG_ATT_REG:
            loss += torch.nn.SmoothL1Loss()(torch.ones((batch_size, tag_N, 1)), total_tag_att)
        return loss


def test_beam(inf_model, beam_size=4, fn=None):
    ks = list(range(len(test_i)))
    correct = 0.0
    with codecs.open(fn, 'w', 'utf-8') as outf:
        for j, k in enumerate(ks):
            out = inf_model.generate_nbest(test_i[k], test_t[k], beam_size)
            if len(out):
                word = ''.join([c for c in out[0][2] if c != EOS])
                out1 = ''.join(out[0][2][1:-1])
            elif out:
                word = ''.join([c for c in out[0][2] if c != EOS])
            else:
                word = ''.join(test_i[k])
            outf.write(''.join(test_i[k]) + '\t' + word + '\t' + ';'.join(test_t[k]) + '\n')

    return


def draw_decode(inf_model, K=20):
    for k in range(K):
        filename = os.path.join(FIGURE_DIR, str(k))
        inf_model.draw_decode(dev_i[k], dev_t[k], dev_o[k], show_att=True, show_tag_att=True, fn=filename)


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
    trainer = trainer or dy.SimpleSGDTrainer(inf_model.model, learning_rate)
    epochs_since_improv = 0
    halvings = 0
    # trainer.set_clip_threshold(-1.0)
    # trainer.set_sparse_updates(True if args.SPARSE == 1 else False)

    prev_acc = prev_acc or 0.0
    prev_edd = prev_edd or 100
    if lang_ids == None:
        lang_ids = np.zeros(len(burnin_pairs))

    if not finetune:
        # Learn to copy -- burn in
        MINIBATCH_SIZE = 10
        for i in range(100):
            shuffle(burnin_pairs)
            total_loss = 0.0
            batch = []
            dy.renew_cg()

            def index_task_to_io(j, t):
                if t == 0:
                    return inputs[j], [NULL], inputs[j], lang_ids[j]
                else:
                    return outputs[j], tags[j], outputs[j], lang_ids[j]

            pairs_io = list(map(lambda x: index_task_to_io(*x), burnin_pairs))

            for example in data.BatchSampler(pairs_io, 1, drop_last=False):
                # task 0 is copy input
                # loss = inf_model.get_loss(inp, tag, otpt, lang_id)
                example = (list(map(lambda x: x[0], example)), # input
                           list(map(lambda x: x[1], example)), # tag
                           list(map(lambda x: x[2], example)), # output
                           list(map(lambda x: x[3], example))) # lang_id
                print(example)
                loss = 0
                exit(1)
                batch.append(loss)
                if len(batch) == MINIBATCH_SIZE:
                    loss = dy.esum(batch) / len(batch)
                    total_loss += loss.value()
                    loss.backward()
                    trainer.update()
                    batch = []
                    dy.renew_cg()
            if batch:
                print(batch)
                loss = dy.esum(batch) / len(batch)
                total_loss += loss.value()
                loss.backward()
                trainer.update()
                dy.renew_cg()
            if i % 1 == 0:
                trainer.status()
                print("Epoch " + str(i) + " : " + str(total_loss))
                acc, edd = eval_dev_copy_greedy(inf_model, 'all', i)
                print("\t COPY Accuracy: " + str(acc) + " average edit distance: " + str(edd))
            if edd < prev_edd:
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME + "edd.model"))
            if (acc > prev_acc and edd < prev_edd) or (acc >= prev_acc and edd < prev_edd) or (
                    acc > prev_acc and edd <= prev_edd):
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME + "both.model"))
                epochs_since_improv = 0
            else:
                epochs_since_improv += 1
            if acc > prev_acc:
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))
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
                trainer.restart(learning_rate)
                epochs_since_improv = 0
                inf_model.model.populate(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))
            if acc > COPY_THRESHOLD:
                print("Accuracy good enough, breaking")
                break

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
            dy.renew_cg()
            for j, t in train_pairs:
                if (t == 0 or t == 1):
                    if random() > COPY_TASK_PROB:
                        continue
                if t == 0:
                    loss = inf_model.get_loss(inputs[j], [NULL], inputs[j], lang_ids[j], COPY_WEIGHT, 0.8)
                    weight += COPY_WEIGHT
                elif t == 1:
                    loss = inf_model.get_loss(outputs[j], tags[j], outputs[j], lang_ids[j], COPY_WEIGHT, 0.8)
                    weight += COPY_WEIGHT
                elif t == 2:
                    loss = inf_model.get_loss(inputs[j], tags[j], outputs[j], lang_ids[j], 1, 0.8)
                    weight += 1.0
                batch.append(loss)
                if len(batch) == MINIBATCH_SIZE or j == total_train_pairs:
                    loss = dy.esum(batch) / weight
                    total_loss += loss.value()
                    loss.backward()
                    trainer.update()
                    batch = []
                    dy.renew_cg()
                    weight = 0.0
            if i % 1 == 0:
                trainer.status()
                print("Epoch ", i, " : ", total_loss)
                acc, edd = eval_dev_copy_greedy(inf_model, 20, 100 + i)
                print("\t COPY Accuracy: ", acc, " average edit distance: ", edd)
                acc, edd = eval_dev_greedy(inf_model, 100, 100 + i)
                print("\t TASK Accuracy: ", acc, " average edit distance: ", edd)
            if acc > prev_acc:
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))
                epochs_since_improv = 0
            if edd < prev_edd:
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME + "edd.model"))
            if (acc > prev_acc and edd < prev_edd) or (acc >= prev_acc and edd < prev_edd) or (
                    acc > prev_acc and edd <= prev_edd):
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME + "both.model"))
                epochs_since_improv = 0
            else:
                epochs_since_improv += 1
            if acc > prev_acc:
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))
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
                trainer.restart(learning_rate)
                epochs_since_improv = 0
                inf_model.model.populate(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))
            if acc > 0.9 and epochs_since_improv == 4:
                print("Accuracy good enough, breaking")
                break


    else:
        MINIBATCH_SIZE = 1
        if learning_rate < 0.05:
            learning_rate = 0.05
        trainer.restart(learning_rate)
        halvings = 0
        for i in range(40):
            shuffle(finetune_pairs)
            total_loss = 0.0
            weight = 0.0
            batch = []
            dy.renew_cg()
            for j, t in finetune_pairs:
                if t == 1:
                    if random() > COPY_TASK_PROB:
                        continue
                    loss = inf_model.get_loss(outputs[j], tags[j], outputs[j], lang_ids[j], COPY_WEIGHT, 0.5)
                    weight += COPY_WEIGHT
                elif t == 2:
                    loss = inf_model.get_loss(inputs[j], tags[j], outputs[j], lang_ids[j], 1, 0.5)
                    weight += 1
                batch.append(loss)
                if len(batch) == MINIBATCH_SIZE or j == total_finetune_pairs:
                    loss = dy.esum(batch) / weight
                    total_loss += loss.value()
                    loss.backward()
                    trainer.update()
                    batch = []
                    dy.renew_cg()
                    weight = 0.0
            if i % 1 == 0:
                print("Epoch ", i, " : ", total_loss)
                trainer.status()
                acc, edd = eval_dev_copy_greedy(inf_model, 20, 140 + i)
                print("\t COPY Accuracy: ", acc, " average edit distance: ", edd)
                acc, edd = eval_dev_greedy(inf_model, "all", 140 + i)
                print("\t TASK Accuracy: ", acc, " average edit distance: ", edd)
            if acc > prev_acc:
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))
            if edd < prev_edd:
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME + "edd.model"))
            if (acc > prev_acc and edd < prev_edd) or (acc >= prev_acc and edd < prev_edd) or (
                    acc > prev_acc and edd <= prev_edd):
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME + "both.model"))
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
                trainer.restart(learning_rate)
                epochs_since_improv = 0
                inf_model.model.populate(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))

        halvings = 0
        for i in range(40):
            shuffle(indexes)
            total_loss = 0.0
            batch = []
            dy.renew_cg()
            for j, t in final_finetune_pairs:
                loss = inf_model.get_loss(inputs[j], tags[j], outputs[j], lang_ids[j], 1, 0.5)
                batch.append(loss)
                if len(batch) == MINIBATCH_SIZE or j == total_final_finetune_pairs:
                    loss = dy.esum(batch) / len(batch)
                    total_loss += loss.value()
                    loss.backward()
                    trainer.update()
                    batch = []
                    dy.renew_cg()
            if i % 1 == 0:
                print("Epoch ", i, " : ", total_loss)
                trainer.status()
                acc, edd = eval_dev_copy_greedy(inf_model, 20, 160 + i)
                print("\t COPY Accuracy: ", acc, " average edit distance: ", edd)
                acc, edd = eval_dev_greedy(inf_model, "all", 160 + i)
                print("\t TASK Accuracy: ", acc, " average edit distance: ", edd)
            if acc > prev_acc:
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))
            if edd < prev_edd:
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME + "edd.model"))
            if (acc > prev_acc and edd < prev_edd) or (acc >= prev_acc and edd < prev_edd) or (
                    acc > prev_acc and edd <= prev_edd):
                inf_model.model.save(os.path.join(MODEL_DIR, MODEL_NAME + "both.model"))
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
                trainer.restart(learning_rate)
                epochs_since_improv = 0
                inf_model.model.populate(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))

    return trainer, prev_acc, prev_edd


# equivalent of main
if TRAIN:
    inflection_model = InflectionModel()
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

elif TEST:
    inflection_model = InflectionModel()
    inflection_model.model.populate(os.path.join(MODEL_DIR, MODEL_NAME + "acc.model"))
    if args.outputfile:
        test_beam(inflection_model, 8, args.outputfile)
    else:
        test_beam(inflection_model, 8, os.path.join(OUTPUT_DIR, MODEL_NAME + "test.output"))

"""
"""
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from pprint import pprint


class Seq2SeqDataModule(pl.LightningDataModule):
    """

    """

    filename = '../data/fra-eng/fra.csv'
    chunk_idx = 0

    def __init__(self, batch_size, debug, n_chunks=None):
        super(Seq2SeqDataModule, self).__init__()
        self.batch_size = batch_size
        self.debug = debug
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.src_vocab_size = None
        self.tgt_vocab_size = None
        self.pad_idx = None
        self.src_vocab = None
        self.tgt_vocab = None
        self.n_chunks = n_chunks
        # if self.n_chunks is not None:
        self.chunk_size = None  # assingned in self.prepare_data()
        if debug:
            self.df = None

    @staticmethod
    def yield_sentence_tokens(sentence, tokenizer):
        return tokenizer(sentence)

    @staticmethod
    def yield_all_tokens(sentences, tokenizer):
        for sentence in sentences:
            yield Seq2SeqDataModule.yield_sentence_tokens(sentence, tokenizer)

    def build_vocab(self, sentences, tokenizer):
        '''
        Build vocab

        :param sentences:
        :param tokenizer:
        :return: Vocab

        '''

        # counter = Counter()
        # [counter.update(tokenizer(s)) for s in sentences]
        # for sentence in sentences:
        #     counter.update(tokenizer(sentence))
        z = Seq2SeqDataModule.yield_all_tokens(sentences, tokenizer)
        out_vocab = build_vocab_from_iterator(
            z, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        out_vocab.set_default_index(index=out_vocab['<unk>'])
        return out_vocab

    @staticmethod
    def data_process(src_tgt_df, src_vocab, tgt_vocab, src_tokenizer,
                     tgt_tokenizer):
        """
        convert source-target sentence pairs in df to list of lists of tokens
        :param src_tgt_df:
        :param src_vocab:
        :param tgt_vocab: 
        :param src_tokenizer: 
        :param tgt_tokenizer: 
        :return: Tensor[[src_tokens_sent1, tgt_tokens_sent1], ...]] 
        """

        src_sentences = src_tgt_df['en'].values
        tgt_sentences = src_tgt_df['fr'].values
        data = []
        for src_sent, tgt_sent in zip(src_sentences, tgt_sentences):
            src_tensor = torch.tensor(
                [src_vocab[x] for x in src_tokenizer(src_sent)])
            tgt_tensor = torch.tensor(
                [tgt_vocab[x] for x in tgt_tokenizer(tgt_sent)])
            data.append((src_tensor, tgt_tensor))
        return data

    @staticmethod
    def test_data_process(pair_df, sent_pairs, idx, src_tokenizer,
                          tgt_tokenizer):
        pprint(f"src: {pair_df.iloc[idx]['en']}")
        pprint(f"tgt: {pair_df.iloc[idx]['fr']}")
        pprint(f'src: {sent_pairs[idx][0]}')
        pprint(f'tgt: {sent_pairs[idx][1]}')

        pprint(f"src len: {len(src_tokenizer(pair_df.iloc[idx]['en']))}")
        pprint(f"tgt len: {len(tgt_tokenizer(pair_df.iloc[idx]['fr']))}")
        pprint(f'proccesed src len: {len(sent_pairs[idx][0])}')
        pprint(f'processed tgt len: {len(sent_pairs[idx][1])}')

    def prepare_data(self, n_chunks=None, chunk_idx=None):
        # load data into dataframe ('en' 'fr') sentence pairs
        en_fr_df = pd.read_csv(Seq2SeqDataModule.filename,
                               header=None,
                               sep='\t',
                               usecols=[0, 1],
                               names=['en', 'fr'])

        if self.debug:
            df_len = 500  # == chunk len
            # idx = range(chunk_idx * df_len, (1 + chunk_idx) * df_len)
            en_fr_df = en_fr_df.iloc[:df_len]
            self.df = en_fr_df

        # lowercase
        en_fr_df['en'] = en_fr_df['en'].str.lower()
        en_fr_df['fr'] = en_fr_df['fr'].str.lower()
        #
        en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
        #

        pprint("building en vocab")
        en_vocab = self.build_vocab(en_fr_df['en'].values, en_tokenizer)
        pprint("building fr vocab")
        fr_vocab = self.build_vocab(en_fr_df['fr'].values, fr_tokenizer)
        #
        self.src_vocab = en_vocab
        self.tgt_vocab = fr_vocab
        self.src_vocab_size = en_vocab.__len__()
        self.tgt_vocab_size = fr_vocab.__len__()
        self.pad_idx = en_vocab["<pad>"]

        pprint(
            f'vocal sizes: en {self.src_vocab_size}, fr {self.tgt_vocab_size}')
        #
        # train, test, val (60%, 20%, 20%)
        # df_len = len(en_fr_df)

        # if self.debug:
        #     # df_len = len(en_fr_df)  # // 100  # == chunk len
        #     # idx = range(chunk_idx * df_len, (1 + chunk_idx) * df_len)
        #     en_fr_df = en_fr_df.iloc  # [:df_len]

        train, val, test = np.split(
            en_fr_df.sample(frac=1),
            [int(0.6 * df_len), int(0.8 * df_len)])

        pprint((f"train shape: {train.shape}"
                f" val shape: {val.shape}"
                f"test shape: {test.shape}"))
        self.train_data = Seq2SeqDataModule.data_process(
            train, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer)
        self.val_data = Seq2SeqDataModule.data_process(val, en_vocab, fr_vocab,
                                                       en_tokenizer,
                                                       fr_tokenizer)
        self.test_data = Seq2SeqDataModule.data_process(
            test, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer)

    def setup(self, n_chunks=None, chunk_idx=None):
        self.prepare_data(n_chunks, chunk_idx)

    @staticmethod
    def generate_batch(data_batch):
        """
        collate function for DataLoader
        :param data_batch: src - target sentence token pairs
        :return: batch
        """
        pad_idx = 1
        bos_idx = 2
        eos_idx = 3
        en_batch, fr_batch = [], []
        for (en_item, fr_item) in data_batch:
            en_batch.append(
                torch.cat([
                    torch.tensor([bos_idx]), en_item,
                    torch.tensor([eos_idx])
                ],
                          dim=0))
            fr_batch.append(
                torch.cat([
                    torch.tensor([bos_idx]), fr_item,
                    torch.tensor([eos_idx])
                ],
                          dim=0))
        fr_batch = pad_sequence(fr_batch,
                                padding_value=pad_idx,
                                batch_first=True)
        en_batch = pad_sequence(en_batch,
                                padding_value=pad_idx,
                                batch_first=True)
        return en_batch, fr_batch

    def train_dataloader(self):
        train_iter = DataLoader(self.train_data,
                                batch_size=self.batch_size,
                                shuffle=False,
                                collate_fn=Seq2SeqDataModule.generate_batch,
                                num_workers=6)
        # , train=True)
        return train_iter

    def val_dataloader(self):
        val_iter = DataLoader(self.val_data,
                              batch_size=self.batch_size,
                              shuffle=False,
                              collate_fn=Seq2SeqDataModule.generate_batch,
                              num_workers=6)

        return val_iter

    def test_dataloader(self):
        test_iter = DataLoader(self.test_data,
                               batch_size=self.batch_size,
                               shuffle=False,
                               collate_fn=Seq2SeqDataModule.generate_batch,
                               num_workers=6)

        return test_iter


def test_seq2seqDataModule():
    dl = Seq2SeqDataModule(batch_size=32, debug=True)
    dl.setup()
    ti = dl.train_dataloader()
    # import ipdb
    # ipdb.set_trace()
    print(" ")

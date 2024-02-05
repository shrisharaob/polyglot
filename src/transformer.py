""" Transformer Class """
import torch
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl
from encoder import Encoder
from decoder import Decoder
from loaders import Seq2SeqDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from utils import translate_sentence
import glob
import random
import pickle


class Transformer(pl.LightningModule):
    """Documentation for Transformer

    """

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 src_pad_idx,
                 tgt_pad_idx,
                 embed_size,
                 num_transformer_modules,
                 num_heads,
                 forward_expansion,
                 dropout_ratio=0.0,
                 max_seq_len=50,
                 learning_rate=1e-3):
        super(Transformer, self).__init__()
        self.save_hyperparameters()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.embed_size = embed_size
        self.num_transformer_modules = num_transformer_modules
        self.num_heads = num_heads
        self.forward_expansion = forward_expansion
        self.dropout_ratio = dropout_ratio
        self.max_seq_len = max_seq_len
        # learning attributes
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tgt_pad_idx)
        # Layers
        self.encoder = Encoder(src_vocab_size, embed_size,
                               num_transformer_modules, num_heads,
                               dropout_ratio, forward_expansion, max_seq_len)

        self.decoder = Decoder(tgt_vocab_size, embed_size,
                               num_transformer_modules, num_heads,
                               dropout_ratio, forward_expansion, max_seq_len)

    def create_src_mask(self, source):
        """

        """
        src_mask = (source != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (batch_size, 1, 1, src_seq_len)
        return src_mask

    def create_tgt_mast(self, target):
        """
        Keyword Arguments:
        target --
        """
        batch_size, seq_len = target.shape
        tgt_mask = torch.tril(torch.ones(
            (seq_len, seq_len))).expand(batch_size, 1, seq_len, seq_len)
        return tgt_mask

    def forward(self, src, tgt):
        """
        Keyword Arguments:
        src --
        tgt --
        """
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mast(tgt)
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, tgt_mask, encoder_output, src_mask)
        return decoder_output

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)

    def compute_loss(self, y_pred, y):
        return self.criterion(y_pred.permute(0, 2, 1), y)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_pred = self(x, y)
        loss = self.compute_loss(y_pred, y)
        # tensorboard_logs = {'train_loss': loss.detach()}
        self.log("loss", loss, logger=True)
        return loss  # {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nd):
        x, y = batch
        y_pred = self(x, y)
        loss = self.compute_loss(y_pred, y)
        # tensorboard_logs = {'val_loss': loss.detach()}
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        # return {'loss': loss, 'log': tensorboard_logs}

    def on_validation_epoch_end(self):
        """Log hp_metric to tensorboard for hparams selection."""
        hp_metric = self.trainer.callback_metrics.get("val_loss", None)
        if hp_metric is not None:
            self.trainer.logger.log_metrics({"hp_metric": hp_metric},
                                            step=self.trainer.global_step)

    def test_step(self, batch, batch_nd):
        x, y = batch
        y_pred = self(x, y)
        loss = self.compute_loss(y_pred, y)
        # out = dict({'test_loss': loss})
        self.log("test_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("avg_train_loss", avg_loss, on_epoch=True, prog_bar=True)

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.
        Args:
            dic (dict): Dictionary to be transformed.
        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.Tensor(v)
        return dic


def test_main():
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], \
                      [1, 8, 7, 3, 4, 5, 6, 7, 2]])
    tgt = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], \
                        [1, 5, 6, 2, 4, 7, 6, 2]])

    src_pad_idx = 0
    tgt_pad_idx = 0
    src_vocab_size = 10
    tgt_vocab_size = 10
    embed_size = 512
    num_transformer_modules = 6
    forward_expansion = 4
    num_heads = 8
    dropout_ratio = 0.0
    max_seq_len = 100
    learning_rate = 1e-3
    model = Transformer(src_vocab_size=src_vocab_size,
                        tgt_vocab_size=tgt_vocab_size,
                        src_pad_idx=src_pad_idx,
                        tgt_pad_idx=tgt_pad_idx,
                        embed_size=embed_size,
                        num_transformer_modules=num_transformer_modules,
                        num_heads=num_heads,
                        forward_expansion=forward_expansion,
                        dropout_ratio=dropout_ratio,
                        max_seq_len=max_seq_len,
                        lr=learning_rate)
    out = model(x, tgt[:, :-1])
    # out = model(x, tgt)
    print(x.shape)
    print(tgt.shape)
    print(out.shape)


def demo_debug(batch_size=3, if_train=False):
    embed_size = 512
    num_transformer_modules = 6
    forward_expansion = 4
    num_heads = 4
    dropout_ratio = 0.0
    max_seq_len = 100
    learning_rate = 1e-3

    #
    data_loader = Seq2SeqDataModule(batch_size=batch_size, debug=True)
    data_loader.setup()

    #
    src_pad_idx = data_loader.pad_idx
    tgt_pad_idx = data_loader.pad_idx
    src_vocab_size = data_loader.src_vocab_size
    tgt_vocab_size = data_loader.tgt_vocab_size

    #
    model = Transformer(src_vocab_size=src_vocab_size,
                        tgt_vocab_size=tgt_vocab_size,
                        src_pad_idx=src_pad_idx,
                        tgt_pad_idx=tgt_pad_idx,
                        embed_size=embed_size,
                        num_transformer_modules=num_transformer_modules,
                        num_heads=num_heads,
                        forward_expansion=forward_expansion,
                        dropout_ratio=dropout_ratio,
                        max_seq_len=max_seq_len,
                        learning_rate=learning_rate)

    # try:
    #     load_checkpoint(model)
    # except Exception:
    #     pass

    #

    logger = TensorBoardLogger("test_run_logs", name="my_model")
    if if_train:
        refresh_rate = 10
        max_epochs = 2000

        #
        early_stopping = EarlyStopping('val_loss')
        #
        trainer = pl.Trainer(progress_bar_refresh_rate=refresh_rate,
                             log_every_n_steps=2,
                             max_epochs=max_epochs,
                             logger=logger)
        # callbacks=[early_stopping])

        # learn
        trainer.fit(model, data_loader)

        # test on train dataset
        trainer.test(test_dataloaders=data_loader.train_dataloader())

        # test on test dataset
        trainer.test(test_dataloaders=data_loader.test_dataloader())

        #
        demo_translate(model, data_loader, logger.root_dir)

    else:
        demo_translate_train_sentences(model, data_loader, logger.root_dir)
        demo_translate(model, data_loader, logger.root_dir)


# def chunk_train(batch_size):
#     embed_size = 50  # 512
#     num_transformer_modules = 2  # 6
#     forward_expansion = 2  # 4
#     num_heads = 2  # 4
#     dropout_ratio = 0.0
#     max_seq_len = 100
#     learning_rate = 1e-3

#     #

#     #
#     data_loader = Seq2SeqDataModule(batch_size=batch_size, debug=True)
#     data_loader.setup()

#     #
#     src_pad_idx = data_loader.pad_idx
#     tgt_pad_idx = data_loader.pad_idx
#     src_vocab_size = data_loader.src_vocab_size
#     tgt_vocab_size = data_loader.tgt_vocab_size

#     #
#     model = Transformer(src_vocab_size=src_vocab_size,
#                         tgt_vocab_size=tgt_vocab_size,
#                         src_pad_idx=src_pad_idx,
#                         tgt_pad_idx=tgt_pad_idx,
#                         embed_size=embed_size,
#                         num_transformer_modules=num_transformer_modules,
#                         num_heads=num_heads,
#                         forward_expansion=forward_expansion,
#                         dropout_ratio=dropout_ratio,
#                         max_seq_len=max_seq_len,
#                         learning_rate=learning_rate)
#     # try:
#     #     load_checkpoint(model)
#     # except Exception:
#     #     pass
#     #
#     logger = TensorBoardLogger("tb_logs", name="my_model")

#     import ipdb
#     ipdb.set_trace()
#     refresh_rate = 10
#     max_epochs = 10
#     trainer = pl.Trainer(progress_bar_refresh_rate=refresh_rate,
#                          log_every_n_steps=50,
#                          max_epochs=max_epochs,
#                          logger=logger)
#     # learn
#     trainer.fit(model, data_loader)
#     # test on train dataset
#     trainer.test(test_dataloaders=data_loader.train_dataloader())
#     # test on test dataset
#     trainer.test(test_dataloaders=data_loader.test_dataloader())
#     #
#     demo_translate(model, data_loader, chkp_fldr="./")


def load_checkpoint(model, chk_fldr):
    # trainer = pl.Trainer()
    # chk_path = glob.glob("lightning_logs/*/checkpoints/*.ckpt")
    # chk_path = glob.glob("./tb_logs/my_model/*/checkpoints/*.ckpt")
    chk_path = glob.glob(f"{chk_fldr}/*/checkpoints/*.ckpt")
    chk_path.sort()
    chk_path = chk_path[-1]
    print(51 * '-')
    print(f"loading form: {chk_path}")
    print(51 * '-')
    model = model.load_from_checkpoint(chk_path)
    return model


def demo_translate(model, data_loader, chkp_fldr, sentence="Hi what up?"):
    load_checkpoint(model, chkp_fldr)
    max_length = model.max_seq_len // 2
    model.eval()
    while True:
        sentence = input("input sentence: ")
        translate_sentence(model, data_loader, sentence, max_length)
    model.train()


def demo_translate_train_sentences(model,
                                   data_loader,
                                   chkp_fldr,
                                   sentence="Hi what up?"):
    load_checkpoint(model, chkp_fldr)
    max_length = model.max_seq_len // 2

    src_tgt_df = data_loader.df
    df_len = len(src_tgt_df)

    model.eval()
    for i in range(10):
        idx = random.randint(0, df_len - 1)
        sentence = src_tgt_df['en'].iloc[idx]
        tgt = src_tgt_df['fr'].iloc[idx]
        print("=" * 50)
        print(f"src: {sentence}")
        print(50 * '-')
        print(f"tgt: {tgt}")
        print("=" * 50)

        translate_sentence(model, data_loader, sentence, max_length)
    model.train()


if __name__ == '__main__':
    demo_debug(if_train=True, batch_size=64)
    # test_main()

# Transformer Neural Machine Translation

This repository contains vanilla implementation of a Transformer-based neural machine translation model using PyTorch and PyTorch Lightning. The Transformer model is a popular architecture for sequence-to-sequence tasks including machine translation.

## Files

`transformer.py`

Main implementation of the Transformer class, a PyTorch Lightning module.
Includes the encoder and decoder components for sequence-to-sequence tasks.
Designed for machine translation but adaptable for other sequence-to-sequence applications.

`encoder.py`

Implementation of the encoder component of the Transformer model.
Processes the input sequence and produces representations used by the decoder.

`decoder.py`

Implementation of the decoder component of the Transformer model.
Generates the output sequence based on representations obtained from the encoder.

`loaders.py`

Contains the Seq2SeqDataModule for handling sequence-to-sequence data.
Provides functionality for loading and preprocessing datasets for training, validation, and testing.


`utils.py`

Includes utility functions, such as translate_sentence for translating input sequences using the trained model.


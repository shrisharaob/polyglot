import torch
from torchtext.data.utils import get_tokenizer


def translate_sentence(model, data_module, sentence, max_length):
    # Load src_language tokenizer
    en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    # if type(sentence) == str:
    #     tokens = [token.text.lower() for token in en_tokenizer(sentence)]
    # else:
    #     tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    # tokens.insert(0, src_language.init_token)
    # tokens.append(src_language.eos_token)

    # Go through each src_language token and convert to an index
    # text_to_indices = [src_language.vocab.stoi[token] for token in tokens]
    # text_to_indices = [data_module.src_vocab.stoi[token] for token in tokens]

    tokens = list(data_module.yield_sentence_tokens(sentence, en_tokenizer))

    token_indices = data_module.src_vocab.lookup_indices(tokens)
    token_indices = ([data_module.src_vocab["<bos>"]] + token_indices +
                     [data_module.src_vocab["<eos>"]])
    sentence_tensor = torch.LongTensor(token_indices).unsqueeze(0)

    # Convert to Tensor
    # sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1)

    # import ipdb
    # ipdb.set_trace()

    outputs = [data_module.tgt_vocab["<bos>"]]
    model.eval()
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(0)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        # best_guess = output.argmax(2)[-1, :].item()
        best_guess = output.argmax(2)[-1, :][-1].item()
        # if best_guess == 204:
        #     import ipdb
        #     ipdb.set_trace()
        outputs.append(best_guess)

        # import ipdb
        # ipdb.set_trace()
        if best_guess == data_module.tgt_vocab["<eos>"]:
            break

    model.train()
    # translated_sentence = [tgt_language.vocab.itos[idx] for idx in outputs]

    target_tokens = data_module.tgt_vocab.lookup_tokens(outputs)  # [1:])
    tgt_sentence = ' '.join([w for w in target_tokens])

    print(' ')
    print(51 * "-")
    print(f"input sentence: {sentence}")
    print(51 * "-")
    print(f"translated sentence: {tgt_sentence}")
    print(51 * "-")

    # remove start token
    # return translated_sentence[1:]
    return tgt_sentence

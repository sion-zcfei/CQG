import torch
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP.embeddings import StaticEmbedding
from fastNLP.io import DataBundle, Pipe
from fastNLP.io import Loader
from fastNLP import Trainer, GradientClipCallback, WarmupCallback, EvaluateCallback
from fastNLP import CrossEntropyLoss
from fastNLP.embeddings import BertWordPieceEncoder
from fastNLP import BucketSampler
from transformers.utils.dummy_flax_objects import FlaxGPTNeoForCausalLM
#from fastNLP.modules.decoder import TransformerSeq2SeqDecoder
from model.Decoder import TransformerSeq2SeqDecoder
#from fastNLP.modules.generator import SequenceGenerator
from model.Generator import SequenceGenerator
from torch.optim import Adam
from fastNLP.models import CNNText, STSeqCls
from fastNLP import AccuracyMetric
from transformers import BertTokenizer
from torch.autograd import Variable
from functools import partial
import random
import argparse
import json
from torch import nn
from transformers import BertModel
import copy
from rouge import Rouge
import numpy as np


SEP = '[SEP]'
CLS = '[CLS]'
PAD = '[PAD]'
BOS = '[CLS]'
EOS = '[SEP]'
ANS_SPLIT = '[unused3]'

class Bert2tf(nn.Module):
    def __init__(self, opt, tokenizer):
        super(Bert2tf, self).__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        #self.encoder = RobertaModel.from_pretrained("bert-base-multilingual-cased")
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.vocab)
        self.hidden_size = 768
        self.pad_index = 0
        #print(tokenizer.cls_token, tokenizer.pad_token, tokenizer.sep_token)
        self.bos_index = self.tokenizer.vocab[BOS]
        self.eos_index = self.tokenizer.vocab[EOS]
        tgt_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        tgt_embedding = self.encoder.get_input_embeddings()
        
        self.ffn_layer_norm = nn.LayerNorm(self.hidden_size)

    
        self.decoder = TransformerSeq2SeqDecoder(tgt_embedding,
                                                 d_model=self.hidden_size,
                                                 pos_embed=None,
                                                 num_layers=6)

        self.generator = SequenceGenerator(self.decoder, num_beams=1,
                                           do_sample=False,
                                           max_length=opt.max_seq_length,
                                           bos_token_id=self.bos_index,
                                           eos_token_id=self.eos_index)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_index)


    def forward(self, words, target, flag):
        #print(words)
        #print(target)
        #print(self.vocab_size)
        encoder_output = self.encoder(words)[0]
        encoder_output = self.ffn_layer_norm(encoder_output)
        # print(select_binary.squeeze(-1))
        #print(encoder_output.size())
        encoder_mask = words.ne(self.pad_index)
        state = self.decoder.init_state(encoder_output, encoder_mask)
        output = self.decoder(target[:, :-1], state, flag[:, :-1, :])
        batch_size, tgt_len, _ = output.size()
        pred = output.reshape(batch_size * tgt_len, -1)
        gold = target[:, 1:].contiguous().view(-1)
        loss = self.criterion(pred, gold)
        
        return {'loss': loss}

    def generate(self, words, flag):
        encoder_output = self.encoder(words)[0]
        encoder_output = self.ffn_layer_norm(encoder_output)
        encoder_mask = words.ne(self.pad_index)
        state = self.decoder.init_state(encoder_output, encoder_mask)
        pred = self.generator.generate(state, flag=flag)
        batch_size, tgt_len = pred.size(0), pred.size(1)
        hypothesis = []
        for i in range(batch_size):
            words = pred[i].tolist()
            prediction = []
            for word in words:
                if word == self.bos_index:
                    continue
                elif word == self.eos_index:
                    break
                else:
                    prediction.append(word)
            hypothesis.append(prediction)
        return hypothesis

import random
import numpy as np
import torch
import torch.nn as nn
import argparse
from utils.config import Config
from dataset import FCDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.util import get_optimizers, clip_gradients
from transformers.models.mbart.modeling_mbart import shift_tokens_right
import nltk
import os


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if 'cuda' in args.device:
        torch.cuda.manual_seed_all(args.seed)


def parse_arguments(parser: argparse.ArgumentParser):
    # data Hyperparameters
    parser.add_argument('--device', type=str, default="cuda:0", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'],
                        help="GPU/CPU devices")
    parser.add_argument('--batch_size', type=int, default=32, help="default batch size is 10 (works well)")
    parser.add_argument('--max_seq_length', type=int, default=100, help="maximum sequence length")
    parser.add_argument('--generated_max_length', type=int, default=150, help="maximum target length")
    parser.add_argument('--max_candidate_length', type=int, default=20, help="maximum number of candidate tokens")
    parser.add_argument('--train_num', type=int, default=-1, help="The number of training data, -1 means all data")
    parser.add_argument('--dev_num', type=int, default=-1, help="The number of development data, -1 means all data")
    parser.add_argument('--test_num', type=int, default=-1, help="The number of development data, -1 means all data")

    parser.add_argument('--train_file', type=str, default="data/train.json")
    parser.add_argument('--dev_file', type=str, default="data/dev.json")
    parser.add_argument('--test_file', type=str, default="data/test.json")
    parser.add_argument('--gpus', type=bool, default=True)


    parser.add_argument('--seed', type=int, default=42, help="random seed")

    # model
    parser.add_argument('--model_folder', type=str, default="mbart_generate_answer",
                        help="the name of the models, to save the model")
    parser.add_argument('--bert_folder', type=str, default="",
                        help="The folder name that contains the BERT model")
    parser.add_argument('--bert_model_name', type=str, default="mbart-large-cc25", help="The bert model name to used")

    # training
    parser.add_argument('--mode', type=str, default="train", help="training or testing")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="learning rate of the AdamW optimizer")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The maximum gradient norm")
    parser.add_argument('--num_epochs', type=int, default=50, help="The number of epochs to run")
    parser.add_argument('--early_stop', type=int, default=8, help="The number of epochs to early stop")
    parser.add_argument('--task', type=str, default="bert2tf")

    parser.add_argument('--fp16', type=int, default=0, choices=[0, 1], help="fp16")

    args = parser.parse_args()
    # Print out the arguments
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def train(config: Config, train_dataloader: DataLoader, num_epochs: int, early_stop: int, 
          bert_model_name: str, dev: torch.device, valid_dataloader: DataLoader = None, tokenizer=None):
    metric = None
    #metric = load_metric("bleu")
    gradient_accumulation_steps = 1
    t_total = int(len(train_dataloader) // gradient_accumulation_steps * num_epochs)

    model = Bert2tf(config, tokenizer)
    model.to(dev)
    device_ids = [0,1,2,3]
    if config.gpus:
        model = nn.DataParallel(model, device_ids=device_ids, output_device=0)
    if config.fp16:
        scaler = torch.cuda.amp.GradScaler(enabled=bool(config.fp16))
    optimizer, scheduler = get_optimizers(config, model, t_total, warmup_step=0, eps=1e-8, weight_decay=0.0)
    #optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
    optimizer.zero_grad()
    model.zero_grad()
    best_accuracy = -1
    os.makedirs(f"model_files/{config.model_folder}", exist_ok=True)  ## create model files. not raise error if exist
    stop_num = 0
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for iter, batch in tqdm(enumerate(train_dataloader, 1), desc="--training batch", total=len(train_dataloader)):
            target_id = batch.question_ids.to(dev)
            #print(target_id.size())
            lm_labels = target_id.clone()
            lm_labels[target_id == 0] = -100
            input_ids = batch.input_ids.to(dev)
            flag = batch.flag.to(dev)
            decoder_input_ids = shift_tokens_right(target_id, 0)
            #mask = batch.attention_mask.to(dev)
            #decoder_input_ids = shift_tokens_right(target_id, tokenizer.pad_token_id)
            with torch.cuda.amp.autocast(enabled=bool(config.fp16)):
                outputs = model(input_ids, decoder_input_ids, flag)
            loss = outputs['loss']
            loss = torch.sum(loss)
            if config.fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            clip_gradients(model, config.max_grad_norm, dev)
            total_loss += loss.item()
            if config.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            model.zero_grad()
            if iter % 1000 == 0:
                print(f"epoch: {epoch}, iteration: {iter}, current mean loss: {total_loss / iter:.2f}", flush=True)
                torch.cuda.empty_cache()
        print(f"Finish epoch: {epoch}, loss: {total_loss:.2f}, mean loss: {total_loss / len(train_dataloader):.2f}",
              flush=True)
        if valid_dataloader is not None:
            model.eval()
            accuracy = test(config=config, valid_dataloader=valid_dataloader, model=model, dev=dev, tokenizer=tokenizer)
            print(f"The dev bleu is: {accuracy}")
            if accuracy > best_accuracy:
                print(f"[Model Info] Saving the best model...")
                stop_num = 0
                torch.save(model.state_dict(), 'model.pth')
                best_accuracy = accuracy
                print(f"The best bleu is: {best_accuracy}, the early_stop_num is: {stop_num}")
            else:
                stop_num += 1
                print(f"The best bleu is: {best_accuracy}, the early_stop_num is: {stop_num}")
                if stop_num == early_stop:
                    print("Early Stop!")
                    break
    print(f"[Model Info] Returning the best model")
    modeldict = torch.load('model.pth')
    model.load_state_dict(modeldict)
    return model


def test(config: Config, valid_dataloader: DataLoader, model: nn.Module, dev: torch.device,
         tokenizer):
    model.eval()
    predictions_text = []
    targets_text = []
    predictions = []
    targets = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        for index, batch in tqdm(enumerate(valid_dataloader), desc="--validation", total=len(valid_dataloader)):
            generated_ids = model.module.generate(batch.input_ids.to(dev), batch.flag.to(dev))
            '''
            target_id = batch.label_id.to(dev)
            lm_labels = target_id.clone()
            mask = batch.attention_mask.to(dev)
            lm_labels[target_id == tokenizer.pad_token_id] = -100
            decoder_input_ids = shift_tokens_right(target_id, tokenizer.pad_token_id)
            print(generated_ids.size(), mask.size(), decoder_input_ids.size(), lm_labels.size())
            loss = model(generated_ids, attention_mask=mask, labels=lm_labels)
            eval_loss += loss.item()
            '''
            preds = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
                for g in generated_ids
            ]
            predictions_text.extend(preds)
            target = [tokenizer.decode(t[1:-1], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for t in
                      batch.question_ids]
            targets_text.extend(target)

    print("####PREDICTIIONS####")
    print(predictions_text[:20])
    print("####TARGETS####")
    print(targets_text[:20])
    
    preds = []
    golds = []
    m_score = 0.0
    
    rouge_score = 0.0
    rouge = Rouge()
    num = 0
    for pred, gold in zip(predictions_text, targets_text):
        #pred = tokenizer.tokenize(pred)
        #gold = tokenizer.tokenize(gold)
        #print(rouge.get_scores([pred], [gold]))
        rouge_score += rouge.get_scores([pred], [gold])[0]['rouge-l']['f']
        pred = nltk.word_tokenize(pred)
        gold = nltk.word_tokenize(gold)
        preds.append(pred)
        golds.append([gold])
        num += 1
        #print(gold)
        m_score += nltk.translate.meteor_score.meteor_score([gold], pred)
    BLEUscore = nltk.translate.bleu_score.corpus_bleu(golds, preds)
    '''
    print(nltk.translate.bleu_score.corpus_bleu(golds, preds, weights=(1,0,0,0)))
    print(nltk.translate.bleu_score.corpus_bleu(golds, preds, weights=(0.5,0.5,0,0)))
    print(nltk.translate.bleu_score.corpus_bleu(golds, preds, weights=(0.33,0.33,0.33,0)))
    print(nltk.translate.bleu_score.corpus_bleu(golds, preds, weights=(0.25,0.25,0.25,1)))
    print(m_score / num)
    print(rouge_score / num)
    #print(nltk.translate.meteor_score.meteor_score(golds[0], preds))
    '''
    
    file_data = []
    new_data = []
    for pred, gold in zip(predictions_text, targets_text):
        dict1 = {'pred': pred, 'gold': gold}
        pred1 = tokenizer.tokenize(pred)
        gold = tokenizer.tokenize(gold)
        s = nltk.translate.bleu_score.sentence_bleu([gold], pred1)
        file_data.append((pred, s))
        new_data.append(dict1)
    data_str = json.dumps(new_data, indent=4, ensure_ascii=False)
    with open("case.json", 'w') as f:
        f.write(data_str)
    #print(len(file_data))
    '''
    with open("pred/bert2tf.txt", "w") as f:
        for line in file_data:
            f.write(line[0] + "\n")
    '''
    
    '''
    file_data = []
    for pred, gold in zip(predictions_text, targets_text):
        pred1 = tokenizer.tokenize(pred)
        gold = tokenizer.tokenize(gold)
        s = nltk.translate.bleu_score.sentence_bleu([gold], pred1)
        file_data.append((pred, s))
    with open("pred.txt", "w") as f:
        for line in file_data:
            f.write(line[0] + "\n")
    with open("bleu.txt", "w") as f:
        for line in file_data:
            f.write(str(line[1]) + "\n")
    '''
    print(BLEUscore)
    return BLEUscore
    


def main():
    parser = argparse.ArgumentParser(description="Cloze Test question answering")
    opt = parse_arguments(parser)
    set_seed(opt)
    conf = Config(opt)
    if conf.bert_folder != "":
        bert_model_name = f'{conf.bert_folder}/{conf.bert_model_name}'
    else:
        bert_model_name = conf.bert_model_name
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Read dataset
    print("[Data Info] Reading training data", flush=True)
    dataset = FCDataset(tokenizer=tokenizer, file=conf.train_file, max_question_len=conf.max_seq_length,
                        max_answer_length=conf.generated_max_length, pretrain_model_name=bert_model_name,
                        number=conf.train_num, is_training=True)
    print("[Data Info]  Reading validation data", flush=True)
    eval_dataset = FCDataset(tokenizer=tokenizer, file=conf.dev_file, max_question_len=conf.max_seq_length,
                             max_answer_length=conf.generated_max_length, pretrain_model_name=bert_model_name,
                             number=conf.dev_num)

    test_dataset = FCDataset(tokenizer=tokenizer, file=conf.test_file, max_question_len=conf.max_seq_length,
                             max_answer_length=conf.generated_max_length, pretrain_model_name=bert_model_name,
                             number=conf.test_num)


    # Prepare data loader
    if opt.mode == "train":
        print("[Data Info] Loading training data", flush=True)
        train_dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=True,
                                      num_workers=conf.num_workers,
                                      collate_fn=dataset.collate_fn)
        print("[Data Info] Loading validation data", flush=True)
        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False,
                                      num_workers=conf.num_workers,
                                      collate_fn=eval_dataset.collate_fn)
        
        print("[Data Info] Loading test data", flush=True)
        test_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False,
                                      num_workers=conf.num_workers,
                                      collate_fn=eval_dataset.collate_fn)

        # Train the model
        model = train(conf, train_dataloader,
                      num_epochs=conf.num_epochs,
                      early_stop=conf.early_stop,
                      bert_model_name=bert_model_name,
                      valid_dataloader=valid_dataloader,
                      dev=conf.device,
                      tokenizer=tokenizer)

        device_ids = [0,1,2,3]
        if conf.gpus:
            model = nn.DataParallel(model, device_ids=device_ids, output_device=0)
        BLEU_score = test(conf, test_dataloader, model, conf.device, tokenizer)
        print(f"The test bleu is: {BLEU_score}")
    elif opt.mode == "test":
        print("[Data Info] Loading validation data", flush=True)
        valid_dataloader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False,
                                      num_workers=conf.num_workers,
                                      collate_fn=eval_dataset.collate_fn)
        print("[Model Info] Loading the saved model", flush=True)
        model = Bert2tf(conf, tokenizer)
        if conf.gpus:
            device_ids = [0, 1, 2, 3]
            model = nn.DataParallel(model, device_ids=device_ids, output_device=0)
        modeldict = torch.load(f'model.pth')
        model.load_state_dict(modeldict)
        model.to(conf.device)
        test(conf, valid_dataloader, model, conf.device, tokenizer)


if __name__ == "__main__":
    main()
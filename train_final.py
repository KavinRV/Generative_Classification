# import required libraries
import pandas as pd
import torch
from transformers import AutoTokenizer, T5Config
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
import argparse
import os
import wandb
from accelerate import Accelerator
import tqdm
import numpy as np
import namegenerator
import sys
import pickle
import random
from genret import T5ForConditionalGeneration
sys.path.append('./code')
from utills_final import AutocompleteDataset, merge_prefix_suffix, suffix_decoder
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

accelerator = Accelerator()
device = accelerator.device
print("PROCESS STARTED")





def get_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--bs',  type=int, default=4)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--tkmax_length', type=int, default=512)
    parser.add_argument('--mdmax_length', type=int, default=512)
    parser.add_argument('--initial_eval', action='store_true')
    parser.add_argument('--eval_every', type=int, default=3000)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--model_name', type=str, default="t5-base")
    parser.add_argument('--pred_type', type=str, default="mod")
    parser.add_argument('--da', action='store_true')
    args = parser.parse_args()
    return args

def main(args):
    #come with a new interseting name every time it is none
    if(args.model_dir is None and args.ckpt is None and accelerator.is_main_process):
        args.model_dir = args.model_name + "-" + namegenerator.gen()
        if(args.dev):
            args.model_dir += "-dev"
        os.makedirs(args.model_dir)
        # save args to model directory along with model name
        with open(os.path.join(args.model_dir, "args.txt"), "w") as f:
            f.write(args.model_name + "\n")
            f.write(str(args))
        
        print("The model directory is created!")
        print("Model Directory: ", args.model_dir)
    
    print("Using device:", device)

    # with open(args.train_data, "r") as f:
    #   data = f.read()
    # dataset = data.split("\n")
    # train_data = pd.DataFrame(dataset)
    # sentences = train_data.values.flatten().tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, tkmax_length=args.tkmax_length)

    seq_len = 18
    if args.pred_type == "mod":
        seq_len = 6
    elif args.pred_type == "br":
        seq_len = 14

    print("Tokenizing sentences...")
    dataset = AutocompleteDataset(data_path=args.train_data, tkmax_length=args.tkmax_length, tokenizer=tokenizer, pred_type=args.pred_type)
    print("total size of train dataset: ", len(dataset))
    dataloader = DataLoader(dataset, batch_size=args.bs, num_workers=4)

    # Load pre-trained model and tokenizer
    config = T5Config.from_pretrained(args.model_name)
    config.decoder_vocab_size = dataset.d_max + 1
    config.custom = True
    config.da = args.da
    config.alpha_grl = 1.0
    config.disc_labels = 2
    # model = T5ForConditionalGeneration.from_pretrained(args.model_name, config=config)
    model = T5ForConditionalGeneration(config=config)
    pretrain_model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    pretrain_params = dict(pretrain_model.named_parameters())
    for name, param in model.named_parameters():
        if name.startswith(("shared.", "encoder.")):
            with torch.no_grad():
                param.copy_(pretrain_params[name])
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # load ckpt if any
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        initial_epoch = ckpt["epoch"] + 1
        args.model_dir = os.path.dirname(args.ckpt)
        print("model loaded from checkpoint")
        print("model directory: ", args.model_dir)
    else:
        initial_epoch = 0
        print("pretrained model loaded -- no checkpoint found")
    
    # print("after loading")
    # print(model.shared.weight.shape)

    # def stop_gen(input_ids):
    #     if input_ids[-1] == 32100:
    #         return [1]
    #     else:
    #         return [i for i in range(32100, len(tokenizer))]
        # elif input_ids[-1] == 0:
        #     return [i for i in range(32105+5066)]
        # else:
        #     return [32100]

    if(args.wandb and accelerator.is_main_process):
        key = "cb9214905ae1b9737fed5614df1d085d1ddee3b2"
        wandb.login(key=key, relogin=True)
        wandb.init(project="T5-autocompletion", name=args.model_dir)
    else:
        wandb.init(project="T5-autocompletion", name=args.model_dir, mode="disabled")

    # log to wandb the model directory
    wandb.config.update(args)
    # setup dataloader and optimizer
    
    if args.val_data is not None:
        # with open(args.val_data, "r") as f:
        #   data = f.read()
        # dataset = data.split("\n")
        # val_data = pd.DataFrame(dataset)
        # val_sentences = val_data.values.flatten().tolist()
        print("Tokenizing validation sentences...")
        val_dataset = AutocompleteDataset(data_path=args.val_data, tkmax_length=args.tkmax_length, tokenizer=tokenizer, pred_type=args.pred_type)
        print("total size of validation dataset: ", len(val_dataset))
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs)
        
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # print("after optimizer")
    # print(model.shared.weight.shape)

    doc_dict_path = "doc_dict.pkl"

    if args.pred_type == "mod":
        doc_dict_path = "mod_out_dct.pkl"
    elif args.pred_type == "br":
        doc_dict_path = "br_out_dct.pkl"
    
    with open(doc_dict_path, "rb") as f:
        doc_dict = pickle.load(f)
    

    print("STARTING TRAINING")
    model, optimizer, dataloader, val_dataloader = accelerator.prepare(model, optimizer, dataloader, val_dataloader)

    # print("after accelerator")
    # print(model.shared.weight.shape)

    # if args.initial_eval:
    #     model.eval()
    #     total_val_loss = 0
    #     print("STARTING INITIAL EVAL")
    #     with torch.no_grad():
    #         for batch in tqdm.tqdm(val_dataloader):
    #             inputs, targets, seq = batch
    #             # Prepare data
    #             input_ids = inputs['input_ids'].squeeze(1)
    #             attention_mask = inputs['attention_mask'].squeeze(1)
    #             labels = targets['input_ids'].squeeze(1)
    #             seq_id = seq.squeeze(1)
    #             labels[labels == tokenizer.pad_token_id] = -100
    #             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #             loss = outputs.loss

    #             total_val_loss += loss.item()

    #     avg_val_loss = total_val_loss / len(val_dataloader)

    #     print(f"Initial Evaluation, Validation Loss: {avg_val_loss}")
    #     # also log perplexity
    #     wandb.log({
    #         "avg_val_loss": avg_val_loss,
    #         "perplexity_val_set": torch.exp(torch.tensor(avg_val_loss))
    #     })

        
    for epoch in tqdm.tqdm(range(initial_epoch, args.num_epochs)):
            # Training phase
            model.train()
            # print("after train load")
            # print(model.shared.weight.shape)
            total_train_loss = 0
            total_da_loss = 0
            latest_val_loss = 0
            iterations = 0
            for batch in tqdm.tqdm(dataloader):
                inputs, targets = batch
                # Prepare data
                input_ids = inputs['input_ids'].squeeze(1)
                attention_mask = inputs['attention_mask'].squeeze(1)
                # labels = targets['input_ids'].squeeze(1)
                labels = targets.squeeze(1)
                labels[labels == tokenizer.pad_token_id] = -100
                optimizer.zero_grad()
                # if args.da:
                #     # outputs, da_loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, da_labels=domain)
                #     loss = outputs.loss
                #     total_loss = loss + 0.2*da_loss
                # else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss = loss
                da_loss = torch.tensor(0.0)
                
                # if iterations > 310:
                    # print("-*_*-"*1000)
                    # print("loss: ", total_loss)
                accelerator.backward(total_loss)
                optimizer.step()
                rand_prob = random.random()
                if rand_prob < 0.2:
                    # Decode and print input text
                    input_text = [tokenizer.decode(input_ids[0], skip_special_tokens=True)]
                    print(f"Input Text (Epoch: {epoch}, Iteration {iterations}): {input_text}")
                    
                    # Generate model output and decode
                    with torch.no_grad():
                        # see if model has attribute modules
                        if hasattr(model, "module"):
                            model_output = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=seq_len+1, min_length=seq_len-3, prefix_allowed_tokens_fn = lambda batch_id, input_ids: dataset.sub_get(input_ids.tolist()))
                        else:
                            model_output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=seq_len+1, min_length=seq_len-3, prefix_allowed_tokens_fn = lambda batch_id, input_ids: dataset.sub_get(input_ids.tolist()))
                        output_text = model_output[0].detach().clone().cpu().tolist()
                        if len(output_text[1:]) < seq_len:
                            output_text = output_text[1:] + [tokenizer.pad_token_id] * (seq_len - len(output_text[1:]))
                        else:
                            output_text = output_text[1:]
                        
                        pred_doc = dataset.code_doc[str(output_text)]
                        prediction = doc_dict[pred_doc]

                        gt_enc = labels[0].detach().clone().cpu()
                        gt_enc[gt_enc == -100] = tokenizer.pad_token_id
                        gt_enc = gt_enc.tolist()

                        if sum(gt_enc) != 0:
                            gt_doc = dataset.code_doc[str(gt_enc)]
                            gt_text = doc_dict[gt_doc]
                        else:
                            gt_text = " "
                        # gt_text = suffix_decoder(tokenizer, gt_enc)
                        
                        print(f"Model Output (Epoch: {epoch}, Iteration {iterations}): {prediction}, Ground Truth: {gt_text}")
                    
                    wandb.log({
                        "training loss": loss.item(),
                        })
                
                total_train_loss += loss.item()
                total_da_loss += da_loss.item()
                # Evaluation phase
                if(args.val_data is not None and (iterations+1)%args.eval_every == 0):
                    model.eval()
                    total_val_loss = 0
                    print("STARTING VALIDATION")
                    with torch.no_grad():
                        for batch in tqdm.tqdm(val_dataloader):
                            inputs, targets = batch
                            # Prepare data
                            input_ids = inputs['input_ids'].squeeze(1)
                            attention_mask = inputs['attention_mask'].squeeze(1)
                            labels = targets.squeeze(1)
                            # seq_id = seq.squeeze(1)
                            # labels[labels == tokenizer.pad_token_id] = -100
                            if hasattr(model, "module"):
                                outputs = model.module.generate(input_ids=input_ids,
                                                        attention_mask=attention_mask,
                                                        # max_length=seq_len+1, 
                                                        # min_length=seq_len-1,
                                                        prefix_allowed_tokens_fn = lambda batch_id, input_ids: dataset.sub_get(input_ids.tolist())
                                                        #  prefix_allowed_tokens_fn = lambda batch_id, input_ids: stop_gen(input_ids.tolist())
                                                        )
                            else:
                                outputs = model.generate(input_ids=input_ids, 
                                                        attention_mask=attention_mask,
                                                        # max_length=19, 
                                                        # min_length=17,
                                                        prefix_allowed_tokens_fn = lambda batch_id, input_ids: dataset.sub_get(input_ids.tolist())
                                                        )

                            # outputs = model(
                            #     input_ids=input_ids,
                            #     attention_mask=attention_mask,
                            #     labels=labels,
                            # )
                            # loss = outputs.loss
                            outputs = outputs[:, 1:]

                            if labels.shape[1] > outputs.shape[1]:
                                # extend outputs with pad tokens
                                pad = torch.full((outputs.shape[0], labels.shape[1] - outputs.shape[1]), tokenizer.pad_token_id, dtype=torch.long).to(device)
                                outputs = torch.cat((outputs, pad), dim=1)
                            elif labels.shape[1] < outputs.shape[1]:
                                pad = torch.full((labels.shape[0], outputs.shape[1] - labels.shape[1]), tokenizer.pad_token_id, dtype=torch.long).to(device)
                                labels = torch.cat((labels, pad), dim=1)
                            
                            checker = outputs == labels
                            checker = checker.all(dim=1)
                            loss = checker.float().sum()

                            total_val_loss += loss.item()

                    avg_val_loss = total_val_loss / len(val_dataset)

                    print(f"Epoch: {epoch}, Iteration: {iterations}, Training Loss: {loss.item()}, Validation Loss: {avg_val_loss}")
                    # also log perplexity
                    wandb.log({
                        "avg_val_acc": avg_val_loss,
                    })
                    
                    model.train()

                iterations = iterations + 1

            avg_train_loss = total_train_loss / len(dataloader)
            avg_da_loss = total_da_loss / len(dataloader)

            if hasattr(model, "module"):
                _model = accelerator.unwrap_model(model.module)
                # _optimizer = optimizer.module
            else:
                _model = model
                # _optimizer = optimizer

            print(f"Epoch: {epoch}, Training Loss: {avg_train_loss}, DA Loss: {avg_da_loss}")
            wandb.log({
                "avg_train_loss": avg_train_loss,
                "avg_da_loss": avg_da_loss
                })
            # Save model checkpoint at the end of each epoch only if is the main process
            if(accelerator.is_main_process):
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': _model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_train_loss
                    }
                    checkpoint_path = os.path.join(args.model_dir, f'epoch_{epoch}.pth')
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Checkpoint saved at '{checkpoint_path}'")

if __name__ == "__main__":
    main(get_args())
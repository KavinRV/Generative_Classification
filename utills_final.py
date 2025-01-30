import torch
from torch.utils.data import Dataset
import tqdm
import numpy as np
import argparse
import pandas as pd
import pickle
import torch
import random


def suffix_encoder(tokenizer, text, max_length, batching = False, prev_space = True):
    encoded = tokenizer(text, padding="max_length", max_length=88, truncation=True, return_tensors='pt')
    return encoded



def suffix_decoder(tokenizer, encoded):
    text = tokenizer.decode(encoded, skip_special_tokens=True)
    # if(len(text)==0):
    #     print("Empty text")
    #     print(encoded)
    return text


def prefix_encoder(tokenizer, text, max_length=64, batch = False):
    encoded = tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors='pt')
    # encoded = tokenizer(text, return_tensors='pt')
    return encoded

def merge_prefix_suffix(prefix, suffix):
    if(len(suffix)>0 and len(prefix)>0 and suffix[0] == " " and prefix[-1] == " "):
        return prefix[:-1] + suffix
    else:
        return prefix + suffix



class Node(object):
    def __init__(self, token_id) -> None:
        self.token_id = token_id
        self.children = {}

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.token_id) + "\n"
        for child in self.children.values():
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node representation>'
    
class TreeBuilder(object):
    def __init__(self) -> None:
        self.root = Node(0)

    def build(self) -> Node:
        return self.root

    def add(self, seq) -> None:
        cur = self.root
        for tok in seq:
            if tok == 0:  # reach pad_token
                return
            # print(seq)
            if tok not in cur.children:
                cur.children[tok] = Node(tok)
            cur = cur.children[tok]

class BrandBuilder(object):
    def __init__(self) -> None:
        self.root = Node(0)

    def build(self) -> Node:
        return self.root

    def add(self, seq) -> None:
        cur = self.root
        for tok in seq:
            if tok == 32100:  # reach " <|SEP|> "
                if tok not in cur.children:
                    cur.children[tok] = Node(tok)
                return
            # print(seq)
            if tok not in cur.children:
                cur.children[tok] = Node(tok)
            cur = cur.children[tok]


class AutocompleteDataset(Dataset):
    def __init__(self,
        data_path,
        tokenizer, 
        tkmax_length=512,
        infer=False,
        pred_type="full", 
        domain=False
        ):
        self.tokenizer = tokenizer
        self.pred_type = pred_type
        # preprocessing

        doc_code_path = "doc_code2.pkl"
        num_lab = 64
        tree_path = "tree.pkl"
        code_doc_path = "code_doc.pkl"
        type_name = "doc"
        seq_len = 18

        if pred_type == "mod":
            doc_code_path = "mod_code.pkl"
            num_lab = 6
            tree_path = "mod_tree.pkl"
            code_doc_path = "mod_code_doc.pkl"
            type_name = "module"
            data_path = f"_{data_path}"
            seq_len = 6
        
        if pred_type == "br":
            doc_code_path = "br_code.pkl"
            num_lab = 48
            tree_path = "br_tree.pkl"
            code_doc_path = "br_code_doc.pkl"
            type_name = "brand"
            data_path = f"_{data_path}"
            seq_len = 14

        self.data = pd.read_csv(data_path)
        self.infer = infer
        self.d_max = 0

        with open(doc_code_path, "rb") as f:
            self.doc_map = pickle.load(f)
        
        self.doc_code_new = {}
        self.v_lst = {}
        self.doc_full = {}
        self.max_val = 0

        if not infer:
            for new_k, v in self.doc_map.items():
                # idx = k.index(' retailers: ')
                # new_k = k[:idx]
                if str(v.tolist()) not in self.v_lst:
                    self.v_lst[str(v.tolist())] = 1
                else:
                    self.v_lst[str(v.tolist())] += 1
                    # print("Duplicate code found")

                v = v.tolist()
                new_v = []
                dst = 0
                for val in v:
                    new_val = num_lab*dst + val + 2
                    dst += 1
                    new_v.append(new_val)
                    if new_val > self.d_max:
                        self.d_max = new_val
                        # if new_val > 51:
                        #     print(dst, val, num_lab)
                    
                    if val > self.max_val:
                        self.max_val = val

                if self.v_lst[str(v)] > 1:
                    end_val = num_lab*dst + 2 + self.v_lst[str(v)] - 2
                    if end_val > self.d_max:
                        self.d_max = end_val
                    new_v = new_v + [end_val, 1]
                else:
                    new_v = new_v + [tokenizer.eos_token_id]
                    new_v += [tokenizer.pad_token_id]*(seq_len - len(new_v))
                    # new_v = v.tolist() + [v_lst[str(v.tolist())]]
                
                
                self.doc_code_new[new_k] = torch.tensor(new_v)

            self.tree = TreeBuilder()
            
            for v in self.doc_code_new.values():
                self.tree.add(v.tolist())
            
            self.root = self.tree.build()
            # save the tree
            print("Saving the tree")
            with open(tree_path, "wb") as f:
                pickle.dump(self.tree, f)

            print("Saving the doc_code")
            
            self.code_doc = {str(v.tolist()): k for k, v in self.doc_code_new.items()}
            with open(code_doc_path, "wb") as f:
                pickle.dump(self.code_doc, f)
            
            print("Tree and doc_code saved")

            #remove all the rows in self.data where doc is " "
            # if not domain and self.pred_type != "simp":
            #     self.data = self.data[self.data[type_name] != " "]

            dct_mb = {"module": 15, "brand": 5}
            mb_bm = {"module": "brand", "brand": "module"}

            self.data = self.data.drop(columns=[mb_bm[type_name]])

            doc_as_query = {"query": [], type_name: []}
            for k, v in self.doc_code_new.items():
                doc_as_query["query"] += [k]*dct_mb[type_name]
                doc_as_query[type_name] += [k]*dct_mb[type_name]
            
            self.data = self.data.append(pd.DataFrame(doc_as_query), ignore_index=True)
            self.data = self.data.sample(frac=1).reset_index(drop=True)

            self.data["code"] = self.data[type_name].apply(lambda x: self.doc_code_new[x] if x != " " else torch.tensor([tokenizer.pad_token_id]*seq_len))
            # self.data["code"] = self.data["doc"].apply(lambda x: self.doc_code_new[x] if x != " " else torch.tensor([77] + [77]*16+ [1]))
        self.max_length = tkmax_length
        # self.infer = infer
    
    def get_children(self, input_ids, trie_dict):

        if len(input_ids) == 0:
            output = list(trie_dict.children.keys())
            if len(output) == 0:
                return [0]
            return output
        elif input_ids[0] in trie_dict.children:
            return self.get_children(input_ids[1:], trie_dict.children[input_ids[0]])
        else:
            return [0] 
    
    def sub_get(self, input_ids):
        input_ids_new = input_ids[1:]
        out =  self.get_children(input_ids_new, self.root)
        # print(input_ids[1:], "out", out)
        return out
            
    def __len__(self):
        # return 100
        return len(self.data)

    def __getitem__(self, idx):
        curr_eg = self.data.iloc[idx]
        # if self.infer == True :
        #     # i think both can be of same format during inference
        #     input_text = curr_sentence.split("\t")[0]
        #     target_text = curr_sentence.split("\t")[1]
        #     return input_text, target_text

        if self.infer:
            if self.pred_type == "com":
                input_1 = "find module for " + curr_eg["query"]
                input_2 = "find brand for " + curr_eg["query"]
                input_encoded_1 = prefix_encoder(self.tokenizer, input_1, max_length=32)
                input_encoded_2 = prefix_encoder(self.tokenizer, input_2, max_length=32)
                return input_encoded_1, input_encoded_2, curr_eg["indoml_id"]
            
            input_text = curr_eg["query"]
            input_encoded = prefix_encoder(self.tokenizer, input_text, max_length=32)
            
            return input_encoded, curr_eg["indoml_id"]
        # 
        # rand_prob = random.random()
        input_text = curr_eg["query"]

        # if "nan"==str(curr_eg["indoml_id"]):
        #     input_text = curr_eg["description"]
        # else:
        #     # print(curr_eg["indoml_id"])
        #     # print(np.nan)
        #     # print("nan"==str(curr_eg["indoml_id"]))
        #     input_text = "description: " + curr_eg["description"] + " retailer: " + curr_eg["retailer"]
            
        # if rand_prob < 0.01:
        #     input_text = curr_eg["full_doc"]

        # if(self.context):
        #     context_sentence = ' '.join(curr_sentence.split('\t')[:-1])
        #     breaking_sentence = curr_sentence.split('\t')[-1]
        #     r = np.random.randint(1, len(breaking_sentence))
        #     input_text = context_sentence + breaking_sentence[:r]
        #     target_text = breaking_sentence[r:]


        # print("input = ", [input_text])
        # print("target = ", [target_text])

        input_encoded = prefix_encoder(self.tokenizer, input_text, max_length=28)
        
        # if self.infer:
        #     return input_encoded, curr_eg["indoml_id"]
        
        target_encoded = curr_eg["code"]
        # tg_idx = target_text.index("<|SEP|>")
        # spl_tok = target_text[:tg_idx-1]
        # spl_tok = self.tokenizer(spl_tok, return_tensors='pt')['input_ids'][:, 0:1]
        # target_encoded = suffix_encoder(self.tokenizer, target_text, max_length=self.max_length, prev_space = (input_text[-1]==" "))
        # print("input = ", [input_text])
        # print("target = ", [target_text])
        # print("target decoded = ", [suffix_decoder(self.tokenizer, target_encoded['input_ids'][0])])
        return input_encoded, target_encoded
    
    def preprocess_text(self, text):
        text = text.strip().lower()
        text = text.replace("<eou>", "<|EOU|>")
        return text
""" 
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
Utility fuctions 
"""

import argparse
import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='Path to evaluation dataset. i.e. implicitHate.json or toxiGen.json')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to result text file')
    parser.add_argument('--model', type=str, required=True,
                        help="a local path to a model or a model tag on HuggignFace hub.")
    parser.add_argument('--lmHead', type=str, required=True,
                        choices=['mlm', 'clm'])
    parser.add_argument('--config', type=str,
                        help='Path to model config file')
    parser.add_argument("--force", action="store_true", 
                        help="Overwrite output path if it already exists.")
    args = parser.parse_args()

    return args


def load_tokenizer_and_model(args, from_tf=False):
    '''
    Load tokenizer and model to evaluate.
    '''
    pretrained_model = args.model
    pretrained_adapter = ''
    if 'bactrian' in pretrained_model:
        pretrained_model, pretrained_adapter = pretrained_model.split('---')

    if args.config:
        config = AutoConfig.from_pretrained(args.config)
    else:
        config = None
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, truncation_side='left', padding_side='left', trust_remote_code=True)

    # Load Masked Language Model Head
    if args.lmHead == 'mlm':
        model = AutoModelForMaskedLM.from_pretrained(pretrained_model, from_tf=from_tf, config=config, trust_remote_code=True)
    # load Causal Language Model Head
    else:
        if pretrained_adapter != "":
            model = AutoModelForCausalLM.from_pretrained(pretrained_model, device_map="auto", load_in_8bit=True, trust_remote_code=True)
            model = PeftModel.from_pretrained(model, pretrained_adapter, torch_dtype=torch.float16)
        elif "bloom" in pretrained_model or "xglm" in pretrained_model or "gpt2" in pretrained_model or "sealion7b" in pretrained_model \
            or "Merak" in pretrained_model or "SeaLLM" in pretrained_model or  "Llama" in pretrained_model:
            model = AutoModelForCausalLM.from_pretrained(pretrained_model, device_map="auto", load_in_8bit=True, trust_remote_code=True)
            if "sealion7b" in pretrained_model or  "Llama" in pretrained_model:
                tokenizer.pad_token = tokenizer.eos_token # Use EOS to pad label
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model, device_map="auto", load_in_8bit=True, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token # Use EOS to pad label
    
    model = model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    return tokenizer, model

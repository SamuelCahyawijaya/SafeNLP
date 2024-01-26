# Sample Scripts
# # Calculate safety score for GPT2-small on ToxiGen human annotated dataset
# python safety_score.py --data data/toxiGen.json --output results --model gpt2 --lmHead clm 
#
# # Calculate safety score for BERT-base-uncased on ToxiGen human annotated dataset
# python safety_score.py --data data/toxiGen.json --output results --model bert-base-uncased --lmHead mlm 
#
# # Calculate safety score for GPT2-small on implicitHate dataset
# python safety_score.py --data data/implicitHate.json --output results --model gpt2 --lmHead clm 

###
# Implicit Hate Ind
###

# mT0
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  bigscience/mt0-small
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  bigscience/mt0-base
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  bigscience/mt0-large
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  bigscience/mt0-xl
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  bigscience/mt0-xxl

# BLOOMZ
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  bigscience/bloomz-560m
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  bigscience/bloomz-1b1
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  bigscience/bloomz-1b7
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  bigscience/bloomz-3b
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  bigscience/bloomz-7b1

# LLaMA2
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  meta-llama/Llama-2-7b-chat-hf
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  meta-llama/Llama-2-13b-chat-hf

# Bactrian
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  bigscience/bloom-7b1---MBZUAI/bactrian-x-bloom-7b1-lora
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  bigscience/bloom-7b1---haonan-li/bactrian-id-bloom-7b1-lora

# SEALion & SEALLM
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  aisingapore/sealion7b-instruct-nc
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  SeaLLMs/SeaLLM-7B-Chat

# Merak
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  Ichsan2895/Merak-7B-v4

# Cendol-Instruct
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-small
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-base
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-large
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-xl
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-xxl-merged
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-llama2-7b
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-llama2-13b-merged

# Cendol-Chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-small-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-base-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-large-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-xl-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-xxl-merged-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-llama2-7b-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-llama2-13b-merged-chat

###
# Toxigen Ind
###

# mT0
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  bigscience/mt0-small
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  bigscience/mt0-base
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  bigscience/mt0-large
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  bigscience/mt0-xl
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  bigscience/mt0-xxl

# BLOOMZ
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  bigscience/bloomz-560m
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  bigscience/bloomz-1b1
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  bigscience/bloomz-1b7
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  bigscience/bloomz-3b
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  bigscience/bloomz-7b1

# LLaMA2
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  meta-llama/Llama-2-7b-chat-hf
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  meta-llama/Llama-2-13b-chat-hf

# Bactrian
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  bigscience/bloom-7b1---MBZUAI/bactrian-x-bloom-7b1-lora
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  bigscience/bloom-7b1---haonan-li/bactrian-id-bloom-7b1-lora

# SEALion & SEALLM
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  aisingapore/sealion7b-instruct-nc
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  SeaLLMs/SeaLLM-7B-Chat

# Merak
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  Ichsan2895/Merak-7B-v4

# Cendol-Instruct
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-small
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-base
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-large
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-xl
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-xxl-merged
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  indonlp/cendol-llama2-7b
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output results --lmHead clm  --model  indonlp/cendol-llama2-7b

# Cendol-Chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-small-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-base-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-large-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-xl-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-mt5-xxl-merged-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-llama2-7b-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output results --lmHead clm  --model  indonlp/cendol-llama2-13b-merged-chat
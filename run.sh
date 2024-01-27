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

# # mT0
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_mt0_small --lmHead clm  --model  bigscience/mt0-small
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_mt0_base --lmHead clm  --model  bigscience/mt0-base
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_mt0_large --lmHead clm  --model  bigscience/mt0-large
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_mt0_xl --lmHead clm  --model  bigscience/mt0-xl
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_mt0_xxl --lmHead clm  --model  bigscience/mt0-xxl

# # BLOOMZ
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_bloomz_560m --lmHead clm  --model  bigscience/bloomz-560m
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_bloomz_1b1 --lmHead clm  --model  bigscience/bloomz-1b1
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_bloomz_1b7 --lmHead clm  --model  bigscience/bloomz-1b7
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_bloomz_3b --lmHead clm  --model  bigscience/bloomz-3b
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_bloomz_7b1 --lmHead clm  --model  bigscience/bloomz-7b1

# # LLaMA2
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_llama2_7b_chat --lmHead clm  --model  meta-llama/Llama-2-7b-chat-hf
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_llama2_13b_chat --lmHead clm  --model  meta-llama/Llama-2-13b-chat-hf

# # SEALion & SEALLM
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_sealion_7b_instruct --lmHead clm  --model  aisingapore/sealion7b-instruct-nc
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_seallm_7b_chat --lmHead clm  --model  SeaLLMs/SeaLLM-7B-Chat

# Bactrian
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_bactrian_x --lmHead clm  --model  bigscience/bloom-7b1---MBZUAI/bactrian-x-bloom-7b1-lora
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_bactrian_id --lmHead clm  --model  bigscience/bloom-7b1---haonan-li/bactrian-id-bloom-7b1-lora

# Merak
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_merak --lmHead clm  --model  Ichsan2895/Merak-7B-v4

# Cendol-Instruct
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_cendol_mt5_small --lmHead clm  --model  indonlp/cendol-mt5-small
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_cendol_mt5_base --lmHead clm  --model  indonlp/cendol-mt5-base
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_cendol_mt5_large --lmHead clm  --model  indonlp/cendol-mt5-large
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_cendol_mt5_xl --lmHead clm  --model  indonlp/cendol-mt5-xl
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_cendol_mt5_xxl --lmHead clm  --model  indonlp/cendol-mt5-xxl-merged
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_cendol_llama2_7b --lmHead clm  --model  indonlp/cendol-llama2-7b
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_cendol_llama2_13b --lmHead clm  --model  indonlp/cendol-llama2-13b-merged

# Cendol-Chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_cendol_mt5_small_chat --lmHead clm  --model  indonlp/cendol-mt5-small-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_cendol_mt5_base_chat --lmHead clm  --model  indonlp/cendol-mt5-base-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_cendol_mt5_large_chat --lmHead clm  --model  indonlp/cendol-mt5-large-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_cendol_mt5_xl_chat --lmHead clm  --model  indonlp/cendol-mt5-xl-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_cendol_mt5_xxl_chat --lmHead clm  --model  indonlp/cendol-mt5-xxl-merged-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_cendol_llama2_7b_chat --lmHead clm  --model  indonlp/cendol-llama2-7b-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/implicit_hate_ind.json --output implicit_hate/results_cendol_llama2_13b_chat --lmHead clm  --model  indonlp/cendol-llama2-13b-merged-chat

###
# Toxigen Ind
###

# # mT0
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_mt0_small --lmHead clm  --model  bigscience/mt0-small
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_mt0_base --lmHead clm  --model  bigscience/mt0-base
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_mt0_large --lmHead clm  --model  bigscience/mt0-large
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_mt0_xl --lmHead clm  --model  bigscience/mt0-xl
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_mt0_xxl --lmHead clm  --model  bigscience/mt0-xxl

# # BLOOMZ
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_bloomz_560m --lmHead clm  --model  bigscience/bloomz-560m
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_bloomz_1b1 --lmHead clm  --model  bigscience/bloomz-1b1
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_bloomz_1b7 --lmHead clm  --model  bigscience/bloomz-1b7
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_bloomz_3b --lmHead clm  --model  bigscience/bloomz-3b
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_bloomz_7b1 --lmHead clm  --model  bigscience/bloomz-7b1

# # LLaMA2
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_llama2_7b_chat --lmHead clm  --model  meta-llama/Llama-2-7b-chat-hf
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_llama2_13b_chat --lmHead clm  --model  meta-llama/Llama-2-13b-chat-hf

# # SEALion & SEALLM
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_sealion_7b_instruct --lmHead clm  --model  aisingapore/sealion7b-instruct-nc
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_seallm_7b_chat --lmHead clm  --model  SeaLLMs/SeaLLM-7B-Chat

# Bactrian
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_bactrian_x --lmHead clm  --model  bigscience/bloom-7b1---MBZUAI/bactrian-x-bloom-7b1-lora
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_bactrian_id --lmHead clm  --model  bigscience/bloom-7b1---haonan-li/bactrian-id-bloom-7b1-lora

# # Merak
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_merak --lmHead clm  --model  Ichsan2895/Merak-7B-v4

# # Cendol-Instruct
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_cendol_mt5_small --lmHead clm  --model  indonlp/cendol-mt5-small
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_cendol_mt5_base --lmHead clm  --model  indonlp/cendol-mt5-base
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_cendol_mt5_large --lmHead clm  --model  indonlp/cendol-mt5-large
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_cendol_mt5_xl --lmHead clm  --model  indonlp/cendol-mt5-xl
# CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_cendol_mt5_xxl --lmHead clm  --model  indonlp/cendol-mt5-xxl-merged
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_cendol_llama2_7b --lmHead clm  --model  indonlp/cendol-llama2-7b
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_cendol_llama2_13b --lmHead clm  --model  indonlp/cendol-llama2-7b

# # Cendol-Chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_cendol_mt5_small_chat --lmHead clm  --model  indonlp/cendol-mt5-small-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_cendol_mt5_base_chat --lmHead clm  --model  indonlp/cendol-mt5-base-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_cendol_mt5_large_chat --lmHead clm  --model  indonlp/cendol-mt5-large-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_cendol_mt5_xl_chat --lmHead clm  --model  indonlp/cendol-mt5-xl-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_cendol_mt5_xxl_chat --lmHead clm  --model  indonlp/cendol-mt5-xxl-merged-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_cendol_llama2_7b_chat --lmHead clm  --model  indonlp/cendol-llama2-7b-chat
CUDA_VISIBLE_DEVICES=1 python safety_score.py --data data/toxigen_ind.json --output toxigen/results_cendol_llama2_13b_chat --lmHead clm  --model  indonlp/cendol-llama2-13b-merged-chat
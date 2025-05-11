# DomainNet
python3 -u main.py  -stage1_epochs 0 -generator_layer 2 -debug_mode 0  -es 2 -e 1 -data DomainNet -hd painting -sd infograph -bs 50 -log 100 -lr 0.0001 -use_NTP 0 -tp_N_CTX 16 -GP_CLS_NUM_TOKENS 1 -GP_DOM_NUM_TOKENS 1
python3 -u main.py  -stage1_epochs 0 -generator_layer 2 -debug_mode 0  -es 2 -e 1 -data DomainNet -hd quickdraw -sd sketch -bs 50 -log 100 -lr 0.0001 -use_NTP 0 -tp_N_CTX 16 -GP_CLS_NUM_TOKENS 1 -GP_DOM_NUM_TOKENS 1
python3 -u main.py  -stage1_epochs 0 -generator_layer 2 -debug_mode 0  -es 2 -e 1 -data DomainNet -hd sketch -sd quickdraw -bs 50 -log 100 -lr 0.0001 -use_NTP 0 -tp_N_CTX 16 -GP_CLS_NUM_TOKENS 1 -GP_DOM_NUM_TOKENS 1
python3 -u main.py  -stage1_epochs 0 -generator_layer 2 -debug_mode 0  -es 2 -e 1 -data DomainNet -hd clipart -sd painting -bs 50 -log 100 -lr 0.0001 -use_NTP 0 -tp_N_CTX 16 -GP_CLS_NUM_TOKENS 1 -GP_DOM_NUM_TOKENS 1
python3 -u main.py  -stage1_epochs 0 -generator_layer 2 -debug_mode 0  -es 2 -e 1 -data DomainNet -hd infograph -sd painting  -bs 50 -log 100 -lr 0.0001 -use_NTP 0 -tp_N_CTX 16 -GP_CLS_NUM_TOKENS 1 -GP_DOM_NUM_TOKENS 1
#
## Sketchy
#python3 -u main.py -stage1_epochs 0 -data Sketchy -bs 50 -log 100 -lr 0.0001 -use_NTP 0 -tp_N_CTX 16 -GP_CLS_NUM_TOKENS 1 -GP_DOM_NUM_TOKENS 1 -debug_mode 0
#
## # TUBerlin
#python3 -u main.py -stage1_epochs 0 -data TUBerlin -bs 50 -log 15 -lr 0.0001 -use_NTP 0 -tp_N_CTX 16 -GP_CLS_NUM_TOKENS 1 -GP_DOM_NUM_TOKENS 1 -debug_mode 0
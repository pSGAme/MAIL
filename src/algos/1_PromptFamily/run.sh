#MaPLe
python3 -u main.py  -data DomainNet -e 1 -hd sketch -sd quickdraw   --maple 1 -bs 60 --log_name maple --num_shots 2
#python3 -u main.py  -data DomainNet -e 1 -hd quickdraw -sd sketch   --maple 1 -bs 60 --log_name maple --num_shots 2
#python3 -u main.py  -data DomainNet -e 1 -hd painting -sd infograph --maple 1 -bs 60 --log_name maple --num_shots 2
#python3 -u main.py  -data DomainNet -e 1 -hd infograph -sd painting --maple 1 -bs 60 --log_name maple --num_shots 2
#python3 -u main.py  -data DomainNet -e 1 -hd clipart -sd painting   --maple 1 -bs 60 --log_name maple --num_shots 2
#
#python3 -u main.py -data  Sketchy  -e 20 -bs 24 --maple 1  --log_name maple --num_shots 2
#python3 -u main.py -data  TUBerlin -e 20 -bs 24 --maple 1  --log_name maple --num_shots 2
#
##IVLP
#python3 -u main.py  -data DomainNet -hd sketch -sd quickdraw   -textNumTokens 2 -visualNumTokens 2  -bs 60 --log_name ivlp --num_shots 2
#python3 -u main.py  -data DomainNet -hd quickdraw -sd sketch   -textNumTokens 2 -visualNumTokens 2  -bs 60 --log_name ivlp --num_shots 2
#python3 -u main.py  -data DomainNet -hd painting -sd infograph -textNumTokens 2 -visualNumTokens 2  -bs 60 --log_name ivlp --num_shots 2
#python3 -u main.py  -data DomainNet -hd infograph -sd painting -textNumTokens 2 -visualNumTokens 2  -bs 60 --log_name ivlp --num_shots 2
#python3 -u main.py  -data DomainNet -hd clipart -sd painting   -textNumTokens 2 -visualNumTokens 2  -bs 60 --log_name ivlp --num_shots 2
#
#python3 -u main.py -data  Sketchy  -e 20 -bs 24 -textNumTokens 2 -visualNumTokens 2  --log_name maple --num_shots 2
#python3 -u main.py -data  TUBerlin -e 20 -bs 24 -textNumTokens 2 -visualNumTokens 2  --log_name maple --num_shots 2
#
## VPT
#python3 -u main.py  -data DomainNet -hd sketch -sd quickdraw  -visualDepth 1 -textNumTokens 0 -visualNumTokens 4  -bs 60 --log_name vpt --num_shots 2
#python3 -u main.py  -data DomainNet -hd quickdraw -sd sketch  -visualDepth 1 -textNumTokens 0 -visualNumTokens 4  -bs 60 --log_name vpt --num_shots 2
#python3 -u main.py  -data DomainNet -hd painting -sd infograph -visualDepth 1  -textNumTokens 0 -visualNumTokens 4  -bs 60 --log_name vpt --num_shots 2
#python3 -u main.py  -data DomainNet -hd infograph -sd painting  -visualDepth 1 -textNumTokens 0 -visualNumTokens 4  -bs 60 --log_name vpt --num_shots 2
#python3 -u main.py  -data DomainNet -hd clipart -sd painting  -visualDepth 1 -textNumTokens 0 -visualNumTokens 4 -bs 60 --log_name vpt --num_shots 2
#
#python3 -u main.py -data  Sketchy  -e 20 -bs 24 -visualDepth 1 -textNumTokens 0 -visualNumTokens 4  --log_name maple --num_shots 2
#python3 -u main.py -data  TUBerlin -e 20 -bs 24 -visualDepth 1 -textNumTokens 0 -visualNumTokens 4  --log_name maple --num_shots 2
#
#
##VPT-Deep
#python3 -u main.py  -data DomainNet -hd sketch -sd quickdraw  -visualDepth 12 -textNumTokens 0 -visualNumTokens 4  -bs 60 --log_name vptd --num_shots 2
#python3 -u main.py  -data DomainNet -hd quickdraw -sd sketch  -visualDepth 12 -textNumTokens 0 -visualNumTokens 4  -bs 60 --log_name vptd --num_shots 2
#python3 -u main.py  -data DomainNet -hd painting -sd infograph -visualDepth 12  -textNumTokens 0 -visualNumTokens 4  -bs 60 --log_name vptd --num_shots 2
#python3 -u main.py  -data DomainNet -hd infograph -sd painting  -visualDepth 12 -textNumTokens 0 -visualNumTokens 4  -bs 60 --log_name vptd --num_shots 2
#python3 -u main.py  -data DomainNet -hd clipart -sd painting  -visualDepth 12 -textNumTokens 0 -visualNumTokens 4 -bs 60 --log_name vptd --num_shots 2
#
#python3 -u main.py -data  Sketchy  -e 20 -bs 24 -visualDepth 12 -textNumTokens 0 -visualNumTokens 4  --log_name maple --num_shots 2
#python3 -u main.py -data  TUBerlin -e 20 -bs 24 -visualDepth 12 -textNumTokens 0 -visualNumTokens 4  --log_name maple --num_shots 2
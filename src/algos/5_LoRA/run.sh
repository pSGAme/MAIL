python3 -u main.py  -data DomainNet -hd sketch -sd quickdraw    --log_name  lora --num_shots 2
python3 -u main.py  -data DomainNet -hd quickdraw -sd sketch    --log_name  lora --num_shots 2
python3 -u main.py  -data DomainNet -hd painting -sd infograph  --log_name  lora --num_shots 2
python3 -u main.py  -data DomainNet -hd infograph -sd painting  --log_name  lora --num_shots 2
python3 -u main.py  -data DomainNet -hd clipart -sd painting    --log_name  lora --num_shots 2

python3 -u main.py  -e 20 -data Sketchy  -bs 24 --log_name ivla --num_shots 2
python3 -u main.py  -e 20 -data TUBerlin -bs 24 --log_name ivla --num_shots 2

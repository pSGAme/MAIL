import os

def check_null_bytes(direc):
    for root,dirs,files in os.walk(direc):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                    if b'\x00' in content:
                        print(file_path)
            except:
                print("nad")

direc = "/home/user/Code/DePro"
check_null_bytes(direc)
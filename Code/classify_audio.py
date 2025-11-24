import os

if os.path.isdir("data"):
    if not os.path.isdir("fernando"):
        try:
            os.mkdir("data/fernando")
        except:
            pass
    if not os.path.isdir("kaffee"):
        try:
            os.mkdir("data/kaffee")
        except:
            pass
    if not os.path.isdir("gross"):
        try:
            os.mkdir("data/gross")
        except:
            pass
    if not os.path.isdir("klein"):
        try:
            os.mkdir("data/klein")
        except:
            pass
    if not os.path.isdir("ok"):
        try:
            os.mkdir("data/ok")
        except:
            pass

files = os.listdir("data")

for file in files:
    if file.endswith('.wav'):
        if file.startswith("kaffee"):
            src_path = os.path.join("data", file)
            dest_path = os.path.join("data","kaffee", file)
            os.rename(src_path, dest_path)
        if file.startswith("fernando"):
            src_path = os.path.join("data", file)
            dest_path = os.path.join("data","fernando", file)
            os.rename(src_path, dest_path)
        if file.startswith("gross"):
            src_path = os.path.join("data", file)
            dest_path = os.path.join("data","gross", file)
            os.rename(src_path, dest_path)
        if file.startswith("klein"):
            src_path = os.path.join("data", file)
            dest_path = os.path.join("data","klein", file)
            os.rename(src_path, dest_path)
        if file.startswith("ok"):
            src_path = os.path.join("data", file)
            dest_path = os.path.join("data","ok", file)
            os.rename(src_path, dest_path)

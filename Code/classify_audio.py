import os

if os.path.isdir("data2"):
    if not os.path.isdir("fernando"):
        try:
            os.mkdir("data2/fernando")
        except:
            pass
    if not os.path.isdir("kaffee"):
        try:
            os.mkdir("data2/kaffee")
        except:
            pass
    if not os.path.isdir("gross"):
        try:
            os.mkdir("data2/gross")
        except:
            pass
    if not os.path.isdir("klein"):
        try:
            os.mkdir("data2/klein")
        except:
            pass
    if not os.path.isdir("ok"):
        try:
            os.mkdir("data2/ok")
        except:
            pass

files = os.listdir("data2")

for file in files:
    if file.endswith('.wav'):
        if file.startswith("kaffee"):
            src_path = os.path.join("data2", file)
            dest_path = os.path.join("data2","kaffee", file)
            os.rename(src_path, dest_path)
        if file.startswith("fernando"):
            src_path = os.path.join("data2", file)
            dest_path = os.path.join("data2","fernando", file)
            os.rename(src_path, dest_path)
        if file.startswith("gross"):
            src_path = os.path.join("data2", file)
            dest_path = os.path.join("data2","gross", file)
            os.rename(src_path, dest_path)
        if file.startswith("klein"):
            src_path = os.path.join("data2", file)
            dest_path = os.path.join("data2","klein", file)
            os.rename(src_path, dest_path)
        if file.startswith("ok"):
            src_path = os.path.join("data2", file)
            dest_path = os.path.join("data2","ok", file)
            os.rename(src_path, dest_path)

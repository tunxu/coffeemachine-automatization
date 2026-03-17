import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

# Map BuiltinOperator enum int -> name (fallback-safe)
try:
    BUILTIN = schema_fb.BuiltinOperator
    builtin_names = {getattr(BUILTIN, k): k for k in dir(BUILTIN) if k.isupper()}
except Exception:
    builtin_names = {}

def op_list(path):
    buf = open(path, "rb").read()
    m = schema_fb.Model.GetRootAsModel(buf, 0)
    sub = m.Subgraphs(0)

    ops = []
    for i in range(sub.OperatorsLength()):
        op = sub.Operators(i)
        opcode = m.OperatorCodes(op.OpcodeIndex())
        code = opcode.BuiltinCode()
        ops.append(builtin_names.get(code, f"BuiltinCode({code})"))
    return ops

def inspect_io(path):
    itp = tf.lite.Interpreter(model_path=path)
    itp.allocate_tensors()
    print(f"\n== {path} ==")
    for d in itp.get_input_details():
        print("IN ", d["name"], d["shape"], d["dtype"], d.get("quantization"))
    for d in itp.get_output_details():
        print("OUT", d["name"], d["shape"], d["dtype"], d.get("quantization"))

if __name__ == "__main__":
    a = "model_files/kws_model.tflite"

    inspect_io(a)

    print("\nOps in", a, ":\n ", op_list(a))

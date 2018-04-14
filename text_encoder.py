
def encode_line(text, line_length=80):
    temp = [encode_char(ord(t) - 32) for t in text[:line_length]]
    for _ in range(line_length - len(temp)):
        temp.append(encode_char(95)) # padding
    return temp

def decode_line(text):
    temp = []
    for t in text:
        for i, y in enumerate(t):
            if y:
                temp.append(chr(i + 32))
    return ''.join(temp)

def encode_char(x):
    q = [0] * 96
    try:
        q[x] = 1
    except IndexError:
        q[-1] = 1
    return q

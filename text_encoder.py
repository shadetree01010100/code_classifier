LINE_LENGTH = 80
NUM_CHARS = 96

def encode_line(text):
    temp = [encode_char(ord(t) - 32) for t in text[:LINE_LENGTH]]
    for _ in range(LINE_LENGTH - len(temp)):  # padding
        temp.append(encode_char(NUM_CHARS - 1))
    return temp

def decode_line(text):
    temp = []
    for t in text:
        for i, y in enumerate(t):
            if y:
                temp.append(chr(i + 32))
    return ''.join(temp)

def encode_char(x):
    q = [0] * NUM_CHARS
    try:
        q[x] = 1
    except IndexError:
        q[-1] = 1
    return q

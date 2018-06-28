

def tokenize_line(line):
    punc = ["!", "?", ".", ",", ";", "/", "-", "$"]
    line = line.lower()
    line = line.replace("doesn't", "does not")
    line = line.replace("can't", "can not")
    line = line.replace("n't", " not")
    line = line.replace("i'm", "i am")
    for p in punc:
        line = line.replace(p, " "+p+" ")
    return line.split()













from multiprocessing import Pool
from collections import Counter
import sys, re
import argparse


BUFSIZE = 100000
MAX_LEN = 1000

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', type=str)
    parser.add_argument('--tgt_file', type=str)
    parser.add_argument('--nprocessors', type=int)
    return parser.parse_args()


_split_set = set(['!', '?', '.'])
def _is_split_point(ch):
    if ch in _split_set:
        return True
    return False

def process(line):
    xs = []
    line = line.strip("\n").strip().lower()
    if line == "":
        return xs
    char_seq = line.split()
    xs = []
    sent = []
    for ch in char_seq:
        sent.append(ch)
        if len(sent) >= 10 and _is_split_point(ch):
            xs.append(sent)
            sent = []
    res = []
    xi = []
    for i in range(len(xs)):
        sent = xs[i]
        if len(xi) + len(sent) <= MAX_LEN:
            xi.extend(sent)
        else:
            res.append(xi)
            xi = []
            i -= 2 # overlap
    if xi:
        res.append(xi)
    return res

def save(cnt, lines, nprocessors, fo):
    res = pool.map(process, lines, len(lines)//nprocessors)
    all_lines = []
    for xs in res:
         all_lines.extend(xs)
    
    for x in all_lines:
        cnt.update(x)
        fo.write(' '.join(x)+'\n')


if __name__ == "__main__":
    print("start..")
    args = parse_config()
    pool = Pool(args.nprocessors)
    cnt = Counter()
    lines = []
    with open(args.tgt_file, 'w', encoding ='utf8') as fo:
        with open(args.src_file, "r") as fi:
            for line in fi:
                lines.append(line)
                if len(lines) == BUFSIZE:
                    save(cnt, lines, args.nprocessors, fo)                    
                    lines = []
                    print(BUFSIZE)
        if lines:
            save(cnt, lines, args.nprocessors, fo)
            print(len(lines))

    print("vocab")
    with open(args.tgt_file + '_vocab', 'w', encoding ='utf8') as f:
        for x, y in cnt.most_common():
            f.write(x + '\t' + str(y) + '\n')
    print("done")

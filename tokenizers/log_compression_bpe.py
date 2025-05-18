import re
import json
from collections import defaultdict, Counter
import argparse
import os

# --------- BPE Tokenizer Implementation ----------
class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = []

    def get_stats(self, tokens):
        pairs = defaultdict(int)
        for word in tokens:
            symbols = word
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += 1
        return pairs

    def merge_vocab(self, pair, tokens):
        new_tokens = []
        bigram = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in tokens:
            word_str = ' '.join(word)
            word_str = pattern.sub(''.join(pair), word_str)
            new_tokens.append(word_str.split())
        return new_tokens

    def train(self, corpus, num_merges=100):
        tokens = [list(word) for word in corpus.split()]
        for _ in range(num_merges):
            pairs = self.get_stats(tokens)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            tokens = self.merge_vocab(best, tokens)
        self.vocab = {''.join(pair): pair for pair in self.merges}

    def encode(self, text):
        tokens = list(text)
        for a, b in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == a and tokens[i+1] == b:
                    tokens[i:i+2] = [''.join((a, b))]
                    i -= 1 if i else 0
                i += 1
        return tokens

    def decode(self, tokens):
        text = tokens[:]
        for merged, pair in reversed(self.vocab.items()):
            new_text = []
            for token in text:
                if token == merged:
                    new_text.extend(pair)
                else:
                    new_text.append(token)
            text = new_text
        return ''.join(text)

# --------- Timestamp Extraction and Placeholder ----------
def extract_timestamps(lines):
    timestamp_regex = re.compile(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}')
    ts_map = []
    new_lines = []
    for i, line in enumerate(lines):
        match = timestamp_regex.search(line)
        if match:
            ts = match.group(0)
            ts_map.append(ts)
            line = line.replace(ts, f'<TS{i}>', 1)
        new_lines.append(line)
    return new_lines, ts_map

# --------- Compression and Decompression Interface ----------
def compress_log(input_path, output_path, num_merges=100):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    processed_lines, ts_map = extract_timestamps(lines)
    text = ' '.join(processed_lines)

    tokenizer = BPETokenizer()
    tokenizer.train(text, num_merges=num_merges)
    tokens = tokenizer.encode(text)

    result = {
        'tokens': tokens,
        'vocab': tokenizer.vocab,
        'timestamps': ts_map
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Compressed to {output_path}")


def decompress_log(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    tokenizer = BPETokenizer()
    tokenizer.vocab = data['vocab']
    tokenizer.merges = [tuple(pair) for pair in data['vocab'].values()]

    text = tokenizer.decode(data['tokens'])
    for i, ts in enumerate(data['timestamps']):
        text = text.replace(f'<TS{i}>', ts)

    with open(output_path, 'w') as f:
        f.write(text)

    print(f"Decompressed to {output_path}")

# --------- CLI Interface ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['compress', 'decompress'], required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--merges', type=int, default=100)
    args = parser.parse_args()

    if args.mode == 'compress':
        compress_log(args.input, args.output, args.merges)
    elif args.mode == 'decompress':
        decompress_log(args.input, args.output)

if __name__ == '__main__':
    main()

# --------- Step 1: Log Pattern Extractor + Semantic Token Replacer ----------
import re
import json
import argparse
from collections import defaultdict

# Predefined semantic patterns (you can expand this)
PATTERN_MAP = [
    (r"User (\S+) logged in from (\d+\.\d+\.\d+\.\d+)", "INFO_LOGIN user=\1 ip=\2"),
    (r"Failed to write to disk (\/\S+)", "ERR_DISK_WRITE path=\1"),
    (r"Connection timeout for (\d+\.\d+\.\d+\.\d+)", "ERR_CONN_TIMEOUT ip=\1"),
    (r"New connection from (\d+\.\d+\.\d+\.\d+)", "INFO_CONN_NEW ip=\1")
]

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

def replace_patterns(lines):
    compressed = []
    for line in lines:
        replaced = line
        for pattern, replacement in PATTERN_MAP:
            replaced = re.sub(pattern, replacement, replaced)
        compressed.append(replaced)
    return compressed

def compress_log(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    lines, ts_map = extract_timestamps(lines)
    compressed_lines = replace_patterns(lines)

    result = {
        'compressed_logs': compressed_lines,
        'timestamps': ts_map
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Compressed semantic log written to {output_path}")

def decompress_log(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    lines = data['compressed_logs']
    ts_map = data['timestamps']

    for i, ts in enumerate(ts_map):
        lines[i] = lines[i].replace(f'<TS{i}>', ts)

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Decompressed log written to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['compress', 'decompress'], required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    if args.mode == 'compress':
        compress_log(args.input, args.output)
    elif args.mode == 'decompress':
        decompress_log(args.input, args.output)

if __name__ == '__main__':
    main()

import sys
from collections import Counter

try:
    num_words = int(sys.argv[1])
except:
    print("Usage: python most_common_words.py num_words")
    sys.exit(1)

counter = Counter([word.lower() for line in sys.stdin
        for word in line.strip().split()])

for word, count  in counter.most_common(num_words):
    sys.stdout.write(f"{word}: {count} \n")

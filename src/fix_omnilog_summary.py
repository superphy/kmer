import os
import re


pattern = re.compile(r'[P][M]\d+\s\{[a-z]*\s\=\s\"[C-z]*\s[a-z]*\s[a-z]*\"\}\s\w\d+\s')

with open("data/data_summary.txt", 'r') as read, open("data/onmilog_data_summary.txt", 'w') as w:
    for line in read:
        if re.search(pattern, line):
            w.write(re.sub(pattern, '', line))
        else:
            w.write(line)

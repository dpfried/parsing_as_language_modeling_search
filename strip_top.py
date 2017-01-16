#!/usr/bin/env python
import fileinput

def strip_line(line):
    l = line.strip()
    if l.startswith("(TOP"):
        l = l[4:-1].strip()
    elif l.startswith("(S1"):
        l = l[3:-1].strip()
    return l

if __name__ == "__main__":
    for line in fileinput.input():
        proc_line = strip_line(line)
        print(proc_line)


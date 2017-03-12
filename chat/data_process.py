#coding:utf-8
from __future__ import unicode_literals
from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from snownlp import SnowNLP
import jieba

def wordseg(fp):
    for line in fp:
        line = line.decode('utf-8',errors='ignore').strip().split("\t")
        line = [" ".join(jieba.cut(x)) for x in line]
        print("\t".join(line))

if __name__ == "__main__":
    wordseg(sys.stdin)

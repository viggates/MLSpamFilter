from collections import defaultdict
import math
import glob
import os
import re

DF = defaultdict(int)
path = "/opt/Tanium/BoW/code" 
for filename in glob.glob(os.path.join(path, 'enron/enron1/ham/*.txt')):
    words = re.findall(r'\w+', open(filename).read().lower())
    for word in set(words):
        if len(word) >= 3 and word.isalpha():
            DF[word] += 1  # defaultdict simplifies your "if key in word_idf: ..." part.

import pdb;pdb.set_trace()
# Now you can compute IDF.
IDF = dict()
doccounter = 1
for word in DF:
    IDF[word] = math.log(doccounter / float(DF[word]))

print IDF

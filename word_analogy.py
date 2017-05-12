import pickle
import numpy as np
import scipy as sc
from scipy import spatial
import sys

dictionary, embeddings = pickle.load(open('Models_nce\word2vec2.model', 'rb'))
with open('Developing.txt') as input:
    count = 0
    with open('Answers_files_nce\Answers2', 'w') as output:
        for line in input:
            line = line[0:-1]
            count += 1
            pairs = []
            pairs = line.split(",")
            minPair = ""
            minValue = sys.maxsize
            maxPair = ""
            maxValue = -sys.maxsize
            diff = []
            for pair in pairs:
                pair = pair[1:-1]
                word = pair.split(":")
                try:
                    vector1 = embeddings[dictionary[word[0]]]
                    vector2 = embeddings[dictionary[word[1]]]
                except:
                    continue
                diff.append(vector1 - vector2)
            pairMetrics = []
            for d1 in diff:
                sum = 0
                for d2 in diff:
                    if set(d1) != set(d2):
                        sum += spatial.distance.cosine(d1, d2)
                pairMetrics.append(sum)

            maxIndex = np.argmax(pairMetrics)
            minIndex = np.argmin(pairMetrics)
            maxPair = pairs[maxIndex]
            minPair = pairs[minIndex]

            pairs.append(maxPair)
            pairs.append(minPair)
            output.write(" ".join(pairs) + "\n")


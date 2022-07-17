# Acoustic Handwriting Recognition

- [Full report (pdf)](https://github.com/ollieperree/emc-acoustic-handwriting/raw/master/report.pdf)

Handwriting recognition using only acoustic data has previously been explored using techniques
such as template matching and, recently, deep learning. In this work we investigate ways to create
features from audio recordings, and train deep neural network models for performing classification on
handwritten digits, and a selection of 72 common English words (48% accuracy; 31% when evaluated
on writers whose handwriting wasn’t used in training). We also implement a nearest neighbour
classifier based on previous works using signal difference integral as a similarity measure.

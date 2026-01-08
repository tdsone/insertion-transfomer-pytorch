# dna-insertion-transfomer-pytorch
A pytorch implementation of Stern's Insertion Transformer

Challenge 1: can you write a loss function that only inserts in the middle? (This is almost like a BERT model where at each step you just insert a mask token in the middle betweeen the two halves with the difference that we do not consider the hypothesis canvas to be done.)
Challenge 2: can you write a loss function that would prioritze starting with annotated gene bodies first (e.g. first start codon, then stop codon, then in between, then promoter, then terminator?)
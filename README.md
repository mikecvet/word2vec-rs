# word2vec-rs
Simple implementation of Word2Vec in Rust

This is an implementation of Word2Vec in Rust, from the ground up, and without using any available ML libraries or purpose-built crates such as [finalfusion](https://docs.rs/finalfusion/latest/finalfusion/) which is a purpose-built Rust toolkit for working with embeddings.

The `word2vec-rs` binary trains a hidden layer via a basic feedforward neural network with a softmax regression classifier. It then extracts embeddings from the model's weights matrix and compares quality of similarity predictions between cosine-similarity and forward-propagation methods for given query terms.

Provided in this repository is a python script which is useful for parsing and sanitizing Wikipedia articles, which can be used as training data. Also provided are same example corpora used for testing. 
```
  ~>> ./target/release/word2vec-rs --input ./corpora/san_francisco.txt --predict gate --load ./model.out --no_train --analogy gate,bridge,golden
  Most similar nearby tokens to [gate] via nn forward propagation:

  [0]: golden	| probability: 0.1078739
  [1]: gate	| probability: 0.0633199
  [2]: park	| probability: 0.0241675
  [3]: bay	| probability: 0.0179222
  [4]: city	| probability: 0.0140273
  [5]: north	| probability: 0.0136982
  [6]: area	| probability: 0.0123178
  [7]: bridge	| probability: 0.0097368
  [8]: county	| probability: 0.0084362
  [9]: california	| probability: 0.0058315

  Most similar nearby tokens to [gate] via embeddings cosine similarity:

  [0]: bridge	| probability: 0.9279431
  [1]: golden	| probability: 0.9071595
  [2]: park	| probability: 0.8965557
  [3]: north	| probability: 0.8283078
  [4]: protect	| probability: 0.7988443
  [5]: national	| probability: 0.7634193
  [6]: makeshift	| probability: 0.7556177
  [7]: jfk	| probability: 0.7491959
  [8]: fort	| probability: 0.7381749
  [9]: tent	| probability: 0.7358739

  Computing analogy for "gate" - "bridge" + "golden" = ?
  best word analogy: "gate" - "bridge" + "golden" = "park" (similarity 0.86515)
```

Information about program arguments and flags:

```
  Usage: word2vec-rs [OPTIONS]

  Options:
        --analogy <a,b,c>        ask the model for a word analogy; A is to B as C is to ???
        --embedding_size <INT>   the dimensionality of the trained embeddings; defaults to 128
        --entropy                if true, prints out the cross-entropy loss after each training epoch
        --epochs <INT>           the number of training epochs to run within this invocation
        --input <FILE_PATH>      path to the training text to ingest
        --learning_rate <FLOAT>  the learning rate; defaults to 0.001
        --load <FILE_PATH>       path to a previously-written file containing model weight vectors
        --no_train               if true, skips any training and only executes query arguments
        --predict <WORD>         prints the top-ten most similar words to this argument, produced by forward propagation 
                                        and embedding cosine similarity
        --save                   if true, will save computed model weights to ./model.out, overwriting any existing local file
        --window_size <INT>      the sliding window size for token co-occurrence; defaults to 4
    -h, --help                   Print help
    -V, --version                Print version  
```

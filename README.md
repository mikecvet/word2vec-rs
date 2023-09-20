# word2vec-rs
Simple implementation of Word2Vec in Rust

This is an implementation of Word2Vec in Rust, from the ground up, and without using any available ML libraries or purpose-built crates such as [finalfusion](https://docs.rs/finalfusion/latest/finalfusion/) which is a purpose-built Rust toolkit for working with embeddings.

The `word2vec-rs` binary trains a hidden layer via a basic feedforward neural network with a softmax regression classifier. It then extracts embeddings from the model's weights matrix and compares quality of similarity predictions between cosine-similarity and forward-propagation methods for given query terms.

Provided in this repository is a python script which is useful for parsing and sanitizing Wikipedia articles, which can be used as training data. Also provided are same example corpora used for testing. 
```
  ~>> ./target/release/word2vec-rs --input ./corpora/sf_ba_oak.txt --predict oakland --epochs 65 --save
  initialized model with weight vector dimensions [5482, 256]

  Most similar nearby tokens to [oakland] via nn forward propagation:

  [0]: san        | probability: 0.0002793
  [1]: francisco  | probability: 0.0002664
  [2]: bay        | probability: 0.0002495
  [3]: area       | probability: 0.0002217
  [4]: jose       | probability: 0.0002187
  [5]: city       | probability: 0.0001807
  [6]: california | probability: 0.0000878

  ...

  Computing analogy for "alameda" - "oakland" + "clara" = ?
  ...
  best word analogy: "alameda" - "oakland" + "clara" = "peninsula" (0.92173)
```

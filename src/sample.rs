use std::collections::HashMap;

pub struct SubSampler {
  term_counts: HashMap<String, usize>,
  term_total: f64
}

impl SubSampler {
  pub fn new (term_counts: HashMap<String, usize>, term_total: f64) -> Self 
  {
    SubSampler {
      term_counts,
      term_total,
    }
  }

  /// Returns the probability of keeping a word in the dataset based on its frequency.
  fn keep_probability (&self, word: &str) -> f64 
  {
    let word_frequency = match self.term_counts.get(word) {
        Some(&freq) => freq as f64 / self.term_total,
        None => return 1.0, // If the word doesn't exist in our map, keep it.
    };

    let t = 1e-5; 
    //(word_frequency / t).sqrt() + t / word_frequency
    (1.0 - (t / word_frequency).sqrt()) as f64
  }

  /// Decides whether to keep the word based on a randomly generated value and its keep probability.
  /// Discussed in paper https://arxiv.org/pdf/1310.4546.pdf
  pub fn should_keep (&self, word: &String) -> bool {
    let r: f64 = rand::random(); // A value between 0.0 and 1.0
    r < self.keep_probability(word)
  }
}
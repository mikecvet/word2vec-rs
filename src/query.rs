use ordered_float::OrderedFloat;
use std::collections::{BinaryHeap, HashMap};

pub use crate::metadata::Metadata;
pub use crate::model::*;

/// Executes a prediction from our neural network. Ultimately converts the given query string 
/// into a one-hot-encoded vector and runs it through a forward propagation process. Sorts
/// the set of similar tokens to the given query and prints out the top ten, along with their probabilities
pub fn
nn_prediction (
  query: &str,
  model: &Model, 
  token_to_index: &HashMap<String, usize>,
  index_to_token: &HashMap<usize, String>, 
  vocab_size: usize
) -> Vec<(String, f64)> 
{
  // Encode the query into a one-hot vector
  let query_vector = encode_to_vector(
    *token_to_index.get(query).unwrap(), 
    vocab_size
  );

  let mut results: Vec<(String, f64)> = Vec::new();

  // Generates predictions for the query vector
  let p_results = model.forward_propagation(&query_vector);

  // The probability distribution from the propagation
  let probabilities = p_results.1.row(0);

  // Create a vector of indices which are then sorted in descending order acording to the probability of
  // tokens at that index, when compared to the probabilities vector
  let mut indices: Vec<usize> = (0..probabilities.len()).collect();
  indices.sort_by(|&a, &b| probabilities[b].partial_cmp(&probabilities[a]).unwrap());
  let sorted_values: Vec<f64> = indices.iter().map(|&i| probabilities[i]).collect();

  let mut i = 0;
  for iter in indices.iter().zip(sorted_values.iter()) {
    let w = &index_to_token[iter.0];

    if !query.eq(w) {
      results.push((w.clone(), *iter.1));
      i += 1;

      if i >= 10 {
          break;
      }
    }
  }

  results
}

/// Given a query string and a map of token->vector embeddings, executes a cosine similarity evaluation of this
/// query term's embedding against the complete set of available embeddings to find the nearest matches. 
/// Returns the top ten matches and their corresponding similarity
pub fn 
closest_embeddings (query: &str, embeddings: &HashMap<String, Vec<f64>>) -> Vec<(String, f64)>
{
  let mut results: Vec<(String, f64)> = Vec::new();
  let mut heap: BinaryHeap<(OrderedFloat<f64>, String)> = BinaryHeap::new();

  // Get the embedded representation of this query term
  let query_vector = embeddings.get(query).unwrap();

  // Iterate over all word embeddings, calculate their cosine similarity, and insert into a heap. 
  // This heap will be used to extract the most similar terms afterwards
  for (word, vector) in embeddings.iter() {
    if !word.eq(query) {
      let similarity = cosine_similarity(&query_vector, vector);
      heap.push((OrderedFloat(similarity), word.to_string()));
    }
  }

  // Collect the top-ten most similar terms and add them to the results vector
  for _ in 0..10 {
    let tmp = heap.pop().unwrap();
    results.push((tmp.1, tmp.0.0));
  }

  results
}

/// Calculates and returns the cosine similarity between two vectors
pub fn 
cosine_similarity (v1: &[f64], v2: &[f64]) -> f64 
{
  let dot_product: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
  let norm_v1: f64 = v1.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();
  let norm_v2: f64 = v2.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();

  dot_product / (norm_v1 * norm_v2)
}

/// Computes an embeddings-based "word analogy", based on the example provided in the original Word2Vec paper
/// https://arxiv.org/pdf/1301.3781.pdf
/// 
///   vector(”King”) - vector(”Man”) + vector(”Woman”) -> vector("Queen")
/// 
/// Given three strings, a, b, and c, attempts to return the most likely analogy given that context, which is
///   a - b + c = ?
/// 
/// This is done by computing that vector math as described, and then iterating over the entire set of
/// embeddings to find the highest cosine similarity to the target vector `u`
pub fn
word_analogy (a: &str, b: &str, c: &str, embeddings: &HashMap<String, Vec<f64>>) -> Option<(String, f64)> 
{
  println!("\nComputing analogy for \"{}\" - \"{}\" + \"{}\" = ?", a, b, c);
  let v_a = embeddings.get(a)?;
  let v_b = embeddings.get(b)?;
  let v_c = embeddings.get(c)?;
  
  let mut u = vec![0.0; v_a.len()];
  for i in 0..v_a.len() {
      u[i] = v_a[i] - v_b[i] + v_c[i];
  }

  let mut best_word = None;
  let mut max_similarity = f64::NEG_INFINITY;

  for (word, vector) in embeddings.iter() {
      if word != a && word != b && word != c {
          let similarity = cosine_similarity(&u, vector);

          if similarity > max_similarity {
              max_similarity = similarity;
              best_word = Some((word.clone(), max_similarity));
          }
      }
  }
  
  best_word
}
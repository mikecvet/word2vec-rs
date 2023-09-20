use clap::{arg, Command};
use std::fs;
use ndarray::{Array2};

use ordered_float::OrderedFloat;
use rand::prelude::*;

use std::collections::{BinaryHeap, HashMap};

use word2vec_rs::*;

pub use crate::metadata::Metadata;
pub use crate::train::*;
pub use crate::subsampler::SubSampler;

const EMBEDDINGS_SIZE: usize = 96;
const DEFAULT_EPOCHS: usize = 65;
const WINDOW_SIZE: i32 = 5;
//const LEARNING_RATE: f64 = 0.001;
//const LEARNING_RATE: f64 = 0.0004;
const LEARNING_RATE: f64 = 0.0001;

fn
run (
  epochs: usize, 
  print_entropy: bool, 
  query: Option<&String>, 
  analogy: Vec<String>,
  model_path: Option<&String>, 
  save: bool, 
  text: &str
) {
  let metadata = Metadata::init(text);
  let mut model = Model::new(metadata.vocab_size, EMBEDDINGS_SIZE);

  match model_path {
    // Load a serialized model, if it exists
    Some(path) => model.load_from_file(path).unwrap(),

    // Otherwise, initialize weights from scratch
    _ => model.init_nn_weights_he()
  }

  if epochs > 0 {
    // Train the model for the given number of epochs
    train_model(&mut model, &metadata, WINDOW_SIZE, epochs, LEARNING_RATE, print_entropy);
  }
  
  if save {
    match model.save_to_file("./model.out") {
      Err(error) => println!("error: {}", error),
      _ => ()
    }
  }

  let mut word_embeddings: HashMap<String, Vec<f64>> = HashMap::new();
  for entry in metadata.token_to_id.iter() {
    word_embeddings.insert(
      entry.0.clone(), 
      model.extract_embedding(entry.0, &metadata.token_to_id).unwrap()
    );
  }

  match query {
    Some(q) => {
        let nn_results = nn_forward_propagation(q, &model, &metadata.token_to_id, &metadata.id_to_token, metadata.vocab_size);
        let e_results = closest_embeddings (q, &word_embeddings);

        print_query_results(q, &nn_results, &e_results);
    },
    _ => ()
  }

  if analogy.len() == 3 {
    let a = analogy.get(0).unwrap();
    let b = analogy.get(1).unwrap();
    let c = analogy.get(2).unwrap();

    match word_analogy(a, b, c, &word_embeddings) {
      Some(result) => {
        println!("best word analogy: \"{}\" - \"{}\" + \"{}\" = \"{}\" (similarity {:.5})", 
          a, b, c, result.0, result.1)
      },
      _ => println!("could not find an analogy for {}", ""),
    }
  }
}

fn 
print_query_results (query: &str, nn_results: &Vec<(String, f64)>, e_results: &Vec<(String, f64)>) 
{  
  let mut i = 0;

  println!("Most similar nearby tokens to [{}] via nn forward propagation:\n", query);
  for iter in nn_results.iter() {
    println!("[{}]: {}\t| probability: {:.7}", i, iter.0, iter.1);
    i += 1;
  }

  println!("\nMost similar nearby tokens to [{}] via embeddings cosine similarity:\n", query);
  for iter in e_results.iter() {
    println!("[{}]: {}\t| probability: {:.7}", i, iter.0, iter.1);
    i += 1;
  }
}

/// Executes a prediction from our neural network. Ultimately converts the given query string 
/// into a one-hot-encoded vector and runs it through a forward propagation process. Sorts
/// the set of similar tokens to the given query and prints out the top ten, along with their probabilities
fn
nn_forward_propagation (
  query: &str,
  model: &Model, 
  token_to_index: &HashMap<String, usize>,
  index_to_token: &HashMap<usize, String>, 
  vocab_size: usize
) -> Vec<(String, f64)> 
{
  // Encode the query into a one-hot vector
  let query_vector = encode(
    *token_to_index.get(query).unwrap(), 
    vocab_size
  );

  let mut results: Vec<(String, f64)> = Vec::new();
  // Generates predictions for the query vector
  let p_results = model.forward_propagation(&query_vector);

  // The probability distribution from the propagation
  let probabilities = p_results.1.row(0);

  let mut indices: Vec<usize> = (0..probabilities.len()).collect();
  indices.sort_by(|&a, &b| probabilities[b].partial_cmp(&probabilities[a]).unwrap());
  let sorted_values: Vec<f64> = indices.iter().map(|&i| probabilities[i]).collect();

  let mut i = 0;
  for iter in indices.iter().zip(sorted_values.iter()) {
    results.push((index_to_token[iter.0].clone(), *iter.1));
    i += 1;

    if i >= 10 {
        break;
    }
  }

  results
}

/// Given a query string and a map of token->vector embeddings, executes a cosine similarity evaluation of this
/// query term's embedding against the complete set of available embeddings to find the nearest matches. 
/// Returns the top ten matches and their corresponding similarity
fn 
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
fn 
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
fn
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

fn 
main () 
{
  let matches = Command::new("word2vec-rs")
    .version("0.1")
    .about("Simple word2vec implementation")
    .arg(arg!(--analogy <VALUE>).required(false))
    .arg(arg!(--entropy).required(false))
    .arg(arg!(--input <VALUE>).required(false))
    .arg(arg!(--epochs <VALUE>).required(false))
    .arg(arg!(--predict <VALUE>).required(false))
    .arg(arg!(--load <VALUE>).required(false))
    .arg(arg!(--save).required(false))
    .get_matches();

  let analogy_opt = matches.get_one::<String>("analogy");
  let entropy_opt = matches.get_one::<bool>("entropy");
  let input_opt = matches.get_one::<String>("input");
  let epochs_opt = matches.get_one::<String>("epochs");
  let predict_opt = matches.get_one::<String>("predict");
  let load_opt = matches.get_one::<String>("load");
  let save_opt = matches.get_one::<bool>("save");

  let epochs = match epochs_opt.as_deref() {
    Some(epoch_string) => epoch_string.parse::<usize>().unwrap(),
    _ => DEFAULT_EPOCHS
  };

  // Load the text to process and train upon
  let text = match input_opt {
    Some(path) => {
        fs::read_to_string(path).expect("unable to open file!").parse().unwrap()
    },
    _ => {
        "".to_string()
    }
  };

  let save = match save_opt {
    Some(s) => s,
    _ => &true
  };

  let entropy = match entropy_opt {
    Some(e) => e,
    _ => &false
  };

  let mut analogy: Vec<String> = Vec::new();
  match analogy_opt {
    Some(a) => {
      let parts = a.split(",");
      for iter in parts {
        analogy.push(iter.to_string());
      }

      if analogy.len() != 3 {
        panic!("--analogy requires an argument of three tokens separated by commas; for example, << a,b,c >>")
      }
    }
    _ => ()
  }

  run(epochs, *entropy, predict_opt, analogy, load_opt, *save, &text);
}
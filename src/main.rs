use clap::{arg, Command};
use std::{fs, io::Read};
use ndarray::{Array, Array1, Array2, ArrayView, Axis, Ix2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ordered_float::OrderedFloat;
use rand::distributions::Standard;
use rand::prelude::*;
use rand::{thread_rng, Rng};
use regex::Regex;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Reverse;
use std::ops::{Mul, Sub, SubAssign};
use word2vec_rs::*;

pub use crate::metadata::Metadata;
pub use crate::train::*;
pub use crate::sample::SubSampler;

const EMBEDDINGS_SIZE: usize = 256;
const DEFAULT_EPOCHS: usize = 65;
const WINDOW_SIZE: i32 = 3;
//const LEARNING_RATE: f64 = 0.001
const LEARNING_RATE: f64 = 0.0003;

fn
run (
  epochs: usize, 
  print_entropy: bool, 
  query: Option<&str>, 
  model_path: Option<&String>, 
  save: bool, 
  text: &str
) {
  let metadata = Metadata::init(text);

  let mut model = Model::new(metadata.vocab_size, EMBEDDINGS_SIZE);

  match model_path {
    Some(path) => model.load_from_file(path).unwrap(),
    _ => model.init_nn_weights_he()
  }

  //println!("initialized model with weight vector dimensions {:?}", model.w1.shape());

  train_model(&mut model, &metadata, WINDOW_SIZE, epochs, LEARNING_RATE, print_entropy);
  
  println!("model trained");

  if save {
    match model.save_to_file("./model.out") {
      Err(error) => println!("error: {}", error),
      _ => ()
    }
  }

  match query {
    Some(q) => {
        nn_forward_propagation(q, &model, &metadata.token_to_index, &metadata.index_to_token, metadata.vocab_size);
        
        let mut word_embeddings: HashMap<String, Vec<f64>> = HashMap::new();
        for entry in metadata.token_to_index.iter() {
          word_embeddings.insert(
            entry.0.clone(), 
            get_embedding(&model, entry.0, &metadata.token_to_index).unwrap()
          );
        }

        closest_embeddings (&word_embeddings, q);

        match word_analogy(&word_embeddings, q, "oakland", "clara") {
          Some(analogy) => println!("best word analogy: {}", analogy),
          _ => println!("could not find an analogy for {}", ""),
        }
    },
    _ => ()
  }
}

fn
nn_forward_propagation (
  query: &str,
  model: &Model, 
  token_to_index: &HashMap<String, usize>,
  index_to_token: &HashMap<usize, String>, 
  vocab_size: usize
) {
  let a_learning = encode(
    *token_to_index.get(query).unwrap(), 
    vocab_size
  );

  let results = forward_propagation(&model, &a_learning);
  let probabilities = results.1.row(0);

  let mut indices: Vec<usize> = (0..probabilities.len()).collect();
  indices.sort_by(|&a, &b| probabilities[b].partial_cmp(&probabilities[a]).unwrap());
  let sorted_values: Vec<f64> = indices.iter().map(|&i| probabilities[i]).collect();

  println!("Most similar nearby tokens to [{}] via nn forward propagation:\n", query);

  let mut i = 0;
  for iter in indices.iter().zip(sorted_values.iter()) {
    println!("[{}]: {}\t| probability: {}", i, index_to_token[iter.0], iter.1);
    i += 1;

    if i >= 10 {
        break;
    }
  }

  // print_embedding(
  //   query,
  //   &get_embedding(&model, query, &token_to_index).unwrap()
  // );
}

fn 
closest_embeddings (embeddings: &HashMap<String, Vec<f64>>, query: &str)
{
  let mut heap: BinaryHeap<(OrderedFloat<f64>, String)> = BinaryHeap::new();
  heap.push((OrderedFloat(0.4 as f64), "a".to_string()));

  let query_vector = embeddings.get(query).unwrap();

  for (word, vector) in embeddings.iter() {
    let similarity = cosine_similarity(&query_vector, vector);
    heap.push((OrderedFloat(similarity), word.to_string()));
  }

  println!("\nMost similar nearby tokens to [{}] via embeddings cosine similarity:\n", query);

  for i in 0..10 {
    let tmp = heap.pop().unwrap();
    println!("[{}]: {}\t| probability: {}", i, tmp.1, tmp.0.0);
  }
}

fn
print_embedding (token: &str, v: &Array2<f64>)
{
  println!("\nembedding vector for [{}]:", token);
  println!("[");
  for iter in v.iter() {
    println!("  {},", *iter);
  }
  println!("]");
}

fn 
cosine_similarity (v1: &[f64], v2: &[f64]) -> f64 
{
  let dot_product: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
  let norm_v1: f64 = v1.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();
  let norm_v2: f64 = v2.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();
  dot_product / (norm_v1 * norm_v2)
}

fn
word_analogy (embeddings: &HashMap<String, Vec<f64>>, a: &str, b: &str, c: &str) -> Option<String> 
{
  println!("\nComputing analogy for {} - {} + {} = ?", a, b, c);
  let v_a = embeddings.get(a)?;
  let v_b = embeddings.get(b)?;
  let v_c = embeddings.get(c)?;
  
  let mut target_vector = vec![0.0; v_a.len()];
  for i in 0..v_a.len() {
      target_vector[i] = v_a[i] - v_b[i] + v_c[i];
  }

  let mut best_word = None;
  let mut max_similarity = f64::NEG_INFINITY;

  for (word, vector) in embeddings.iter() {
      if word != a && word != b && word != c {
          let similarity = cosine_similarity(&target_vector, vector);

          if word.eq("clara") {
            println!(">> clara: {}", similarity);
          }

          if similarity > max_similarity {
              max_similarity = similarity;
              best_word = Some(word.clone());
              println!("setting best word to {:?} | {:.5}", best_word, similarity);
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
    .arg(arg!(--entropy).required(false))
    .arg(arg!(--input <VALUE>).required(false))
    .arg(arg!(--epochs <VALUE>).required(false))
    .arg(arg!(--predict <VALUE>).required(false))
    .arg(arg!(--load <VALUE>).required(false))
    .arg(arg!(--save).required(false))
    .get_matches();

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

  let text = match input_opt {
    Some(path) => {
        // https://www.corpusdata.org/formats.asp
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

  match (entropy_opt, predict_opt) {
    (Some(true), None) => {
        run(epochs, true, None, load_opt, *save, &text);
    },
    (Some(true), Some(query)) => {
        run(epochs, true, Some(query), load_opt, *save, &text)
    },
    (Some(false), Some(query)) => {
        run(epochs, false, Some(query), load_opt, *save, &text)
    },
    _ => {
        println!("no options provided");
    }
  }
}
use clap::{arg, Command};
use std::fs;
use ndarray::{Array2};

use ordered_float::OrderedFloat;
use rand::prelude::*;

use std::collections::{BinaryHeap, HashMap};

use word2vec_rs::*;

pub use crate::metadata::Metadata;
pub use crate::query::*;
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
        let nn_results = nn_prediction(q, &model, &metadata.token_to_id, &metadata.id_to_token, metadata.vocab_size);
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
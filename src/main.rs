use clap::{arg, Command};
use std::fs;
use std::collections::{HashMap};

use word2vec_rs::*;
pub use crate::args::Args;
pub use crate::metadata::Metadata;
pub use crate::query::*;
pub use crate::train::*;
pub use crate::subsampler::SubSampler;

fn
run (args: &Args)
{
  let metadata = Metadata::init(&args.text);
  let mut model = Model::new(metadata.vocab_size, args.hyper_params.embeddings_size);

  match &args.model_load_path {
    // Load a serialized model, if it exists
    Some(path) => model.load_from_file(&path).unwrap(),

    // Otherwise, initialize weights from scratch
    _ => model.init_nn_weights_he()
  }

  if args.hyper_params.num_epochs > 0 {
    // Train the model for the given number of epochs
    train_model(
      &mut model, 
      &metadata, 
      &args.hyper_params,
      args.print_entropy
    );
  }
  
  if args.save_model {
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

  match &args.predict {
    Some(q) => {
        let nn_results = nn_prediction(&q, &model, &metadata.token_to_id, &metadata.id_to_token, metadata.vocab_size);
        let e_results = closest_embeddings (&q, &word_embeddings);

        print_query_results(&q, &nn_results, &e_results);
    },
    _ => ()
  }

  match &args.analogy {
    Some((a, b, c)) => {
      match word_analogy(&a, &b, &c, &word_embeddings) {
        Some(result) => {
          println!("best word analogy: \"{}\" - \"{}\" + \"{}\" = \"{}\" (similarity {:.5})", 
            a, b, c, result.0, result.1)
        },
        _ => println!("could not find an analogy for {}", ""),
      }
    }
    _ => ()
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
    .arg(arg!(--embeddings_size <VALUE>).required(false))
    .arg(arg!(--entropy).required(false))
    .arg(arg!(--epochs <VALUE>).required(false))
    .arg(arg!(--input <VALUE>).required(false))
    .arg(arg!(--learning_rate <VALUE>).required(false))
    .arg(arg!(--load <VALUE>).required(false))
    .arg(arg!(--predict <VALUE>).required(false))
    .arg(arg!(--save).required(false))
    .arg(arg!(--window_size <VALUE>).required(false))
    .get_matches();

  let analogy_opt = matches.get_one::<String>("analogy").cloned();
  let entropy_opt = matches.get_one::<bool>("entropy").cloned();
  let input_opt = matches.get_one::<String>("input").cloned();
  let epochs_opt = matches.get_one::<String>("epochs").cloned();
  let predict_opt = matches.get_one::<String>("predict").cloned();
  let embeddings_size_opt = matches.get_one::<String>("embeddings_size").cloned();
  let learning_rate_opt = matches.get_one::<String>("learning_rate").cloned();
  let window_size_opt = matches.get_one::<String>("window_size").cloned();
  let load_opt = matches.get_one::<String>("load").cloned();
  let save_opt = matches.get_one::<bool>("save").cloned();

  let args = Args::new(
    analogy_opt, 
    input_opt, 
    load_opt,
    save_opt, 
    predict_opt, 
    entropy_opt, 
    embeddings_size_opt, 
    learning_rate_opt, 
    epochs_opt, 
    window_size_opt
  );

  run(&args);
}
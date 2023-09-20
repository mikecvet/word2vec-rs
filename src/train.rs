use ndarray::{Array2, Axis};
use std::collections::HashMap;

pub use crate::args::*;
pub use crate::metadata::Metadata;
pub use crate::model::*;
pub use crate::subsampler::SubSampler;

/// Builds a set of training matrices based on the skip-graph architecture for word2vec.
pub fn
build_skip_gram_training_data (
  tokens: &Vec<String>,
  token_to_index: &HashMap<String, usize>,
  sampler: &SubSampler,
  vocabulary_size: usize,
  window: i32,
) -> TrainingData
{
  let tlen: i32 = tokens.len() as i32;

  // Initialize empty vectors with estimated backing capacity needs
  let mut x: Vec<f64> = Vec::with_capacity(vocabulary_size * window as usize * 2);
  let mut y: Vec<f64> = Vec::with_capacity(vocabulary_size * window as usize * 2);

  let mut nrows = 0;

  for i in 0i32..tlen {

    // If this token has been downsampled, skip
    if !sampler.should_keep(&tokens[i as usize]) {
      continue;
    }

    // Sliding window over tokens[i]
    for j in (i - window)..(i + window + 1) {

      // Don't encode the position of a token relative to itself. Also skip tokens[j] if the token
      // should be downsampled due to its frequency
      if j >= 0 && j < tlen && i != j && sampler.should_keep(&tokens[j as usize]) {
        nrows += 1;

        // Encodes the position of tokens[i] juxtaposed to the position of tokens[j] in the token vector. This
        // is the skip-gram model architecture.
        // 
        // Given the two training matrices X and Y, each iteration of this loop encodes the distance between 
        // the tokens in two complementary matrices. For example, when i == 4, the relevant rows in X and Y 
        // look something like the following:
        // 
        // X:
        // [
        //   ...
        //
        //   [0, 0, 0, 0, 1, 0, 0, 0]  (index(tokens[4]))
        //   [0, 0, 0, 0, 1, 0, 0, 0]
        //   [0, 0, 0, 0, 1, 0, 0, 0]
        //   [0, 0, 0, 0, 1, 0, 0, 0]
        //
        //   ...
        // ]
        // 
        // Y:
        // [
        //   ...
        //
        //   [0, 0, 1, 0, 0, 0, 0, 0]  (index(tokens[2]))
        //   [0, 0, 0, 1, 0, 0, 0, 0]  (index(tokens[3]))
        //   [0, 0, 0, 0, 0, 1, 0, 0]  (index(tokens[5]))
        //   [0, 0, 0, 0, 0, 0, 1, 0]  (index(tokens[6]))
        //
        //   ...
        // ]
        // 
        // This trains the model of the relationship between the token at index(tokens[4]) and the surrounding tokens
        // at index(tokens[2, 3, 5 and 6]).

        let a = encode_to_vector(*token_to_index.get(&(tokens[i as usize])).unwrap(), vocabulary_size);
        let b = encode_to_vector(*token_to_index.get(&(tokens[j as usize])).unwrap(), vocabulary_size);

        // Append these encoding vectors to the X and Y vectors, which are currently 
        // lengthy single-dimension vectors to be encoded into 2D matrices later
        x.extend(a.iter());
        y.extend(b.iter());
      }
    }
  }
  
  // Convert the vectors into [nrows, vocab_size] shaped 2D matrices, which encode ground-truth
  // token proximal relationships.
  TrainingData {
    x: Array2::from_shape_vec((nrows, vocabulary_size), x).unwrap(), 
    y: Array2::from_shape_vec((nrows, vocabulary_size), y).unwrap()
  }
}

/// Trains the model. 
/// 
/// Given corpus metadata and hyperparams like window size and number of epochs, generates a 
/// training data set for each epoch and runs backpropagation against it.
pub fn 
train_model (
  model: &mut Model,
  metadata: &Metadata,
  hyper_params: &HyperParams,
  print: bool
) {
    // The SubSampler is used to nondeterministically remove frequent terms from the training data set. 
    // Since training data is recomputed during each epoch, each round will be trained against slight variations
    // in the training data matrices.
    let sampler = SubSampler::new(metadata.token_counts.clone(), metadata.tokens.len() as f64);
    let epochs = hyper_params.num_epochs;

    if print && epochs > 0 {
      println!("entopy per epoch:");
    }

    // epochs
    for _ in 0..epochs 
      {
        let td = build_skip_gram_training_data(
          &metadata.tokens, 
          &metadata.token_to_id, 
          &sampler, 
          metadata.vocab_size, 
          hyper_params.window_size
        );

        // Run backpropagation and possibly record the cross-entropy error from this process
        let ce = model.back_propagation(td, hyper_params.learning_rate);

        if ce.is_nan() {
          panic!("gradient explosion!");
        }

        if print {
            println!("{:.3}", ce);
        }
      }
}

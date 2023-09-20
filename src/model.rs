use ndarray::Array2;
use ndarray_rand::rand_distr::{Uniform, Normal};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Error, Read, Write};

pub use crate::metadata::Metadata;
pub use crate::sample::SubSampler;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
  /// Input embeddings; predicts context words for center word
  pub w1: Array2<f64>,

  /// Output embeddings; predicts probabilities for every word to be a context word
  pub w2: Array2<f64>,
}

pub struct TrainingData {
  pub x: Array2<f64>,
  pub y: Array2<f64>
}

impl Model {
  pub fn 
  new (x: usize, y: usize) -> Self 
  {
    Model {
      w1: Array2::zeros((x, y)),
      w2: Array2::zeros((y, x))
    }
  }

  /// Return a new Model using the provided Array2s as underlying weight vectors
  pub fn 
  update (w1: Array2<f64>, w2: Array2<f64>) -> Self 
  {
    Model {
      w1: w1,
      w2: w2
    }
  }

  /// Initializes the weight vectors for this model using the He (et al) initialization method 
  /// introduced here https://arxiv.org/pdf/1502.01852.pdf
  /// 
  /// The distribution mean is 0, with variance sqrt(2 / n or m) where for w1 n = vocabulary_size
  /// and m = embedding_size
  pub fn 
  init_nn_weights_he (&mut self) 
  {
    let mut rng = StdRng::from_entropy();

    let std_w1 = (2.0 / self.w1.len() as f64).sqrt();
    let distribution_w1 = Normal::new(0.0, std_w1).expect("Failed to create distribution");

    for val in self.w1.iter_mut() {
        *val = rng.sample(distribution_w1);
    }

    let std_w2 = (2.0 / self.w2.len() as f64).sqrt();
    let distribution_w2 = Normal::new(0.0, std_w2).expect("Failed to create distribution");

    println!("distriubtion 1 {:?} 2 {:?}", distribution_w1, distribution_w2);

    for val in self.w2.iter_mut() {
        *val = rng.sample(distribution_w2);
    }

    println!("initialized He weights: {}\n{}", self.w1, self.w2);
  }

  /// Returns an embedding vector for the given token string, if it exists. The embedding vector is a row in this 
  /// model's underlying trained `w1` weights matrix, corresponding to the index associated with the token via the 
  /// `token_to_index` map.
  /// 
  /// For example, given a `w1` matrix
  /// 
  /// [
  ///  [0.4, 0.5, 0.3],
  ///  [0.1, 0.2, 0.3],
  ///  [0.3, 0.2, 0.1]
  /// ]
  /// 
  /// And a `token_to_index` map of {'a' -> 2}
  /// 
  /// This function returns the third row of `w1` corresponding to [0.3, 0.2, 0.1] above.
  /// 
  pub fn 
  extract_embedding (&self, token: &str, token_to_index: &HashMap<String, usize>) -> Option<Vec<f64>> 
  {
    match token_to_index.get(token) {
      Some(indx) => {
        Some(self.w1.row(*indx).to_owned().into_raw_vec())
      },

      None => None
    }
  }

  /// Saves the contents of this model's `w1` and `w2` weight matrices to the given path.
  pub fn 
  save_to_file (&self, path: &str) -> Result<(), Error> 
  {
    let serialized_data = serde_json::to_string(&self)?;
    
    let mut file = File::create(path)?;

    match file.write_all(serialized_data.as_bytes()) {
      Ok(_) => println!("model data saved to {}", path),
      Err(e) => println!("error saving model data: {}", e)
    }

    Ok(())
  }

  /// Loads the contents of the model file located at the given path, if it exist, and sets this model's 
  /// `w1` and `w2` weight matrices to those contents.
  pub fn 
  load_from_file (&mut self, path: &str) -> Result<(), Error> 
  {
    let mut file = File::open(path)?;
    let mut serialized_data = String::new();

    file.read_to_string(&mut serialized_data)?;

    let deserialized_model: Model = serde_json::from_str(&serialized_data)?;
    self.w1 = deserialized_model.w1;
    self.w2 = deserialized_model.w2;

    println!("loaded model data from {}, w1.shape {:?} w2.shape {:?}", path, self.w1.shape(), self.w2.shape());
    
    Ok(())
  }
}

#[cfg(test)]
mod tests 
{
  use super::*;

  #[test]
  fn test_init () 
  {
    let model = Model::new(1, 1);

    assert!(model.w1.len() == 1);
    assert!(model.w2.len() == 1);
  }

  #[test]
  fn test_update () 
  {
    let x = Array2::zeros((1, 4));
    let y = Array2::zeros((1, 4));
    let model = Model::update(x, y);

    assert!(model.w1.len() == 4);
    assert!(model.w2.len() == 4);
  }

  #[test]
  fn test_get_embedding () 
  {
    let model = Model::new(1, 1);
    let mut map: HashMap<String, usize> = HashMap::new();

    map.insert("a".to_string(), 0);

    let e = model.extract_embedding("a", &map).unwrap();
    assert!(e.len() == 1);
  }
}
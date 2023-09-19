use ndarray::{Array, Array1, Array2, ArrayView, Axis, Ix2};
use ndarray_rand::rand_distr::{Uniform, Normal};
use rand::distributions::Standard;
use rand::prelude::*;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Error, Read, Write};
use std::collections::HashMap;

pub use crate::metadata::Metadata;
pub use crate::sample::SubSampler;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub w1: Array2<f64>,
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

  pub fn 
  update (w1: Array2<f64>, w2: Array2<f64>) -> Self 
  {
    Model {
      w1: w1,
      w2: w2
    }
  }

  /// Initializes the weight vectors for this model using the He (et al) initialization method
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

  pub fn 
  init_network_glorot (&mut self)
  {
    let mut rng = StdRng::from_entropy();

    let limit_w1 = (6.0 / self.w1.len() as f64).sqrt();
    let distribution_w1 = Uniform::from(-limit_w1..limit_w1);

    for val in self.w1.iter_mut() {
        *val = rng.sample(distribution_w1);
    }

    let limit_w2 = (6.0 / self.w2.len() as f64).sqrt();
    let distribution_w2 = Uniform::from(-limit_w2..limit_w2);

    for val in self.w2.iter_mut() {
        *val = rng.sample(distribution_w2);
    }
  }

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
}
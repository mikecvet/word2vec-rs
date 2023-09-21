use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Uniform, Normal};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Error, Read, Write};
use std::ops::Sub;

pub use crate::metadata::Metadata;
pub use crate::subsampler::SubSampler;

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

// Generates a vector of zeroes with the exception of the specified index, which is set to 1.0 ("one-hot encoding").
/// This encodes the presence of a specific categorical variable; in this case, a string token present in our 
/// vocabulary set.
pub fn 
encode_to_vector (indx: usize, n: usize) -> Array2<f64>
{
  let mut a = Array2::zeros((1, n));
  a[[0, indx]] = 1.0;
  a
}

/// The softmax function converts a vector of K real numbers into a probability distribution of 
/// K possible outcomes
/// 
/// https://en.wikipedia.org/wiki/Softmax_function
fn 
softmax (x: &Array2<f64>) -> Array2<f64> 
{
  let x_exp = x.mapv(f64::exp);
  let sum_exp = x_exp.sum_axis(Axis(1)).insert_axis(Axis(1));
  &x_exp / &sum_exp
}

/// The cross-entropy loss measures the dissimilarity between the predicted probabilities 
/// (from the softmax layer) and the true labels. In essence, it calculates how well the 
/// predicted probabilities match up with the actual labels (here, identified by Y)
/// 
/// The idea is that for the correct class, the model should assign a high probability, 
/// and for incorrect classes, it should assign a low probability. The cross-entropy loss 
/// quantifies how well the model does this.
/// 
/// https://en.wikipedia.org/wiki/Cross-entropy
fn 
cross_entropy (p: &Array2<f64>, q: &Array2<f64>) -> f64 
{
  let mut ce: f64 = 0.0;
  for iter in p.iter().zip(q.iter()) {
    ce += iter.0.log2() * iter.1;
  }

  -ce
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

    for val in self.w2.iter_mut() {
        *val = rng.sample(distribution_w2);
    }
  }

  /// Runs forward propagation against this neural network. Collects predicted output
  pub fn 
  forward_propagation (&self, x: &Array2<f64>) -> (Array2<f64>, Array2<f64>) 
  {
    let a1 = x.dot(&self.w1);
    let a2 = a1.dot(&self.w2);
    let probabilities = softmax(&a2);
    (a1, probabilities)
  }

  /// Runs back propagation aginst this neural network. Compute the gradient of the 
  /// loss function with respect to the weights and biases. This gradient is then 
  /// used to update the weights and biases using gradient descent.
  pub fn 
  back_propagation (&mut self, training_data: TrainingData, rate: f64) -> f64 
  {
    // Run prediction
    let (a, probabilities) = self.forward_propagation(&training_data.x);

    // Compute the cross-entropy loss from the forward propagation step
    let ce = cross_entropy(&probabilities, &training_data.y);

    // Compute loss gradient
    let p0 = probabilities.sub(training_data.y);
    let a0 = (a.t()).dot(&p0);
    let p1 = p0.dot(&self.w2.t());
    let a1 = training_data.x.t().dot(&p1);

    // Adjust the model after applying gradient descent values
    self.w1 = self.w1.clone().sub(rate * a1);
    self.w2 = self.w2.clone().sub(rate * a0);

    ce
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

      // The token was not found in the map
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

  const EPSILON: f64 = 1e-10;

  fn is_approx_1(sum: f64) -> bool {
      (1.0 - EPSILON..=1.0 + EPSILON).contains(&sum)
  }

  fn is_approx_0(val: f64) -> bool {
      (-EPSILON..=EPSILON).contains(&val)
  }

  #[test]
  fn test_encode ()
  {
      let v1 = encode_to_vector(1, 10);
      let v2 = encode_to_vector(10, 100);
      let v3 = encode_to_vector(64, 256);

      assert!(v1[[0, 0]] == 0.0);
      assert!(v1[[0, 1]] == 1.0);
      
      assert!(v2[[0, 0]] == 0.0);
      assert!(v2[[0, 10]] == 1.0);

      assert!(v3[[0, 0]] == 0.0);
      assert!(v3[[0, 64]] == 1.0);
  }
  
  #[test]
  fn test_softmax_basic () 
  {
      // 3x3 matrix filled with 1s
      let matrix = Array2::from_elem((3, 3), 1.0); 
  
      let sm_matrix = softmax(&matrix);
      for sum in sm_matrix.sum_axis(Axis(1)) {
          assert!(is_approx_1(sum));
      }
  }
  
  #[test]
  fn test_softmax_stability () 
  {
      // 3x3 matrix with random values between 0 and 10
      let matrix = Array2::random((3, 3), Uniform::new(0.0, 10.0)); 
  
      let sm_matrix = softmax(&matrix);
      for sum in sm_matrix.sum_axis(Axis(1)) {
          assert!(is_approx_1(sum));
      }
  }

  #[test]
  fn test_cross_entropy_exact_match ()
  {
      // 3x3 matrix filled with 1s
      let matrix = Array2::from_elem((3, 3), 1.0); 

      let ce_value = cross_entropy(&matrix.clone(), &matrix);
      assert!(is_approx_0(ce_value));
  }

  #[test]
  fn test_cross_entropy_completely_wrong () 
  {
      let matrix_a = Array2::from_elem((3, 3), 1.0);
      let matrix_b = Array2::from_elem((3, 3), 0.0); // Opposite values

      let ce_value = cross_entropy(&matrix_a, &matrix_b);
      assert!(ce_value.is_sign_negative());
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
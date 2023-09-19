use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::collections::HashMap;
use std::ops::Sub;

pub use crate::metadata::Metadata;
pub use crate::model::*;
pub use crate::sample::SubSampler;

/// Generates a vector of zeroes with the exception of the specified index, which is set to 1.0 ("one-hot encoding").
/// This encodes the presence of a specific categorical variable; in this case, a string token present in our 
/// vocabulary set.
pub fn 
encode (indx: usize, n: usize) -> Array2<f64>
{
  let mut a = Array2::zeros((1, n));
  a[[0, indx]] = 1.0;
  a
}

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

        let a = encode(*token_to_index.get(&(tokens[i as usize])).unwrap(), vocabulary_size);
        let b = encode(*token_to_index.get(&(tokens[j as usize])).unwrap(), vocabulary_size);

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
  window_size: i32,
  epochs: usize,
  learning_rate: f64,
  print: bool
) {
    // The SubSampler is used to nondeterministically remove frequent terms from the training data set. 
    // Since training data is recomputed during each epoch, each round will be trained against slight variations
    // in the training data matrices.
    let sampler = SubSampler::new(metadata.token_counts.clone(), metadata.tokens.len() as f64);

    if print && epochs > 0 {
      println!("entopy per epoch:");
    }

    // epochs
    for _ in 0..epochs 
      {
        let td = build_skip_gram_training_data(
          &metadata.tokens, 
          &metadata.token_to_index, 
          &sampler, 
          metadata.vocab_size, 
          window_size
        );

        // Run backpropagation and possibly record the cross-entropy error from this process
        let ce = back_propagation(model, td, learning_rate);

        if print {
            println!("{:.3}", ce);
        }
      }
}

/// Runs forward propagation against this neural network.
pub fn 
forward_propagation (model: &Model, x: &Array2<f64>) -> (Array2<f64>, Array2<f64>) 
{
  let a1 = x.dot(&model.w1);
  let a2 = a1.dot(&model.w2);
  let probabilities = softmax(&a2);
  (a1, probabilities)
}

/// Runs back propagation aginst this neural network. 
pub fn 
back_propagation (model: &mut Model, training_data: TrainingData, rate: f64) -> f64 
{
  let (a, probabilities) = forward_propagation(model, &training_data.x);

  // Compute the cross-entropy loss from the forward propagation step
  let ce = cross_entropy(&probabilities, &training_data.y);

  let p0 = probabilities.sub(training_data.y);
  let a0 = (a.t()).dot(&p0);
  let p1 = p0.dot(&model.w2.t());
  let a1 = training_data.x.t().dot(&p1);

  // Update the model with new weights
  model.w1 = model.w1.clone().sub(rate * a1);
  model.w2 = model.w2.clone().sub(rate * a0);

  ce
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
/// (from the softmax) and the true labels. In essence, it calculates how well the 
/// predicted probabilities match up with the actual labels (here, identified by Y)
/// 
/// The idea is that for the correct class, the model should assign a high probability, 
/// and for incorrect classes, it should assign a low probability. The cross-entropy loss 
/// quantifies how well the model does this.
/// 
/// https://en.wikipedia.org/wiki/Cross-entropy
fn 
cross_entropy (z: &Array2<f64>, y: &Array2<f64>) -> f64 
{
  let mut ce: f64 = 0.0;
  for iter in z.iter().zip(y.iter()) {
    ce += iter.0.log2() * iter.1;
  }

  -ce
}

#[cfg(test)]
mod tests 
{
    use ndarray_rand::rand_distr::num_traits::Float;

    use super::*;

    const EPSILON: f64 = 1e-10;

    fn is_approx_1(sum: f64) -> bool {
        (1.0 - EPSILON..=1.0 + EPSILON).contains(&sum)
    }

    fn is_approx_0(val: f64) -> bool {
        (-EPSILON..=EPSILON).contains(&val)
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
    fn test_encode ()
    {
        let v1 = encode(1, 10);
        let v2 = encode(10, 100);
        let v3 = encode(64, 256);

        assert!(v1[[0, 0]] == 0.0);
        assert!(v1[[0, 1]] == 1.0);
        
        assert!(v2[[0, 0]] == 0.0);
        assert!(v2[[0, 10]] == 1.0);

        assert!(v3[[0, 0]] == 0.0);
        assert!(v3[[0, 64]] == 1.0);
    }
}

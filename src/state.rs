pub use crate::metadata::*;
pub use crate::model::*;

pub struct State {
  pub model: Model,
  pub training_data: TrainingData,
  pub metadata: Metadata,
  pub embeddings_size: usize,
  pub window_size: usize
}

impl State {
    pub fn 
    new (
      model: Model, 
      training_data: TrainingData, 
      metadata: Metadata, 
      embeddings_size: usize, 
      window_size: usize
    ) -> Self
    {
        State {
          model: model,
          training_data: training_data,
          metadata: metadata,
          embeddings_size: embeddings_size,
          window_size: window_size
        }
    }
}
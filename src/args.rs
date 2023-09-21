use std::fs;

const DEFAULT_EMBEDDINGS_SIZE: usize = 128;
const DEFAULT_EPOCHS: usize = 65;
// Useful ranges from [0.001 .. 0.0001]
const DEFAULT_LEARNING_RATE: f64 = 0.001;
const DEFAULT_WINDOW_SIZE: i32 = 4;

pub struct HyperParams {
  pub embeddings_size: usize,
  pub learning_rate: f64,
  pub num_epochs: usize,
  pub window_size: i32
}

pub struct Args {
  pub analogy: Option<(String, String, String)>,
  pub model_load_path: Option<String>,
  pub save_model: bool,
  pub predict: Option<String>,
  pub print_entropy: bool,
  pub train: bool,
  pub text: String,
  pub hyper_params: HyperParams
}

impl HyperParams {
  pub fn 
  new (
    embeddings_size_opt: Option<String>,
    learning_rate_opt: Option<String>,
    num_epochs_opt: Option<String>,
    window_size_opt: Option<String>
  ) -> Self 
  {
    let embeddings_size = match embeddings_size_opt {
      Some(embeddings_size) => {
        embeddings_size.parse::<usize>().expect("embeddings size must be an integer")
      }
      _ => DEFAULT_EMBEDDINGS_SIZE
    };

    let learning_rate = match learning_rate_opt {
      Some(learning_rate) => {
        learning_rate.parse::<f64>().expect("learning rate must be a float")
      }
      _ => DEFAULT_LEARNING_RATE
    };

    let num_epochs = match num_epochs_opt {
      Some(num_epochs) => {
        num_epochs.parse::<usize>().expect("number of epochs must be an integer")
      }
      _ => DEFAULT_EPOCHS
    };

    let window_size = match window_size_opt {
      Some(window_size) => {
        window_size.parse::<i32>().expect("window size must be a positive integer")
      }
      _ => DEFAULT_WINDOW_SIZE
    };

    HyperParams { 
      embeddings_size: embeddings_size, 
      learning_rate: learning_rate, 
      num_epochs: num_epochs, 
      window_size: window_size 
    }
  }
}

impl Args {
  pub fn 
  new (
    analogy: Option<String>,
    input_path: Option<String>,
    model_load_path: Option<String>,
    save_model: Option<bool>,
    predict: Option<String>,
    print_entropy: Option<bool>,
    no_train: Option<bool>,
    embeddings_size_opt: Option<String>,
    learning_rate_opt: Option<String>,
    num_epochs_opt: Option<String>,
    window_size_opt: Option<String>
  ) -> Self 
  {
    let hyper_params = HyperParams::new(
      embeddings_size_opt, 
      learning_rate_opt, 
      num_epochs_opt, 
      window_size_opt
    );

    let text: String = match input_path {
      Some(path) => {
          fs::read_to_string(path).expect("unable to open file!").parse().unwrap()
      },
      _ => {
          panic!("Path to input text file must be provided via --input <path>");
      }
    };

    Args { 
      analogy: extract_analogy(&analogy), 
      model_load_path: model_load_path, 
      save_model: save_model.unwrap_or(true), 
      predict: predict, 
      print_entropy: print_entropy.unwrap_or(false),
      train: !no_train.unwrap_or(false),
      text: text,
      hyper_params: hyper_params
    }
  }
}

fn
extract_analogy (analogy_opt: &Option<String>) -> Option<(String, String, String)>
{
  match analogy_opt {
    Some(a) => {
      let parts = a.split(",");
      let mut tmp: Vec<String> = Vec::new();
      for part in parts {
        tmp.push(part.to_string());
      }

      if tmp.len() != 3 {
        panic!("--analogy requires an argument of three tokens separated by commas; for example, << a,b,c >>")
      }

      Some((tmp.get(0).unwrap().to_owned(), tmp.get(1).unwrap().to_owned(), tmp.get(2).unwrap().to_owned()))
    }
    _ => None
  }
}
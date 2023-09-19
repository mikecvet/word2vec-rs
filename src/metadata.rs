use regex::Regex;
use std::collections::{HashMap, HashSet};

pub struct Metadata {
  pub tokens: Vec<String>,
  pub token_to_index: HashMap<String, usize>,
  pub index_to_token: HashMap<usize, String>,
  pub token_counts: HashMap<String, usize>,
  pub vocab_size: usize
}

impl Metadata 
{
  pub fn 
  init (text: &str) -> Self 
  {
    let stop_words: HashSet<String> = 
    vec![
      "the", "in", "of", "a", "am", "an", "and", "any", "are", "as", "at", "because", "i", "by", "it", "from", 
      "is", "has", "on", "over", "s", "such", "was", "which", "with", "for", "to", "that"
    ].iter()
      .cloned()
      .map(String::from)
      .collect();
    let tokenizer_regex = Regex::new(r"[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*").unwrap();
    let tokens = tokenize(&tokenizer_regex, text, stop_words);

    let mut token_to_index: HashMap<String, usize> = HashMap::new();
    let mut index_to_token: HashMap<usize, String> = HashMap::new();
    let mut token_counts: HashMap<String, usize> = HashMap::new();
  
    let mut indx = 0;
  
    for token in tokens.clone() {
      if !token_to_index.contains_key(&token.clone()) {
          token_to_index.insert(token.clone(), indx);
          index_to_token.insert(indx, token.clone());
          indx += 1;
      }
  
      match token_counts.get(&token) {
        Some(count) => token_counts.insert(token, count + 1),
        _ => token_counts.insert(token, 1)
      };
    }

    let vocab_size = token_to_index.len();

    Metadata { 
      tokens:tokens, 
      token_to_index: token_to_index, 
      index_to_token: index_to_token, 
      token_counts: token_counts,
      vocab_size: vocab_size
    }
  }
}

/// Breaks the given text up into a vector of token strings, based on the provided Regex. Lowercases inputs, so
/// all returned tokens are also lowercase.
fn 
tokenize (regex: &Regex, text: &str, stop_words: HashSet<String>) -> Vec<String> 
{
  regex
    .find_iter(text.to_lowercase().as_str())
    .filter_map(|token| token.as_str().parse().ok())
    .filter(|token| !stop_words.contains(token))
    .collect()
}

#[cfg(test)]
mod tests 
{
  use super::*;

  
  #[test]
  fn test_stop_words () 
  {
    let metadata = Metadata::init("The as it over a i and such to that which was");
    let v: Vec<String> = vec![];

    assert!(metadata.tokens.eq(&v));
  }

  #[test]
  fn test_init_tokenization () 
  {
    let metadata = Metadata::init("The quick, brown fox jumps over the lazy dog.");
    let v = vec!["quick", "brown", "fox", "jumps", "lazy", "dog"];

    assert!(metadata.tokens.eq(&v));
  }

  #[test]
  fn test_regex_dash () 
  {
    let metadata = Metadata::init("token-and-dash");
    let v = vec!["token", "dash"];

    println!("output {:?}", metadata.tokens);

    assert!(metadata.tokens.eq(&v));
  }

  #[test]
  fn test_vocab_size () 
  {
    let metadata = Metadata::init("The quick brown fox jumps over the lazy, lazy dog.");

    assert!(metadata.vocab_size == 6);
  }

  #[test]
  fn test_token_counts () 
  {
    let metadata = Metadata::init("The quick brown fox jumps over the lazy, lazy dog.");

    assert!(metadata.token_counts.get("quick").unwrap().eq(&1));
    assert!(metadata.token_counts.get("lazy").unwrap().eq(&2));
    assert!(metadata.token_counts.get("dog").unwrap().eq(&1));
  }

  #[test]
  fn test_token_to_index () 
  {
    let metadata = Metadata::init("The quick brown fox jumps over the lazy dog.");

    // ["quick", "brown", "fox", "jumps", "lazy", "dog"];

    assert!(metadata.token_to_index.get("quick").unwrap().eq(&0));
    assert!(metadata.token_to_index.get("lazy").unwrap().eq(&4));
    assert!(metadata.token_to_index.get("dog").unwrap().eq(&5));
  }

  #[test]
  fn test_index_to_token () 
  {
    let metadata = Metadata::init("The quick brown fox jumps over the lazy dog.");

    // ["quick", "brown", "fox", "jumps", "lazy", "dog"];

    assert!(metadata.index_to_token.get(&0).unwrap().eq("quick"));
    assert!(metadata.index_to_token.get(&4).unwrap().eq("lazy"));
    assert!(metadata.index_to_token.get(&5).unwrap().eq("dog"));
  }
}
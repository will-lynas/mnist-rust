use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Write;
use std::iter::zip;

use ndarray::Array1;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct MnistSample {
    pub label: Array1<f64>,
    pub image: Array1<f64>,
}

pub fn save_mnist_samples(data: Vec<MnistSample>, file_name: &str) {
    let file = File::create(file_name).unwrap();
    let mut writer = BufWriter::new(file);
    let serialized = bincode::serialize(&data).unwrap();
    writer.write_all(&serialized).unwrap();
}

pub fn load_mnist_samples(file_name: &str) -> Vec<MnistSample> {
    let file = File::open(file_name).unwrap();
    let reader = BufReader::new(file);
    bincode::deserialize_from(reader).unwrap()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub struct Network {
    #[allow(dead_code)]
    num_layers: usize,
    #[allow(dead_code)]
    sizes: Vec<usize>,
    biases: Vec<Array1<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
    pub fn new(sizes: Vec<usize>) -> Self {
        let num_layers = sizes.len();
        let biases = sizes
            .iter()
            .skip(1)
            .map(|&size| Array1::random(size, Normal::new(0.0, 1.0).unwrap()))
            .collect();
        let weights = zip(sizes.iter(), sizes.iter().skip(1))
            .map(|(&x, &y)| Array2::random((y, x), Normal::new(0.0, 1.0).unwrap()))
            .collect();

        Network {
            num_layers,
            sizes,
            biases,
            weights,
        }
    }

    pub fn feedforward(&self, a: Array1<f64>) -> Array1<f64> {
        zip(self.biases.iter(), self.weights.iter())
            .fold(a, |a, (b, w)| (w.dot(&a) + b).mapv(sigmoid))
    }
}

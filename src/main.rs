use std::iter::zip;

use ndarray::Array1;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

#[allow(dead_code)]
struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Array1<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
    fn new(sizes: Vec<usize>) -> Self {
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
}

fn main() {
    let _network = Network::new(vec![3, 4, 4, 3]);
}

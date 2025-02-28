use std::iter::zip;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

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

    fn feedforward(&self, a: Array1<f64>) -> Array1<f64> {
        zip(self.biases.iter(), self.weights.iter())
            .fold(a, |a, (b, w)| (w.dot(&a) + b).mapv(sigmoid))
    }
}

fn main() {
    let network = Network::new(vec![3, 4, 4, 3]);
    let a = array![1.0, 2.0, 3.0];
    let b = network.feedforward(a);
    println!("{:?}", b);
}

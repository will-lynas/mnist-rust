use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Write;
use std::iter::zip;

use ndarray::Array1;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use ndarray_stats::QuantileExt;
use rand::seq::SliceRandom;
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

fn sigmoid_prime(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

fn cost_derivative(output_activations: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
    output_activations - y
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

    pub fn feedforward(&self, a: &Array1<f64>) -> Array1<f64> {
        zip(self.biases.iter(), self.weights.iter())
            .fold(a.clone(), |a, (b, w)| (w.dot(&a) + b).mapv(sigmoid))
    }

    pub fn sgd(
        &mut self,
        mut training_data: Vec<&MnistSample>,
        epochs: usize,
        mini_batch_size: usize,
        eta: f64,
        test_data: Option<&[MnistSample]>,
    ) {
        (0..epochs).for_each(|epoch| {
            training_data.shuffle(&mut rand::rng());
            training_data
                .chunks(mini_batch_size)
                .for_each(|mini_batch| {
                    self.update_mini_batch(mini_batch, eta);
                });
            if let Some(test_data) = test_data {
                let accuracy = self.evaluate(test_data);
                println!("Epoch {} : {} / {}", epoch, accuracy, test_data.len());
            } else {
                println!("Epoch {} complete", epoch);
            }
        });
    }

    fn update_mini_batch(&mut self, mini_batch: &[&MnistSample], eta: f64) {
        let mut nabla_b: Vec<_> = self
            .biases
            .iter()
            .map(|b| Array1::<f64>::zeros(b.dim()))
            .collect();
        let mut nabla_w: Vec<_> = self
            .weights
            .iter()
            .map(|w| Array2::<f64>::zeros(w.dim()))
            .collect();

        mini_batch
            .iter()
            .map(|sample| self.backprop(sample))
            .for_each(|(nabla_b_i, nabla_w_i)| {
                nabla_b
                    .iter_mut()
                    .zip(nabla_b_i.iter())
                    .for_each(|(nabla_b, nabla_b_i)| {
                        *nabla_b += nabla_b_i;
                    });
                nabla_w
                    .iter_mut()
                    .zip(nabla_w_i.iter())
                    .for_each(|(nabla_w, nabla_w_i)| {
                        *nabla_w += nabla_w_i;
                    });
            });

        self.biases
            .iter_mut()
            .zip(nabla_b.iter())
            .for_each(|(b, nabla_b)| {
                *b -= &(nabla_b * eta / mini_batch.len() as f64);
            });
        self.weights
            .iter_mut()
            .zip(nabla_w.iter())
            .for_each(|(w, nabla_w)| {
                *w -= &(nabla_w * eta / mini_batch.len() as f64);
            });
    }

    fn backprop(&self, mnist_sample: &MnistSample) -> (Vec<Array1<f64>>, Vec<Array2<f64>>) {
        let mut nabla_b: Vec<_> = self
            .biases
            .iter()
            .map(|b| Array1::<f64>::zeros(b.dim()))
            .collect();
        let mut nabla_w: Vec<_> = self
            .weights
            .iter()
            .map(|w| Array2::<f64>::zeros(w.dim()))
            .collect();

        let mut activations: Vec<Array1<f64>> = vec![mnist_sample.image.clone()];
        let mut zs: Vec<Array1<f64>> = vec![];

        zip(self.biases.iter(), self.weights.iter()).for_each(|(b, w)| {
            let z = w.dot(activations.last().unwrap()) + b;
            activations.push(z.mapv(sigmoid));
            zs.push(z);
        });

        todo!()
    }

    pub fn evaluate(&self, test_data: &[MnistSample]) -> usize {
        test_data
            .iter()
            .filter(|MnistSample { label, image }| {
                // This will be changed later
                self.feedforward(image).argmax() == label.argmax()
            })
            .count()
    }
}

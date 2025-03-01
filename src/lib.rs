use std::iter::zip;

use indicatif::ProgressIterator;
use ndarray::Array1;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use ndarray_stats::QuantileExt;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

pub mod utils;

#[derive(Serialize, Deserialize)]
pub struct MnistSample {
    pub label: Array1<f64>,
    pub image: Array1<f64>,
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
    num_layers: usize,
    biases: Vec<Array1<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
    pub fn new(sizes: Vec<usize>) -> Self {
        let biases = sizes
            .iter()
            .skip(1)
            .map(|&size| Array1::random(size, Normal::new(0.0, 1.0).unwrap()))
            .collect();
        let weights = zip(sizes.iter(), sizes.iter().skip(1))
            .map(|(&x, &y)| Array2::random((y, x), Normal::new(0.0, 1.0).unwrap()))
            .collect();

        Network {
            num_layers: sizes.len(),
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
            println!("Epoch {} of {}", epoch + 1, epochs);
            training_data.shuffle(&mut rand::rng());
            training_data
                .chunks(mini_batch_size)
                .progress()
                .for_each(|mini_batch| {
                    self.update_mini_batch(mini_batch, eta);
                });
            if let Some(test_data) = test_data {
                let accuracy = self.evaluate(test_data);
                println!("  Accuracy: {} / {}", accuracy, test_data.len());
            } else {
                println!("  Complete");
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
        let mut nabla_b = Vec::new();
        let mut nabla_w = Vec::new();

        let mut activations: Vec<Array1<f64>> = vec![mnist_sample.image.clone()];
        let mut zs: Vec<Array1<f64>> = vec![];

        // Forward pass
        for (b, w) in zip(&self.biases, &self.weights) {
            let z = w.dot(activations.last().unwrap()) + b;
            let activation = z.mapv(sigmoid);
            zs.push(z);
            activations.push(activation);
        }

        // Last layer
        let sp = zs.last().unwrap().mapv(sigmoid_prime);
        let mut delta = cost_derivative(activations.last().unwrap(), &mnist_sample.label) * sp;
        nabla_b.push(delta.clone());

        let delta2d = delta.clone().insert_axis(ndarray::Axis(1)); // Column vector
        let a_t = activations[activations.len() - 2]
            .clone()
            .insert_axis(ndarray::Axis(0)); // Row vector
        let prod = delta2d.dot(&a_t);
        nabla_w.push(prod);

        // Backward pass
        for l in 2..self.num_layers {
            let z = zs[zs.len() - l].clone();
            let sp = z.mapv(sigmoid_prime);
            delta = self.weights[self.weights.len() - l + 1].t().dot(&delta) * sp;
            let delta2d = delta.clone().insert_axis(ndarray::Axis(1)); // Column vector
            nabla_b.push(delta.clone());
            let a_t = activations[activations.len() - l - 1]
                .clone()
                .insert_axis(ndarray::Axis(0));
            nabla_w.push(delta2d.dot(&a_t));
        }

        nabla_b.reverse();
        nabla_w.reverse();

        (nabla_b, nabla_w)
    }

    pub fn evaluate(&self, test_data: &[MnistSample]) -> usize {
        test_data
            .iter()
            .filter(|MnistSample { label, image }| {
                self.feedforward(image).argmax() == label.argmax()
            })
            .count()
    }
}

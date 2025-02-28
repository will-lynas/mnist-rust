use mnist_rust::{Network, load_mnist_samples};
use ndarray::array;

fn main() {
    let network = Network::new(vec![3, 4, 4, 3]);
    let a = array![1.0, 2.0, 3.0];
    let b = network.feedforward(a);
    println!("{:?}", b);

    let mnist_train = load_mnist_samples("data/mnist_train_processed.bin");
    println!("{:?}", mnist_train.len());
}

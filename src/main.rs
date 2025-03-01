use mnist_rust::{Network, load_mnist_samples};

fn main() {
    let mnist_train = load_mnist_samples("data/mnist_train_processed.bin");
    let mnist_test = load_mnist_samples("data/mnist_test_processed.bin");

    let mut network = Network::new(vec![28 * 28, 30, 10]);
    let epochs = 30;
    let mini_batch_size = 10;
    let eta = 3.0;
    network.sgd(
        mnist_train.iter().collect(),
        epochs,
        mini_batch_size,
        eta,
        Some(&mnist_test),
    );
}

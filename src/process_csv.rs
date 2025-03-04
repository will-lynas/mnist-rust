use csv::ReaderBuilder;
use ndarray::Array1;

use mnist_rust::{MnistSample, utils::save_mnist_samples};

fn main() {
    process_csv("mnist_train");
    process_csv("mnist_test");
}

fn process_csv(file_name: &str) {
    let csv_file_name = format!("data/{}.csv", file_name);
    println!("Processing {}", csv_file_name);
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_path(csv_file_name)
        .unwrap();

    let dataset: Vec<MnistSample> = reader
        .records()
        .map(|record| {
            let record = record.unwrap();

            let label: usize = record[0].parse().unwrap();
            let mut label_vec = Array1::zeros(10);
            label_vec[label] = 1.0;

            let image = record
                .iter()
                .skip(1)
                .map(|val| val.parse::<f64>().unwrap() / 255.0)
                .collect();

            MnistSample {
                label: label_vec,
                image,
            }
        })
        .collect();

    let output_file = format!("data/{}_processed.bin", file_name);
    println!("  Saving to {}", output_file);
    save_mnist_samples(dataset, &output_file);
}

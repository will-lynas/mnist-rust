use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
};

use crate::MnistSample;

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

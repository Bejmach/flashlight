use flashlight::NeuralNetwork;

fn main() {
    let nnetwork: NeuralNetwork = NeuralNetwork::new(vec!{3, 9, 9, 1});

    println!("{}", nnetwork);
}

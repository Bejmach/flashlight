use flashlight::NeuralNetwork;

fn main() {

    let network_layut: Vec<usize> = vec!{16, 9, 4, 8, 9, 1};
    
    print!("Network layout: ");
    for var in network_layut.iter(){
        print!("{}, ", var);
    }
    println!("\n");

    let nnetwork: NeuralNetwork = NeuralNetwork::new(vec!{16, 9, 4, 8, 9, 1});

    println!("{}", nnetwork);
}

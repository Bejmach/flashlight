use flashlight::NeuralNetwork;
use flashlight_tensor::prelude::*;


fn default_propagation(){
    let network_layut: Vec<u32> = vec!{3, 4, 4, 2};
    
    print!("Network layout: ");
    for var in network_layut.iter(){
        print!("{}, ", var);
    }
    println!("\n");

    let nnetwork: NeuralNetwork = NeuralNetwork::new(network_layut, 1.0, 1.0);

    let input_params: Vec<f32> = vec!(1.3, 3.1, 6.5);

    println!("{}", nnetwork);

    let propagation_output: Vec<f32> = nnetwork.forward_propagation(&input_params).unwrap().get_data().to_vec();

    print!("Propatation output: {{");
    for arg in propagation_output.iter(){
        print!("{}, ", arg);
    }
    println!("}}");
}

fn main() {
    default_propagation();
}

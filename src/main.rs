use flashlight::NeuralNetwork;
use flashlight::math::tensor::Tensor;


fn default_propagation(){
    let network_layut: Vec<usize> = vec!{3, 4, 4, 2};
    
    print!("Network layout: ");
    for var in network_layut.iter(){
        print!("{}, ", var);
    }
    println!("\n");

    let nnetwork: NeuralNetwork = NeuralNetwork::new(network_layut, 1.0, 1.0);

    let input_params: Vec<f32> = vec!(1.3, 3.1, 6.5);

    println!("{}", nnetwork);

    let propagation_output: Vec<f32> = nnetwork.forward_propagation(input_params).unwrap();

    print!("Propatation output: {{");
    for arg in propagation_output.iter(){
        print!("{}, ", arg);
    }
    println!("}}");
}

fn main() {
    let tensor: Tensor<f32> = Tensor::rand_f32(vec![3, 3, 3], 100.0);

    let vector = tensor.vector(vec![1, 0]);

    println!("Tensor: ");
    for arg in tensor.data{
        print!("{}, ", arg);
    }
    println!("\n\nVector: ");

    for arg in vector.unwrap().data{
        print!("{}, ", arg);
    }
    println!("");
}

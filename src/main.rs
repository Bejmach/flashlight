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

    let vector1 = tensor.vector(&[0, 0]).unwrap();
    let vector2 = tensor.vector(&[0, 1]).unwrap();

    
    println!("Tensor: ");
    for arg in tensor.data{
        print!("{}, ", arg);
    }
    println!("\n\nVector1: ");
    for arg in vector1.sizes[..].iter(){
        print!("{}, ", arg);
    }
    println!("");
    for arg in vector1.data[..].iter(){
        print!("{}, ", arg);
    }
    println!("");
    println!("\n\nVector2: ");
    for arg in vector2.sizes[..].iter(){
        print!("{}, ", arg);
    }
    println!("");
    for arg in vector2.data[..].iter(){
        print!("{}, ", arg);
    }
    println!("");

    let dot: f32 = vector1.dot_product(&vector2).unwrap();

    println!("Dot: {}", dot);

    
}

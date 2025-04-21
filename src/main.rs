use flashlight::prelude::*;
#[allow(unused)]
use flashlight_tensor::prelude::*;

fn main() {

    let model: Model = Model::new(&vec!{2, 3, 3, 1}, 1.0, 1.0);

    println!("{}\n\n{}", model.aesthetic_to_string(), model.full_to_string());

    let input_data: Tensor<f32> = Tensor::from_data(&[50.0, 150.0, 100.0, 220.0, 75.0, 190.0], &[3,2]).unwrap();
    let prediction = model.full_forward_propagation(&input_data).unwrap();

    println!("\nPredictions: ");
    for i in 0..prediction.len(){
        println!("{}\n\n", prediction[i].matrix_to_string().unwrap());
    }
}

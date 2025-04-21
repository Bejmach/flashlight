use flashlight::prelude::*;
#[allow(unused)]
use flashlight_tensor::prelude::*;

fn main() {

    let model: Model = Model::new(&vec!{2, 3, 3, 1}, 1.0, 1.0);

    let input_data: Tensor<f32> = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
    let full_propagation: Vec<Tensor<f32>> = model.full_forward_propagation(&input_data).unwrap();

    for i in 0..full_propagation.len(){
        println!("{}, {}\n{}\n", full_propagation[i].get_sizes()[0], full_propagation[i].get_sizes()[1], full_propagation[i].matrix_to_string().unwrap());
    }
}

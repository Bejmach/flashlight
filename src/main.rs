use flashlight::prelude::*;
#[allow(unused)]
use flashlight_tensor::prelude::*;

fn main() {

    let model: Model = Model::new(&vec!{2, 3, 3, 1}, 1.0, 1.0);

    println!("{}\n\n{}", model.aesthetic_to_string(), model.full_to_string());

    let input_data: Tensor<f32> = Tensor::from_data(&[50.0, 150.0, 100.0, 220.0, 75.0, 190.0], &[3,2]).unwrap().matrix_transpose().unwrap();
    let expected_output: Tensor<f32> = Tensor::from_data(&[0.0, 1.0, 0.0], &[3, 1]).unwrap().matrix_transpose().unwrap();

    model.cross_entropy_backprop_loop(&input_data, &expected_output);

}

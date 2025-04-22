use flashlight::prelude::*;
#[allow(unused)]
use flashlight_tensor::prelude::*;

fn main() {

    let mut model: Model = Model::new(&vec!{2, 3, 3, 1}, 1.0, 1.0);

    println!("Model at start:\n{}\n__________________________", model.full_to_string());

    let input_data: Tensor<f32> = Tensor::from_data(&[10.0, 15.0, 25.0, 30.0, 80.0, 20.0], &[3,2]).unwrap().matrix_transpose().unwrap();
    let expected_output: Tensor<f32> = Tensor::from_data(&[sigmoid((10.0+15.0)/100.0), sigmoid((25.0+30.0)/100.0), sigmoid((80.0+20.0)/100.0)], &[3, 1]).unwrap().matrix_transpose().unwrap();

    let outpud_data = model.full_forward_propagation(&input_data).unwrap();
    println!("first Cost: {}", cross_entropy_cost(&outpud_data[outpud_data.len()-1], &expected_output).unwrap());

    for _i in 0..100000{
        model.cross_entropy_backprop_loop(&input_data, &expected_output, 0.05);
        let outpud_data = model.full_forward_propagation(&input_data).unwrap();
        println!("last Cost: {}", cross_entropy_cost(&outpud_data[outpud_data.len()-1], &expected_output).unwrap());
    }

    println!("Model at end:\n{}\n__________________________", model.full_to_string());

    let outpud_data = model.full_forward_propagation(&input_data).unwrap();

    let fill_ones = Tensor::fill(1.0, outpud_data[outpud_data.len()-1].get_sizes());
    let real_output = outpud_data[outpud_data.len()-1].tens_div(&fill_ones.tens_sub(&outpud_data[outpud_data.len()-1]).unwrap()).unwrap().log().mult(100.0);

    println!("Input_data:\n{}\n\nExpected_data:\n{}\n\nOutput_data:\n{}\n\nDecompiled_data:\n{}", input_data.matrix_to_string().unwrap(), expected_output.matrix_to_string().unwrap(), outpud_data[outpud_data.len()-1].matrix_to_string().unwrap(), real_output.matrix_to_string().unwrap());
    println!("last Cost: {}", cross_entropy_cost(&outpud_data[outpud_data.len()-1], &expected_output).unwrap());
}

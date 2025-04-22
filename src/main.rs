use flashlight::prelude::*;
#[allow(unused)]
use flashlight_tensor::prelude::*;

fn f32_to_bits(f: f32) -> Vec<bool> {
    let bits = f.to_bits(); // Convert f32 to its byte representation as u32
    (0..32)
        .map(|i| ((bits >> (31 - i)) & 1) == 1) // Extract each bit from MSB to LSB
        .collect()
}

fn bits_to_f32(bits: &[bool]) -> f32 {
    assert_eq!(bits.len(), 32, "Exactly 32 bits required");
    let mut n: u32 = 0;
    for (i, &bit) in bits.iter().enumerate() {
        if bit {
            n |= 1 << (31 - i); // Set the corresponding bit in u32
        }
    }
    f32::from_bits(n) // Convert u32 to f32
}

fn bools_to_floats(bits: &[bool]) -> Vec<f32> {
    bits.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect()
}
fn floats_to_bools(floats: &[f32]) -> Vec<bool> {
    floats.iter().map(|&f| f > 0.5).collect()
}

fn main() {

    let mut model: Model = Model::new(&vec!{2, 3, 3, 32}, 1.0, 1.0);

    //println!("Model at start:\n{}\n__________________________", model.full_to_string());

    let input_data: Tensor<f32> = Tensor::from_data(&[10.0, 15.0, 25.0, 30.0, 80.0, 20.0], &[3,2]).unwrap().matrix_transpose().unwrap();
    let mut output_vec: Vec<f32> = Vec::with_capacity(96);

    let mut output_1 = bools_to_floats(&f32_to_bits(25.0));
    output_vec.append(&mut output_1);
    let mut output_2 = bools_to_floats(&f32_to_bits(55.0));
    output_vec.append(&mut output_2);
    let mut output_3 = bools_to_floats(&f32_to_bits(100.0));
    output_vec.append(&mut output_3);

    let expected_data: Tensor<f32> = Tensor::from_data(&output_vec, &[3, 32]).unwrap().matrix_transpose().unwrap();

    let output_data = model.full_forward_propagation(&input_data).unwrap();

    println!("expected_data: ");
    for i in 0..input_data.get_sizes()[1]{
        let input_var: f32 = input_data.matrix_col(i).unwrap().sum();

        print!("{}, ", &input_var);
    }
    println!("");

    println!("decoded_data: ");
    for i in 0..output_data[output_data.len()-1].get_sizes()[1]{
        let output_vec: Vec<f32> = output_data[output_data.len()-1].matrix_col(i).unwrap().get_data().to_vec();

        print!("{}, ", bits_to_f32(&floats_to_bools(&output_vec)));
    }
    println!("");
    println!("first Cost: {}", cross_entropy_cost(&output_data[output_data.len()-1], &expected_data).unwrap());

    for _i in 0..100000{
        model.cross_entropy_backprop_loop(&input_data, &expected_data, 0.05);
        let outpud_data = model.full_forward_propagation(&input_data).unwrap();
        println!("Cost: {}", cross_entropy_cost(&outpud_data[outpud_data.len()-1], &expected_data).unwrap());
    }

    //println!("Model at end:\n{}\n__________________________", model.full_to_string());

    let output_data = model.full_forward_propagation(&input_data).unwrap();

    println!("expected_data: ");
    for i in 0..input_data.get_sizes()[1]{
        let input_var: f32 = input_data.matrix_col(i).unwrap().sum();

        print!("{}, ", &input_var);
    }
    println!("");

    println!("decoded_data: ");
    for i in 0..output_data[output_data.len()-1].get_sizes()[1]{
        let output_vec: Vec<f32> = output_data[output_data.len()-1].matrix_col(i).unwrap().get_data().to_vec();

        print!("{}, ", bits_to_f32(&floats_to_bools(&output_vec)));
    }
    println!("");

    println!("_____________________\nNew untested data!!!");

    let input_data_2: Tensor<f32> = Tensor::from_data(&[35.0, 5.0], &[1,2]).unwrap().matrix_transpose().unwrap();
    let _expected_data_2 = Tensor::from_data(&bools_to_floats(&f32_to_bits(40.0)), &[1, 32]).unwrap().matrix_transpose().unwrap();

    let output_data = model.full_forward_propagation(&input_data_2).unwrap();

    println!("expected_data: ");
    for i in 0..input_data_2.get_sizes()[1]{
        let input_var: f32 = input_data_2.matrix_col(i).unwrap().sum();

        print!("{}, ", &input_var);
    }
    println!("");

    println!("decoded_data: ");
    for i in 0..output_data[output_data.len()-1].get_sizes()[1]{
        let output_vec: Vec<f32> = output_data[output_data.len()-1].matrix_col(i).unwrap().get_data().to_vec();

        print!("{}, ", bits_to_f32(&floats_to_bools(&output_vec)));
    }

}

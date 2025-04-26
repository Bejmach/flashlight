use flashlight::prelude::*;
#[allow(unused)]
use flashlight_tensor::prelude::*;

use rand::prelude::*;
use std::time::Instant;

fn main() {

    let mut rng = rand::rng();

    let mut model: FlashlightModel = FlashlightModel::new(&vec!{2, 3, 1});

    let learning_rate: f32 = 0.01;

    let mut input_vec = vec!{67.0, 13.0};

    let output_vec = vec!{1.0};
    let mut input_data: InputPrePrepared = InputPrePrepared::new(&Tensor::from_data(&input_vec, &[1, input_vec.len() as u32]).unwrap(), &Tensor::from_data(&output_vec, &[1, output_vec.len() as u32]).unwrap());
    
    for _i in 0..999{
        let num1 = rng.random_range(-10000.0..10000.0);
        let num2 = rng.random_range(-10000.0..10000.0);
        input_vec = vec!{num1, num2};

        let output_vec;
        if num1>num2 {
            output_vec = vec!{1.0};
        }
        else{
            output_vec = vec!{0.0};
        }

        //println!("Input: {}, {}", input_vec[0], input_vec[1]);

        //print!("Output: ");
        for _i in 0..output_vec.len(){
            //print!("{}", output_vec[i]);
        }
        //println!("");

        input_data.append(&Tensor::from_data(&input_vec, &[1, input_vec.len() as u32]).unwrap(), &Tensor::from_data(&output_vec, &[1, output_vec.len() as u32]).unwrap());
    }

    input_data.set_bach_size(100);

    let input_handler = input_data.to_handler();

    let mut correct_counter: u32 = 0;

    for i in 0..input_data.input_data.len(){
        let outpud_data = model.full_forward_propagation(&input_data.input_data[i].matrix_transpose().unwrap()).unwrap();

        //println!("Sample {}", i);
        //println!("Data: {}", input_data.input_data[i].matrix_to_string().unwrap());
        //println!("Expected: {}", input_data.output_data[i].matrix_to_string().unwrap());
        //println!("Output: {}", outpud_data[outpud_data.len()-1].matrix_to_string().unwrap());
        if input_data.output_data[i].get_data()[0] == 1.0 && outpud_data[outpud_data.len()-1].get_data()[0] > 0.5 {
                //println!("correct");
                correct_counter += 1;
        }
        else if input_data.output_data[i].get_data()[0] == 0.0 && outpud_data[outpud_data.len()-1].get_data()[0] < 0.5 {
                //println!("correct");
                correct_counter += 1;
        }
        else {
            //println!("WRONG");
        }
        //println!("\n");
    }

    println!("Ratio: {}/{}", correct_counter, input_data.input_data.len());

    let start = Instant::now();
    for epoch in 0..=10000{
        if epoch % 100 == 0{
            println!("\nEpoch: {}", epoch);
            let mut cost_sum: f32 = 0.0;
            for i in 0..input_handler.len(){
                model.cross_entropy_backprop_loop(&input_handler.input_bach(i), &input_handler.output_bach(i), learning_rate);
                let outpud_data = model.full_forward_propagation(&input_handler.input_bach(i)).unwrap();
                cost_sum += cross_entropy_cost(&outpud_data[outpud_data.len()-1], &input_handler.output_bach(i)).unwrap();
            }
            println!("avg cost: {}", cost_sum/input_handler.len() as f32);
        }

        else{
            for i in 0..input_handler.len(){
                model.cross_entropy_backprop_loop(&input_handler.input_bach(i), &input_handler.output_bach(i), learning_rate);
            }
        }
    }
    let duration = start.elapsed();
    println!("Learning time: {:?}", duration);

    let mut correct_counter: u32 = 0;

    for i in 0..input_data.input_data.len(){
        let outpud_data = model.full_forward_propagation(&input_data.input_data[i].matrix_transpose().unwrap()).unwrap();

        //println!("Sample {}", i);
        //println!("Data: {}", input_data.input_data[i].matrix_to_string().unwrap());
        //println!("Expected: {}", input_data.output_data[i].matrix_to_string().unwrap());
        //println!("Output: {}", outpud_data[outpud_data.len()-1].matrix_to_string().unwrap());
        if input_data.output_data[i].get_data()[0] == 1.0 && outpud_data[outpud_data.len()-1].get_data()[0] > 0.5 {
                //println!("correct");
                correct_counter += 1;
        }
        else if input_data.output_data[i].get_data()[0] == 0.0 && outpud_data[outpud_data.len()-1].get_data()[0] < 0.5 {
                //println!("correct");
                correct_counter += 1;
        }
        else {
            //println!("WRONG");
        }
        //println!("\n");
    }

    println!("Ratio: {}/{}", correct_counter, input_data.input_data.len());
}

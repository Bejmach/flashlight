use crate::flashlight_model::*;
use flashlight_tensor::prelude::*;

impl FlashlightModel{
    pub fn aesthetic_to_string(&self) -> String{
        
        let mut longest_layer: u32 = 0;

        for var in self.layers.iter(){
            if var>&longest_layer{
                longest_layer = *var;
            }
        }

        let mut return_string: String = String::new();
        
        let layer_difference = longest_layer - self.layers[0];

        for _i in 0..layer_difference{
            return_string.push_str("   ");
        }
        for i in 0..self.layers[0]{
            return_string.push_str(" 0.00 ");
            if i == self.layers[0]-1{
                return_string.push_str("\n\n");
            }
        }

        for i in 0..self.biases.len(){    
            let layer_difference = longest_layer - self.layers[i+1];
            let bias_col: Vec<f32> = self.biases[i].matrix_col(0).unwrap().get_data().to_vec();
            
            for _j in 0..layer_difference{
                return_string.push_str("   ");
            }
            for j in 0..bias_col.len(){
                let bias_value: String = format!("{:>5.2}", bias_col[j]);
                return_string.push_str(&(bias_value + " "));
                if j == bias_col.len()-1 && i != self.biases.len()-1{
                   return_string.push_str("\n\n"); 
                }
            }
        }

        return_string
    }
    
    pub fn full_to_string(&self) -> String{
        let mut return_string: String = String::new();

        return_string.push_str(&Tensor::fill(0.0, &[self.layers[0], 1]).matrix_to_string().unwrap());

        return_string.push_str("\n\n");

        for i in 0..self.layers.len()-1{
            return_string.push_str(&format!("weights {}\n", i));
            return_string.push_str(&self.weights[i].matrix_to_string().unwrap());
            return_string.push_str("\n\n");
            return_string.push_str(&format!("biases {}\n", i));
            return_string.push_str(&self.biases[i].matrix_to_string().unwrap());

            if(i<self.layers.len()-2){
                return_string.push_str("\n\n");
            }
        }

        return_string
    }
}

use std::fmt;

pub struct Matrix{
    width: usize,
    height: usize,
    matrix: Vec<Vec<f32>>,
}

impl Matrix{
    pub fn new(_width: usize, _height: usize) -> Self{
        Self{
            width: _width,
            height: _height,
            matrix: vec![vec![0.0; _width]; _height],
        }
    }
    pub fn from_vec(vec: Vec<Vec<f32>>) -> Self{
        Self{
            width: vec[0].len(),
            height: vec.len(),
            matrix: vec,
        }
    }

    pub fn row(&self, n: usize) -> Option<Vec<f32>>{
        match self.matrix.get(n){
            Some(row) => return Some(row.to_vec()),
            None => return None,
        }
    }
    pub fn col(&self, n: usize) -> Option<Vec<f32>>{
        let mut return_vec: Vec<f32> = Vec::with_capacity(self.matrix.len());

        for arr in self.matrix.iter(){
            match arr.get(n){
                Some(var) => return_vec.push(*var),
                None => return None,
            }
        }

        Some(return_vec)
    }
    pub fn set(&mut self, x: usize, y: usize, value: f32){
        if x>self.width-1 || y>self.height-1{
            return;
        }

        self.matrix[y][x] = value;
    }
    pub fn get(&self, x: usize, y: usize) -> Option<f32>{
        if x>self.width-1 || y>self.height-1{
            return None;
        }

        Some(self.matrix[y][x])
    }

    pub fn transpose(&self) -> Matrix{
        let mut new_matrix = Matrix::new(self.height, self.width);
        
        for i in 0..self.height{
            for j in 0..self.width{
                new_matrix.set(i, j, self.get(j, i).unwrap());
            }
        }

        new_matrix
    }
}

impl fmt::Display for Matrix{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut return_string = String::with_capacity(self.height as usize*5 + self.height as usize * self.width as usize);

        for arr in &self.matrix[0..self.matrix.len()]{
            return_string.push_str("|");
            for (i, element) in arr.iter().enumerate(){
                return_string.push_str(&element.to_string());
                if i<arr.len()-1 {
                    return_string.push_str("; ");
                }
            }
            return_string.push_str("|\n");
        }

        write!(f, "{}", return_string)
    }
}

pub fn dot_product(vec1: Vec<f32>, vec2: Vec<f32>) -> Option<f32>{
    if vec1.len() != vec2.len() {
        return None;
    }
    
    let mut returner: f32 = 0.0;

    for i in 0..vec1.len(){
        returner += vec1[i] * vec2[i];
    }

    Some(returner)
}

pub fn matrix_mult(mat1: Matrix, mat2: Matrix) -> Option<Matrix>{
    if mat1.width != mat2.height{
        return None;
    }

    let mut return_mat = Matrix::new(mat1.height, mat2.width);

    for i in 0..mat1.height{
        for j in 0..mat2.width{
            return_mat.set(j, i, dot_product(mat1.row(i).unwrap(), mat2.col(j).unwrap()).unwrap());
        }
    }

    Some(return_mat)
}

use std::fmt;

use crate::math::sigmoid::*;

#[derive(Clone)]
pub struct Matrix{
    rows: usize,
    collumns: usize,
    matrix: Vec<Vec<f32>>,
}

impl Matrix{
    pub fn new(_rows: usize, _collumns: usize) -> Self{
        Self{
            rows: _rows,
            collumns: _collumns,
            matrix: vec![vec![0.0; _collumns]; _rows],
        }
    }
    pub fn from_vec(vec: Vec<Vec<f32>>) -> Self{
        Self{
            rows: vec.len(),
            collumns: vec[0].len(),
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
    ///set(row: usize, collumn:usize, value: u32)
    pub fn set(&mut self, row: usize, collumn: usize, value: f32){
        if row>self.rows-1 || collumn>self.collumns-1{
            return;
        }

        self.matrix[row][collumn] = value;
    }
    ///get(row: usize, collumn: usize) -> Option<f32>
    pub fn get(&self, row: usize, collumn: usize) -> Option<f32>{
        if row>self.rows-1 || collumn>self.collumns-1{
            return None;
        }

        Some(self.matrix[row][collumn])
    }

    pub fn transpose(&self) -> Matrix{
        let mut new_matrix = Matrix::new(self.collumns, self.rows);

        for row in 0..self.rows{
            for collumn in 0..self.collumns{
                new_matrix.set(collumn, row, self.get(row, collumn).unwrap());
            }
        }

        new_matrix
    }
    pub fn to_sigmoid(&self) -> Matrix{
        let mut new_matrix = Matrix::new(self.rows, self.collumns);

        for row in 0..self.rows{
            for collumn in 0..self.collumns{
                new_matrix.set(row, collumn, sigmoid(self.get(row, collumn).unwrap()));
            }
        }

        new_matrix
    }
}

impl fmt::Display for Matrix{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut return_string = String::with_capacity(self.collumns as usize*5 + self.collumns as usize * self.rows as usize);

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
    if mat1.collumns != mat2.rows{
        return None;
    }

    let mut return_mat = Matrix::new(mat1.rows, mat2.collumns);

    for row in 0..mat1.rows{
        for collumn in 0..mat2.collumns{
            return_mat.set(row, collumn, dot_product(mat1.row(row).unwrap(), mat2.col(collumn).unwrap()).unwrap());
        }
    }

    Some(return_mat)
}
pub fn matrix_add(mat1: Matrix, mat2: Matrix) -> Option<Matrix>{
    if mat1.rows != mat2.rows || mat1.collumns != mat2.collumns{
        return None;
    }

    let mut return_mat = Matrix::new(mat1.rows, mat1.collumns);
    for row in 0..mat1.rows{
        for collumn in 0..mat1.collumns{
            return_mat.set(row, collumn, mat1.get(row, collumn).unwrap() + mat2.get(row, collumn).unwrap());
        }
    }

    Some(return_mat)
}
pub fn matrix_subtract(mat1: Matrix, mat2: Matrix) -> Option<Matrix>{
    if mat1.rows != mat2.rows || mat1.collumns != mat2.collumns{
        return None;
    }

    let mut return_mat = Matrix::new(mat1.collumns, mat1.rows);
    for i in 0..mat1.collumns{
        for j in 0..mat2.rows{
            return_mat.set(j, i, mat1.get(j, i).unwrap() - mat2.get(j, i).unwrap());
        }
    }

    Some(return_mat)
}

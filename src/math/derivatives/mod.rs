use std::fmt;

//y = ax^n
pub struct FunctionPart{
    a: i32,
    n: u32,
}

impl FunctionPart{
    pub const fn new(_a: i32, _n: u32) -> Self{
        Self{
            a: _a,
            n: _n,
        }
    }
    pub fn calculate(&self, x: i32) -> i32{
        self.a*  x.pow(self.n)
    }

    pub fn compose(&self, func_part: FunctionPart) -> FunctionPart{
        FunctionPart{
            a: self.a * func_part.a.pow(self.n),
            n: self.n * func_part.n,
        }
    }

    pub fn derivative(&self) -> FunctionPart{
        FunctionPart{
            a: self.a * self.n as i32,
            n: self.n-1,
        }
    }

    pub fn to_string(&self) -> String{
        return self.a.to_string() + "x^" + &self.n.to_string();
    }
}

impl fmt::Display for FunctionPart{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x^{}", self.a, self.n)
    }
}

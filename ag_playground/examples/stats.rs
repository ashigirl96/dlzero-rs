// use autograd::array_gen::zeros;
use autograd::array_gen::zeros;
use autograd::{Float, NdArray};
use std::cmp;
// use autograd::ndarray::array;
use autograd::ndarray::{
    array as array2, Array2, ArrayBase as ArrayBase2, ArrayD, Data, Dimension, Ix1, IxDyn,
    OwnedRepr,
};
// use autograd::ndarray_ext::zeros;
use autograd::ndarray::IntoDimension;
use ndarray::{array, ArrayBase};
use ndarray_stats::errors::MinMaxError::UndefinedOrder;
use ndarray_stats::errors::{EmptyInput, MinMaxError};
use ndarray_stats::QuantileExt;

pub trait QuantileExt2<A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn argmin(&self) -> Result<D::Pattern, MinMaxError>
    where
        A: PartialOrd;
}

impl<A, S, D> QuantileExt2<A, S, D> for ArrayBase2<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
    A: std::fmt::Debug,
{
    fn argmin(&self) -> Result<D::Pattern, MinMaxError>
    where
        A: PartialOrd,
    {
        let mut current_min = self.first().ok_or(EmptyInput)?;
        let mut current_pattern_min = D::zeros(self.ndim()).into_pattern();

        for (pattern, elem) in self.indexed_iter() {
            eprintln!("pattern = {:#?}", &pattern);
            eprintln!("&elem = {:#?}", &elem);
            if elem.partial_cmp(current_min).ok_or(UndefinedOrder)? == cmp::Ordering::Less {
                current_pattern_min = pattern;
                current_min = elem
            }
        }

        Ok(current_pattern_min)
    }
}

fn main() {
    // let a = array![[1., 3., 7.], [2., 5., 6.]];
    // assert_eq!(a.argmax(), Ok((0, 2)));
    //
    // let a = array![1., 3., 7.];
    // assert_eq!(a.argmax(), Ok(2));
    //
    let a = array2![1., 3., 7.];
    eprintln!("a.argmin() = {:#?}", a.argmin());
    assert_eq!(a.argmin(), Ok(0));

    let mut qs = zeros::<f32>(&[10]);
    qs[4] = -10.0;
    eprintln!("&es = {:#?}", &qs);
    let x = qs.argmin().unwrap();
    eprintln!("x = {:#?}", x[0]);

    // let hgoeaa = autograd::NdArray::<f64>::zeros(autograd::ndarray::Ix2(2, 3));
    // eprintln!("x = {:#?}", hgoeaa);
    // let mut a = ArrayD::<f64>::zeros(IxDyn(&[5]));
    // eprintln!("&a = {:#?}", &a.argmin().unwrap()[0]);

    // NdArray::from_elem(shape, T::zero())
    // eprintln!("&qs.ndim() = {:#?}", &qs.ndim());
    // eprintln!("Dimension::zeros(1); = {:#?}", Dimension::zeros(1));
}

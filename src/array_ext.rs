use autograd::ndarray::{ArrayBase, Data, Dimension};
use ndarray_stats::errors::MinMaxError;
use ndarray_stats::errors::MinMaxError::{EmptyInput, UndefinedOrder};
use std::cmp;
use std::cmp::Ordering;

pub trait QuantileExt<A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    fn argmin(&self) -> Result<D::Pattern, MinMaxError>
    where
        A: PartialOrd;
    fn argmax(&self) -> Result<D::Pattern, MinMaxError>
    where
        A: PartialOrd;

    fn min(&self) -> Result<&A, MinMaxError>
    where
        A: PartialOrd;
    fn max(&self) -> Result<&A, MinMaxError>
    where
        A: PartialOrd;
}

impl<A, S, D> QuantileExt<A, S, D> for ArrayBase<S, D>
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
            if elem.partial_cmp(current_min).ok_or(UndefinedOrder)? == cmp::Ordering::Less {
                current_pattern_min = pattern;
                current_min = elem
            }
        }

        Ok(current_pattern_min)
    }

    fn argmax(&self) -> Result<D::Pattern, MinMaxError>
    where
        A: PartialOrd,
    {
        let mut current_max = self.first().ok_or(EmptyInput)?;
        let mut current_pattern_max = D::zeros(self.ndim()).into_pattern();

        for (pattern, elem) in self.indexed_iter() {
            if elem.partial_cmp(current_max).ok_or(UndefinedOrder)? == Ordering::Greater {
                current_pattern_max = pattern;
                current_max = elem
            }
        }

        Ok(current_pattern_max)
    }

    fn min(&self) -> Result<&A, MinMaxError>
    where
        A: PartialOrd,
    {
        let first = self.first().ok_or(EmptyInput)?;
        self.fold(Ok(first), |acc, elem| {
            let acc = acc?;
            match elem.partial_cmp(acc).ok_or(UndefinedOrder)? {
                Ordering::Less => Ok(elem),
                _ => Ok(acc),
                // Ordering::Equal => {}
                // Ordering::Greater => {}
            }
        })
    }

    fn max(&self) -> Result<&A, MinMaxError>
    where
        A: PartialOrd,
    {
        let first = self.first().ok_or(EmptyInput)?;
        self.fold(Ok(first), |acc, elem| {
            let acc = acc?;
            match elem.partial_cmp(acc).ok_or(UndefinedOrder)? {
                Ordering::Greater => Ok(elem),
                _ => Ok(acc),
            }
        })
    }
}

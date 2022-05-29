use ndarray::{array, s};

fn main() {
    let arr = array![0, 1, 2, 3];
    assert_eq!(arr.slice(s![1..3;-1]), array![2, 1]);
    assert_eq!(arr.slice(s![1..;-2]), array![3, 1]);
    assert_eq!(arr.slice(s![0..4;-2]), array![3, 1]);
    assert_eq!(arr.slice(s![0..;-2]), array![3, 1]);
    assert_eq!(arr.slice(s![..;-2]), array![3, 1]);
    // fn laplacian(v: &ArrayView2<f32>) -> Array2<f32> {
    //     -4. * &v.slice(s![1..-1, 1..-1])
    //         + v.slice(s![ ..-2, 1..-1])
    //         + v.slice(s![1..-1,  ..-2])
    //         + v.slice(s![1..-1, 2..  ])
    //         + v.slice(s![2..  , 1..-1])
    // }
}

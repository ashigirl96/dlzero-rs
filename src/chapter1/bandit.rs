use crate::array_ext::QuantileExt;
use autograd::array_gen::{zeros, ArrayRng};
use autograd::ndarray::{array, Array};
use autograd::rand::prelude::{StdRng, ThreadRng};
use autograd::rand::{Rng, SeedableRng};
use autograd::{rand, NdArray};

pub struct Bandit {
    pub rates: NdArray<f32>,
    unit_rng: StdRng,
}

impl Bandit {
    pub fn new(arms: usize) -> Self {
        let rng = ArrayRng::<f32>::default();
        let unit_rng = StdRng::seed_from_u64(42);
        let rates = rng.random_uniform(&[arms], 0.0, 1.0);
        Self { rates, unit_rng }
    }

    pub fn step(&mut self, arm: usize) -> f32 {
        let rate = self.rates[arm];
        if rate > self.unit_rng.gen::<f32>() {
            1.0
        } else {
            0.0
        }
    }
}

pub struct Agent {
    epsilon: f32,
    qs: NdArray<f32>,
    ns: NdArray<f32>,
    rng: ThreadRng,
    action_size: usize,
}

impl Agent {
    pub fn new(epsilon: f32, action_size: usize) -> Self {
        let qs = zeros(&[action_size]);
        let ns = zeros(&[action_size]);
        let mut rng = rand::thread_rng();
        Self {
            epsilon,
            qs,
            ns,
            rng,
            action_size,
        }
    }

    pub fn learn(&mut self, action: usize, reward: f32) {
        self.ns[action] += 1.0;
        self.qs[action] += (reward - self.qs[action]) / self.ns[action];
    }

    pub fn step(&mut self) -> usize {
        if rand::random::<f32>() < self.epsilon {
            self.rng.gen_range(0..self.action_size)
        } else {
            self.qs.argmax().unwrap()[0]
        }
        .try_into()
        .unwrap()
    }
}

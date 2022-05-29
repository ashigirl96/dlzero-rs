use crate::array_ext::QuantileExt;
use autograd::array_gen::{zeros, ArrayRng};
use autograd::ndarray::{array, Array};
use autograd::rand::prelude::{StdRng, ThreadRng};
use autograd::rand::{Rng, SeedableRng};
use autograd::{rand, NdArray};
use rand_distr::Distribution;

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
        let mut normal = rand_distr::Normal::new(0., 1.).unwrap();
        let noise = normal.sample(&mut self.unit_rng);
        let rate = self.rates[arm];
        for rate in &mut self.rates {
            *rate += 0.1 * noise;
        }
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
    rng: ThreadRng,
    action_size: usize,
    alpha: f32,
}

impl Agent {
    pub fn new(epsilon: f32, action_size: usize, alpha: f32) -> Self {
        let qs = zeros(&[action_size]);
        let mut rng = rand::thread_rng();
        Self {
            epsilon,
            qs,
            rng,
            action_size,
            alpha,
        }
    }

    pub fn learn(&mut self, action: usize, reward: f32) {
        self.qs[action] += (reward - self.qs[action]) * self.alpha;
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

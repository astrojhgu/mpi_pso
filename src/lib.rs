use std::ops::{Add, IndexMut, Mul, Sub};


use mpi::collective::CommunicatorCollectives;
use mpi::collective::Root;
use mpi::datatype::BufferMut;
use mpi::datatype::Equivalence;
use mpi::topology::Rank;
use mpi_sys::MPI_Comm;


use scorus::linear_space::LinearSpace;
use num_traits::cast::NumCast;
use num_traits::float::Float;
use num_traits::identities::{one, zero};
use rand::distributions::uniform::SampleUniform;
use rand::Rng;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct Particle<V, T>
    where
        T: Float
        + NumCast
        + std::cmp::PartialOrd
        + Copy
        + Default
        + SampleUniform
        + Debug
        + Equivalence
        + Send
        + Sync,
        V: Clone + IndexMut<usize, Output = T> + LinearSpace<T> + Debug + Send + Sync,
        for<'b> &'b V: Add<Output = V>,
        for<'b> &'b V: Sub<Output = V>,
        for<'b> &'b V: Mul<T, Output = V>,
{
    pub position: V,
    pub velocity: V,
    pub fitness: T,
    pub pbest: Option<Box<Particle<V, T>>>,
}

fn calc_task_per_node(ntasks:usize, nnodes:usize)->usize{
    let x = ntasks / nnodes;
    if x * nnodes >= ntasks {
        x
    } else {
        x + 1
    }
}



pub struct ParticleSwarmMaximizer<'a, V, T>
    where
        T: Float
        + NumCast
        + std::cmp::PartialOrd
        + Copy
        + Default
        + SampleUniform
        + Debug
        + Equivalence
        + Send
        + Sync,
        V: Clone + IndexMut<usize, Output = T> + LinearSpace<T> + Debug + Send + Sync + AsMut<[T]>,
        for<'b> &'b V: Add<Output = V>,
        for<'b> &'b V: Sub<Output = V>,
        for<'b> &'b V: Mul<T, Output = V>,
{
    pub particle_count: usize,
    pub ndim: usize,
    pub swarm: Vec<Particle<V, T>>,
    pub gbest: Option<Particle<V, T>>,
    pub func: &'a (Fn(&V) -> T + Send + Sync),
}

impl<'a, V, T> ParticleSwarmMaximizer<'a, V, T>
    where
        T: Float
        + NumCast
        + std::cmp::PartialOrd
        + Copy
        + Default
        + SampleUniform
        + Equivalence
        + Debug
        + Send
        + Sync,
        V: Clone + IndexMut<usize, Output = T> + LinearSpace<T> + Debug + Send + Sync + AsMut<[T]>,
        for<'b> &'b V: Add<Output = V>,
        for<'b> &'b V: Sub<Output = V>,
        for<'b> &'b V: Mul<T, Output = V>,
        [T]: BufferMut,
{
    pub fn new<R, C>(
        func: &'a (Fn(&V) -> T + Sync + Send),
        lower: &V,
        upper: &V,
        guess: Option<V>,
        particle_count: usize,
        rng: &mut R,
        comm: &C,
    ) -> ParticleSwarmMaximizer<'a, V, T>
        where
            R: Rng,
            C: CommunicatorCollectives<Raw = MPI_Comm>,
    {
        let swarm = Self::init_swarm(&func, lower, upper, particle_count, rng, comm);
        let ndim = lower.dimension();
        let gbest = guess.map(|p| {
            let f = func(&p);
            Particle {
                position: p,
                velocity: lower * T::zero(),
                fitness: f,
                pbest: None,
            }
        });
        ParticleSwarmMaximizer {
            particle_count,
            ndim,
            swarm,
            gbest,
            func,
        }
    }

    pub fn restart<R, C>(&mut self, lower: &V, upper: &V, particle_count: usize, rng: &mut R, comm: &C)
        where
            R: Rng,
            C: CommunicatorCollectives<Raw = MPI_Comm>,
    {
        self.swarm = Self::init_swarm(&self.func, lower, upper, particle_count, rng, comm);
    }

    pub fn init_swarm<R, C>(
        func: &Fn(&V) -> T,
        lower: &V,
        upper: &V,
        pc: usize,
        rng: &mut R,
        comm: &C
    ) -> Vec<Particle<V, T>>
        where
            R: Rng,
            C: CommunicatorCollectives<Raw = MPI_Comm>,
    {
        let rank = comm.rank();
        let ntasks_per_node=calc_task_per_node(pc, comm.size() as usize);
        let mut result = Vec::<Particle<V, T>>::new();
        let ndim = lower.dimension();

        let mut ps=Vec::new();
        
        for _i in 0..pc{
            let mut p = lower * T::zero();
            for j in 0..ndim {
                p[j] = rng.gen_range(lower[j], upper[j]);
            }
            ps.push(p);
        }
        let root=comm.process_at_rank(0 as Rank);
        for i in &mut ps{
            root.broadcast_into(AsMut::<[T]>::as_mut(i));
        }

        let mut fs=vec![zero(); pc];

        for k in (rank as usize * ntasks_per_node)..((rank + 1) as usize * ntasks_per_node) {
            if k>pc{
                break;
            }
            fs[k]=func(&ps[k]);
        }
        
        for k in 0..pc{
            let temp_root_id = (k / ntasks_per_node) as Rank;
            let temp_root = comm.process_at_rank(temp_root_id);
            temp_root.broadcast_into(&mut fs[k]);
        }

        for (i, p) in ps.into_iter().enumerate() {
            //let mut p = lower * T::zero();
            let mut v = lower * T::zero();
            for j in 0..ndim {
                v[j] = zero();
            }
            let f = fs[i];
            result.push(Particle {
                position: p,
                velocity: v,
                fitness: f,
                pbest: None,
            });
        }
        result
    }

    pub fn update_fitness<C>(&mut self, comm:&C)
    where C: CommunicatorCollectives<Raw = MPI_Comm>,
     {
        /*let f: Vec<T> = self
            .swarm
            .iter()
            .map(|p| (self.func)(&p.position))
            .collect();*/
        let rank = comm.rank();
        let pc=self.swarm.len();
        let ntasks_per_node=calc_task_per_node(pc, comm.size() as usize);
        let mut fs=vec![zero::<T>(); pc];

        for k in (rank as usize * ntasks_per_node)..((rank + 1) as usize * ntasks_per_node) {
            if k>pc{
                break;
            }
            fs[k]=(self.func)(&self.swarm[k].position);
        }
        
        for k in 0..pc{
            let temp_root_id = (k / ntasks_per_node) as Rank;
            let temp_root = comm.process_at_rank(temp_root_id);
            temp_root.broadcast_into(&mut fs[k]);
        }

        fs.iter().zip(self.swarm.iter_mut()).for_each(|(&f, p)| {
            p.fitness = f;
        });
    }

    pub fn sample<R, C>(&mut self, rng: &mut R, c1: T, c2: T, comm:&C)
        where
            R: Rng,
            C: CommunicatorCollectives<Raw = MPI_Comm>,
    {
        for p in &mut self.swarm {
            match self.gbest {
                None => {
                    self.gbest = Some(Particle {
                        position: p.position.clone(),
                        velocity: p.velocity.clone(),
                        fitness: p.fitness,
                        pbest: None,
                    })
                }
                Some(ref mut gb) => {
                    if gb.fitness < p.fitness {
                        gb.position = p.position.clone();
                        gb.velocity = p.velocity.clone();
                        gb.fitness = p.fitness;
                    }
                }
            }

            match p.pbest {
                None => {
                    p.pbest = Some(Box::new(Particle {
                        position: p.position.clone(),
                        velocity: p.velocity.clone(),
                        fitness: p.fitness,
                        pbest: None,
                    }))
                }

                Some(ref mut pb) => {
                    if p.fitness > pb.fitness {
                        pb.position = p.position.clone();
                        pb.velocity = p.velocity.clone();
                        pb.fitness = p.fitness;
                    }
                }
            }
        }

        let mut rn=Vec::new();
        for _ in 0..3*self.ndim*self.swarm.len(){
            rn.push(rng.gen_range(zero::<T>(), one::<T>()));
        }

        let root=comm.process_at_rank(0 as Rank);
        root.broadcast_into(&mut rn[..]);

        for p in &mut self.swarm {
            if let Some(ref pbest) = p.pbest {
                if let Some(ref gbest) = self.gbest {
                    for j in 0..self.ndim {
                        let w = (one::<T>() + rn.pop().unwrap())
                            / (one::<T>() + one::<T>());
                        let part_vel = w * p.velocity[j];
                        let cog_vel = c1
                            * rn.pop().unwrap()
                            * (pbest.position[j] - p.position[j]);
                        let soc_vel = c2
                            * rn.pop().unwrap()
                            * (gbest.position[j] - p.position[j]);
                        p.velocity[j] = part_vel + cog_vel + soc_vel;
                        p.position[j] = p.position[j] + p.velocity[j]
                    }
                }
            }
        }

        self.update_fitness(comm);
    }

    pub fn converged(&self, p: T, m1: T, m2: T) -> bool {
        self.converge_dfit(p, m1) && self.converge_dspace(p, m2)
    }

    pub fn converge_dfit(&self, p: T, m: T) -> bool {
        let mut best_sort: Vec<T> = self.swarm.iter().map(|x| x.fitness).collect();
        best_sort.sort_unstable_by(|&a, &b| {
            if a > b {
                std::cmp::Ordering::Less
            } else if a < b {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        let i1: usize = (T::from(self.particle_count).unwrap() * p)
            .floor()
            .to_usize()
            .unwrap();
        let best_mean = best_sort[1..i1].iter().fold(zero::<T>(), |a, &b| a + b)
            / NumCast::from(i1 - 1).unwrap();
        if let Some(ref x) = self.gbest {
            (x.fitness - best_mean).abs() < m
        } else {
            false
        }
    }

    pub fn converge_dspace(&self, p: T, m: T) -> bool {
        let mut sorted_swarm: Vec<_> = self.swarm.to_vec();
        sorted_swarm.sort_unstable_by(|a, b| {
            if a.fitness > b.fitness {
                std::cmp::Ordering::Less
            } else if a.fitness < b.fitness {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        let i1: usize = (T::from(self.particle_count).unwrap() * p)
            .floor()
            .to_usize()
            .unwrap();
        if let Some(ref gbest) = self.gbest {
            let max_norm: T = sorted_swarm[0..i1]
                .iter()
                .map(|x| {
                    let mut diff_norm: T = zero();

                    for i in 0..self.ndim {
                        diff_norm = diff_norm + (gbest.position[i] - x.position[i]).powi(2);
                    }
                    diff_norm
                }).fold(zero::<T>(), |a: T, b: T| b.max(a));
            max_norm < m
        } else {
            false
        }
    }
}

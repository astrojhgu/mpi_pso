use mpi::topology::Communicator;
use scorus::linear_space::type_wrapper::LsVec;

use std::fs::File;
use std::io::Write;
use std::fs::OpenOptions;

use rand::distributions::uniform::SampleUniform;
use rand::Rng;

fn foo(x: &LsVec<f64, Vec<f64>>) -> f64 {
    x.0.iter().map(|x| -x * x).sum()
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let mut rng = rand::thread_rng();
    let rank = world.rank();

    let ensemble:Vec<_>=(0..168).map(|_|{
        let mut result=Vec::new();
        for i in 0..20{
            result.push(rng.gen_range(0.0, 100.0));
        }
        LsVec(result)
    }).collect();


    let mut pso = mpi_pso::ParticleSwarmMaximizer::from_ensemble(&foo, ensemble, None, &world);


    for _i in 0..10 {
        while !pso.converged(0.7, 1e-11, 1e-11) {
            pso.sample(&mut rng, 1.193, 1.193, &world);
            if rank == 0 {
                if let Some(ref gbest) = pso.gbest {
                    eprintln!("\n{:?} {}", gbest.position, gbest.fitness);
                } else {
                    eprint!(".")
                }

                let mut f_last = File::create("last_state.txt").unwrap();

                for (j, p) in pso.swarm.iter().enumerate() {
                    {
                        let fname = format!("particle_{}.dat", j + 1);

                        let mut f = match OpenOptions::new().append(true).open(&fname) {
                            Ok(f) => f,
                            _ => File::create(&fname).unwrap(),
                        };
                        for x in &p.position.0 {
                            let _ = write!(&mut f_last, "{}", x);
                            let _ = write!(&mut f, "{} ", x);
                        }

                        let _ = writeln!(&mut f, "{}", p.fitness);
                        let _ = writeln!(&mut f_last);
                    }
                }
            }
        }
    }
}

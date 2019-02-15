use mpi_sys::MPI_Comm;
use scorus::linear_space::type_wrapper::LsVec;
use mpi::topology::Communicator;

fn foo(x:&LsVec<f64, Vec<f64>>)->f64{
    x.0.iter().map(|x|{-x*x}).sum()
}

fn main(){
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let mut rng = rand::thread_rng();
    let rank=world.rank();

    let mut pso=mpi_pso::ParticleSwarmMaximizer::new(&foo, &LsVec(vec![-100.0, -100.0]), &LsVec(vec![100.0, 100.0]), None, 20, &mut rng, &world);

    while !pso.converged(0.7, 1e-9, 1e-9) {
        if rank==0 {
            if let Some(ref gbest) = pso.gbest {
                eprintln!("\n{:?} {}", gbest.position, gbest.fitness);
            } else {
                eprint!(".")
            }
        }
        pso.sample(&mut rng, 1.193, 1.193, &world);
    }
}
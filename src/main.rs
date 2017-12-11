extern crate rayon;
extern crate rand;
extern crate statrs;
extern crate petgraph;
extern crate vec_graph;
extern crate capngraph;
extern crate ris;
extern crate bit_set;
extern crate docopt;
#[macro_use]
extern crate slog;
extern crate slog_stream;
extern crate slog_term;
extern crate slog_json;
extern crate serde;
extern crate serde_json;
extern crate bincode;
#[macro_use]
extern crate serde_derive;
extern crate rand_mersenne_twister;
extern crate avarice;
extern crate setlike;
#[macro_use]
extern crate avarice_derive;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use petgraph::visit::NodeCount;
use vec_graph::{Graph, NodeIndex, EdgeIndex};
use rayon::prelude::*;
use slog::{Logger, DrainExt};
use serde_json::to_string as json_string;
use statrs::distribution::Categorical;
use rand::Rng;
use rand::distributions::IndependentSample;
use rand_mersenne_twister::{MTRng64, mersenne};
use bincode::{deserialize_from as bin_read_from, Infinite};
use std::cell::RefCell;
use avarice::objective::{Objective, Set, ElementIterator, ConstrainedObjective, LazyObjective};
use avarice::errors::Result as AvaResult;
use avarice::greedy::lazier_greedy_constrained;
use setlike::Setlike;
use bit_set::BitSet;

use ris::*;

#[cfg_attr(rustfmt, rustfmt_skip)]
const USAGE: &'static str = "
Run the BCT algorithm.

If <delta> is not given, 1/n is used as a default.

If --costs are not given, then they are treated as uniformly 1.
If --benefits are not given, then they are treated as uniformly 1.
Thus, ommitting both is equivalent to the normal unweighted IM problem.

Usage:
    bct <graph> <model> <k> <epsilon> [<delta>] [options]
    bct (-h | --help)

Options:
    -h --help               Show this screen.
    --log <logfile>         Log to given file.
    --threads <threads>     Number of threads to use.
    --costs <cost-file>     Node costs. See the `tiptop` repository for generation binary.
    --benefits <ben-file>   Node benefits. See the `tiptop` repository for generation binary.
";

#[derive(Debug, Serialize, Deserialize)]
struct Args {
    arg_graph: String,
    arg_model: Model,
    arg_k: f64,
    arg_epsilon: f64,
    arg_delta: Option<f64>,
    flag_log: Option<String>,
    flag_threads: Option<usize>,
    flag_costs: Option<String>,
    flag_benefits: Option<String>,
}

type CostVec = Vec<f64>;
type BenVec = Vec<f64>;
type BenDist = Categorical;

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
enum Model {
    IC,
    LT,
}

#[derive(Submodular)]
struct WeightedMaxCoverage<'a> {
    sets: &'a [BTreeSet<NodeIndex>],
    g: &'a Graph<(), f32>,
    costs: Option<&'a CostVec>,
    budget: f64,
    covers: Vec<BitSet>,
}

impl<'a> WeightedMaxCoverage<'a> {
    fn new(sets: &'a [BTreeSet<NodeIndex>], g: &'a Graph<(), f32>, budget: f64, costs: Option<&'a CostVec>) -> Self {
        let mut covers = vec![BitSet::default(); g.node_count()];
        for (i, set) in sets.iter().enumerate() {
            for el in set {
                covers[el.index()].insert(i);
            }
        }
        Self {
            sets, g, budget, costs, covers
        }
    }
}

#[derive(Clone)]
struct WMCState {
    covered: BitSet,
    cost: f64,
}

impl WMCState {
    fn init(obj: &WeightedMaxCoverage) -> Self {
        WMCState {
            covered: BitSet::default(),
            cost: 0.0,
        }
    }
}

impl<'a> Objective for WeightedMaxCoverage<'a> {
    type Element = NodeIndex;
    type State = Option<WMCState>;

    fn elements(&self) -> ElementIterator<Self> {
        Box::new(self.g.node_indices())
    }

    fn benefit<S: Setlike<Self::Element>>(&self, set: &S, state: &Self::State) -> AvaResult<f64> {
        if let &Some(ref s) = state {
            Ok(s.covered.len() as f64)
        } else {
            Ok(0.0)
        }
    }

    fn delta<S: Setlike<Self::Element>>(&self,
                                        u: Self::Element,
                                        set: &S,
                                        state: &Self::State)
                                        -> AvaResult<f64> {
        let cost = self.costs.map(|c| c[u.index()]).unwrap_or(1.0);
        if let &Some(ref s) = state {
            let gain = self.covers[u.index()].difference(&s.covered).count() as f64;
            Ok(gain / cost)
        } else {
            let gain = self.covers[u.index()].len() as f64;
            Ok(gain / cost)
        }
    }

    fn depends(&self, u: Self::Element, state: &Self::State) -> AvaResult<ElementIterator<Self>> {
        Ok(self.elements())
    }

    fn insert_mut(&self, u: Self::Element, state: &mut Self::State) -> AvaResult<()> {
        let mut s = state.get_or_insert_with(|| WMCState::init(self));
        s.covered.union_with(&self.covers[u.index()]);
        s.cost += self.costs.map(|c| c[u.index()]).unwrap_or(1.0);
        Ok(())
    }
}

impl<'a> ConstrainedObjective for WeightedMaxCoverage<'a> {
    fn valid_addition(&self, el: Self::Element, set: &Vec<Self::Element>, state: &Option<WMCState>) -> bool {
        let cost = self.costs.map(|c| c[el.index()]).unwrap_or(1.0);
        if let &Some(ref s) = state {
            s.cost + cost <= self.budget
        } else {
            cost <= self.budget
        }
    }
}

impl<'a> LazyObjective for WeightedMaxCoverage<'a> {
    fn update_lazy_mut<S: Setlike<Self::Element>>(&self, u: Self::Element, previous: &S, state: &mut Self::State) -> AvaResult<Option<f64>> {
        let cost = self.costs.map(|c| c[u.index()]).unwrap_or(1.0);
        let gain = if let &mut Some(ref s) = state {
            let gain = self.covers[u.index()].difference(&s.covered).count();
            if gain == self.covers[u.index()].len() {
                return Ok(None);
            }
            gain
        } else {
            self.covers[u.index()].len()
        };
        Ok(Some(gain as f64 / cost))
    }

    fn insert_lazy_mut(&self, element: Self::Element, state: &mut Self::State) -> AvaResult<()> {
        let s = state.get_or_insert_with(|| WMCState::init(self));
        s.covered.union_with(&self.covers[element.index()]);
        s.cost += self.costs.map(|c| c[element.index()]).unwrap_or(1.0);
        Ok(())
    }
}

thread_local!(static RNG: RefCell<MTRng64> = RefCell::new(mersenne()));

/// log(n choose k) computed using the sum form to avoid overflow.
fn logbinom(n: usize, k: usize) -> f64 {
    (1..(k + 1)).map(|i| ((n + 1 - i) as f64).ln() - (i as f64).ln()).sum()
}

/// Construct a reverse-reachable sample according to the BSA algorithm under
/// either the IC or LT model.
///
/// If no benefits are given, this does uniform sampling.
fn rr_sample<R: Rng>(rng: &mut R,
                     g: &Graph<(), f32>,
                     model: Model,
                     weights: &Option<BenDist>)
                     -> BTreeSet<NodeIndex> {
    if let &Some(ref dist) = weights {
        let v = dist.ind_sample(rng);
        assert_eq!(v, v.trunc());
        let v = NodeIndex::new(v as usize);
        match model {
            Model::IC => IC::new(rng, g, v),
            Model::LT => LT::new(rng, g, v),
        }
    } else {
        match model {
            Model::IC => IC::new_uniform_with(rng, g),
            Model::LT => LT::new_uniform_with(rng, g),
        }
    }
}

/// Compute the value `k_max` that is the size of the largest set with cost at most `b`.
fn determine_kmax(costs: &Option<CostVec>, b: f64) -> usize {
    if costs.is_none() {
        return b as usize;
    }
    let mut costs = costs.as_ref().unwrap().clone();
    costs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut sum = 0.0;
    for (i, &c) in costs.iter().enumerate() {
        if sum + c <= b {
            sum += c;
        } else {
            return i;
        }
    }
    // default case: Σ costs ≤ b ⇒ we can just pick n elements
    costs.len()
}

fn bct(g: Graph<(), f32>,
       costs: Option<CostVec>,
       benefits: Option<BenVec>,
       model: Model,
       k: f64,
       eps: f64,
       delta: f64,
       log: Logger)
       -> Vec<NodeIndex> {
    let n = g.node_count() as f64;
    use std::f64::consts::E;

    let k_max = determine_kmax(&costs, k) as f64;

    let c = 2.0 * (E - 2.0);
    let upsilon_base = 8.0 * c * (1.0 - 1.0 / (2.0 * E)).powi(2) * eps.powi(-2);
    let upsilon = upsilon_base *
                  if costs.is_none() {
        (1.0 / delta).ln() + logbinom(g.node_count(), k as usize) + 2.0 / n
    } else {
        (1.0 / delta).ln() + k_max * n.ln() * 2.0 / n
    };

    let lambda = (1.0 + (eps * E) / (2.0 * E - 1.0)) * upsilon;
    #[allow(non_snake_case)]
    let mut N_t = lambda.ceil() as usize;

    info!(log, "beginning loop"; "Υ" => upsilon, "N_t" => N_t, "Λ" => lambda);

    let dist = benefits.as_ref().map(|w| Categorical::new(w).unwrap());
    let mut samples = vec![];
    loop {
        info!(log, "sampling"; "additional" => N_t - samples.len(), "total" => N_t);
        let mut next_sets = Vec::with_capacity(N_t - samples.len());
        (0..N_t - samples.len())
            .into_par_iter()
            .map(|_| RNG.with(|rng| rr_sample(&mut *rng.borrow_mut(), &g, model, &dist)))
            .collect_into(&mut next_sets);
        samples.append(&mut next_sets);
        N_t *= 2;
        info!(log, "done sampling");

        let obj = WeightedMaxCoverage::new(&samples, &g, k, costs.as_ref());
        let (_, sol, state) = lazier_greedy_constrained(&obj, k_max as usize, |_sol, el, state| {
            let cost = costs.as_ref().map(|c| c[el.index()]).unwrap_or(1.0);
            if let &Some(ref s) = state {
                s.cost + cost <= k
            } else {
                cost <= k
            }
        }, None).unwrap();
        info!(log, "found solution");
        let s = state.unwrap();
        let degree_sol = s.covered.len();
        let u = g.node_indices().max_by_key(|u| obj.covers[u.index()].len()).unwrap();
        let degree_u = obj.covers[u.index()].len();

        let (degree, sol) = if degree_u > degree_sol {
            (degree_u as f64, vec![u])
        } else {
            (degree_sol as f64, sol)
        };

        if degree >= lambda {
            let cost = sol.iter().map(|u| costs.as_ref().map(|c| c[u.index()]).unwrap_or(1.0)).sum::<f64>();
            assert!(cost <= k, "cost exceeds budget!");
            return sol;
        } else {
            info!(log, "solution did not pass, iterating"; "degree" => degree, "Λ" => lambda);
        }
    }
}

fn main() {
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());

    if let Some(threads) = args.flag_threads {
        rayon::initialize(rayon::Configuration::new().num_threads(threads)).unwrap();
    }

    let log =
        match args.flag_log {
            Some(ref filename) => slog::Logger::root(slog::Duplicate::new(slog_term::streamer().color().compact().build(),
                                                                  slog_stream::stream(File::create(filename).unwrap(), slog_json::default())).fuse(), o!("version" => env!("CARGO_PKG_VERSION"))),
            None => {
                slog::Logger::root(slog_term::streamer().color().compact().build().fuse(),
                                   o!("version" => env!("CARGO_PKG_VERSION")))
            }
        };

    info!(log, "parameters"; "args" => json_string(&args).unwrap());
    info!(log, "loading graph"; "path" => args.arg_graph);
    let g = Graph::oriented_from_edges(capngraph::load_edges(args.arg_graph.as_str()).unwrap(),
                                       petgraph::Incoming);
    let delta = args.arg_delta.unwrap_or(1.0 / g.node_count() as f64);
    let costs: Option<CostVec> = args.flag_costs
        .as_ref()
        .map(|path| bin_read_from(&mut File::open(path).unwrap(), Infinite).unwrap());
    let bens: Option<BenVec> = args.flag_benefits
        .as_ref()
        .map(|path| bin_read_from(&mut File::open(path).unwrap(), Infinite).unwrap());

    if let Some(ref c) = costs {
        assert_eq!(c.len(), g.node_count());
    }

    if let Some(ref b) = bens {
        assert_eq!(b.len(), g.node_count());
    }

    let seeds = bct(g,
                    costs,
                    bens,
                    args.arg_model,
                    args.arg_k,
                    args.arg_epsilon,
                    delta,
                    log.new(o!("section" => "bct")));
    info!(log, "solution"; "seeds" => json_string(&seeds.into_iter().map(|node| node.index()).collect::<Vec<_>>()).unwrap());

}

#[cfg(test)]
mod test {
    use super::*;

    /// This test checks the consistency of the solution across changes to the code. The set
    /// `BASIS` is 15 nodes that show up in effectively all solutions with these settings. The set
    /// was found by taking a solution and iteratively running the test and removing seed nodes
    /// until it passed 20 times in a row with no modifications.
    #[test]
    fn grqc_unweighted_consistency() {
        const BASIS: [usize; 15] = [296, 1038, 109, 578, 54, 187, 1734, 21, 12, 280, 366, 1089, 316, 1244, 1033];
        // removed: 102, 347, 451, 1285, 364
        let g = Graph::oriented_from_edges(capngraph::load_edges("ca-GrQc.bin").unwrap(), petgraph::Incoming);
        let n = g.node_count() as f64;
        let seeds = bct(g, None, None, Model::IC, 20.0, 0.1, 1.0 / n, slog::Logger::root(slog::Discard, o!()));
        let seeds = seeds.into_iter().map(|u| u.index()).collect::<Vec<_>>();
        println!("{:?}", seeds);
        for el in &BASIS {
            assert!(seeds.contains(el), "does not contain: {}", el);
        }
    }

    #[test]
    fn grqc_weighted_consistency() {
        const BASIS: [usize; 56] = [1131, 2814, 1445, 5063, 1001, 4572, 1126, 4649, 1004, 2636, 3311, 2223, 3513, 5001, 4294, 4699, 1079, 2477, 3262, 4602, 1567, 5239, 948, 4927, 2291, 3097, 2704, 2414, 2963, 722, 4222, 844, 3923, 4109, 4274, 1728, 5169, 1128, 4175, 1834, 4718, 3613, 1551, 3626, 3328, 2547, 3186, 830, 3731, 2007, 2785, 3532, 3486, 3022, 2830, 0];
        // removed: 1399,3019, 854, 4104, 3466, 4158, 2928, 1997, 3684, 4216, 3124, 1580, 689, 2864, 2637, 1345, 2555, 1464, 2188, 4716, 
        let g = Graph::oriented_from_edges(capngraph::load_edges("ca-GrQc.bin").unwrap(), petgraph::Incoming);
        let costs: Option<CostVec> = Some("grqc_costs.bin")
            .as_ref()
            .map(|path| bin_read_from(&mut File::open(path).unwrap(), Infinite).unwrap());
        let bens: Option<BenVec> = Some("grqc_bens.bin")
            .as_ref()
            .map(|path| bin_read_from(&mut File::open(path).unwrap(), Infinite).unwrap());
        let n = g.node_count() as f64;
        let seeds = bct(g, costs, bens, Model::IC, 20.0, 0.1, 1.0 / n, slog::Logger::root(slog::Discard, o!()));
        let seeds = seeds.into_iter().map(|u| u.index()).collect::<Vec<_>>();
        println!("{:?}", seeds);
        for el in BASIS.iter() {
            assert!(seeds.contains(el), "does not contain: {}", el);
        }
    }

    #[test]
    fn grqc_weighted_budget_constraint() {
        const BUDGET: f64 = 20.0;
        let g = Graph::oriented_from_edges(capngraph::load_edges("ca-GrQc.bin").unwrap(), petgraph::Incoming);
        let costs: Option<CostVec> = Some("grqc_costs.bin")
            .as_ref()
            .map(|path| bin_read_from(&mut File::open(path).unwrap(), Infinite).unwrap());
        let bens: Option<BenVec> = Some("grqc_bens.bin")
            .as_ref()
            .map(|path| bin_read_from(&mut File::open(path).unwrap(), Infinite).unwrap());
        let n = g.node_count() as f64;
        let seeds = bct(g, costs.clone(), bens, Model::IC, BUDGET, 0.1, 1.0 / n, slog::Logger::root(slog::Discard, o!()));
        let costs = costs.unwrap();
        let total_cost = seeds.iter().map(|u| costs[u.index()]).sum::<f64>();
        assert!(total_cost <= BUDGET, "cost exceeded budget: {} > {}", total_cost, BUDGET);
    }
}

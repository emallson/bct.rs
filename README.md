# BCT

A re-implementation of the BCT algorithm from [this
paper][paper] in Rust.

Sampling is done using [this library](https://github.com/emallson/ris.rs)
and solving is done with
[avarice](https://github.com/emallson/avarice.rs/tree/setlike).

*This is not the canonical/reference implementation!* I re-implemented this as
a learning exercise.

## Input Format

The graph input is expected to be in
[Capngraph](https://github.com/emallson/capngraph) format. See the linked
repository for conversion tools.

The cost/benefit inputs are constructed using the [included
binary.](./src/bin/build-data.rs)

## License

Obviously, the method is taken from [here][paper].
The code itself is wholly mine at this point, and is made available under the
[BSD 3-Clause License](./LICENSE).

[paper]: https://doi.org/10.1109/INFOCOM.2016.7524377

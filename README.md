# linefitter
A library written in Rust that solves one problem: Finding a function that approximates a given line.

Currently works with the following types of functions:

- Constant
- Linear
- Affine
- Power
- Polynomial
    - Only with coefficients in 0, 1

Example
----

    fn example() {
        let input_vector: Vec<i32> = vec![-7, -6, -5, -3, -1, 0, 2, 4, 6, 8];
        let true_function: Function = Function { constant: 893,
                                                 linear_coefficient: 23490,
                                                 polynomial: Vec::new(),
                                                 error: 0};
        // create samples from the line f(x) -> 23490x + 893
        let mut samples = true_function.evaluate_at_points(&input_vector);

        // simulate noisy samples
        for index in 0..samples.len() {
            samples[index] += 2 * index * (-1.pow(index % 2));
        }

        let best_fit = fit_line(&input_vector, &samples);
        println!("Best fit for {:?} -> {:?}: {:?}", best_fit);

        let other_inputs: Vec<i32> = vec![-20, -19, -18, -17, 10, 13, 15, 17];
        let extrapolation = best_fit.evaluate_at_points(&other_inputs);
    }



Why?
----
It was/is a good first project to learn Rust. Numerical processing does not
require very deep language features.

There are probably more advanced techniques for solving these problems than
those used in this library.

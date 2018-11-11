// to do:
//  different (simpler) techniques for exponents when not using bignums
//  test_polynomial
//      change to use bignums
//      finish incorporating test for affine function into test polynomial
//          - add unit tests

// changes:
//      added defaults for Function
//      cleaned up unit tests
//      added lazy evaluation for fit_line

// known issues:
//      no way to distinguish between the power function f(x) -> x^1 a linear function with a = 1
mod utilities;

#[derive(Debug)]
pub struct Function {
    constant: i32,
    linear_coefficient: i32,
    polynomial: Vec<u32>,
    error: i32}

impl Function {
    pub fn evaluate(&self, x: i32) -> i32 {
        let mut output = self.constant;
        output += x * self.linear_coefficient;
        for power in self.polynomial.iter() {
            if *power > 0 {
                output += x.pow(*power);
            }
        }
        return output;
    }

    pub fn evaluate_at_points(&self, input: &[i32]) -> Vec<i32> {
        let mut output: Vec<i32> = Vec::new();
        for point in input {
            output.push(self.evaluate(*point));
        }
        return output;
    }
}

impl Default for Function {
    fn default() -> Function {
        return Function { constant: 0, linear_coefficient: 0,
                          error: 0, polynomial: vec![0]};
    }
}

pub fn fit_line(input_vector: &[i32], output_vector: &[i32]) -> Function {
    let identity_function = Function{constant: 0,
                                     linear_coefficient: 1,
                                     polynomial: Vec::new(),
                                     error: std::i32::MAX};

    let mut candidates: Vec<Function> = Vec::new();

    let (error, constant_term) = test_constant(&output_vector);
    let function = Function {constant: constant_term, error: error, ..Default::default()};
    if error == 0 {
        return function;
    } else {
        candidates.push(function);
    }

    let (error, a) = test_linear(&input_vector, &output_vector);
    let function = Function {linear_coefficient: a, error: error, ..Default::default()};
    if error == 0 {
        return function;
    } else {
        candidates.push(function);
    }

    let (error, linear_coefficient, constant) = test_affine(&input_vector, &output_vector);
    let function = Function {constant: constant, linear_coefficient: linear_coefficient,
                             error: error, ..Default::default()};
    if error == 0 {
        return function;
    } else {
        candidates.push(function);
    }

    let (error, exponent) = test_exponential(&input_vector, &output_vector);
    let function = Function {polynomial: vec![exponent], error: error, ..Default::default()};
    if error == 0 {
        return function;
    } else {
        candidates.push(function);
    }

    let (error, polynomial, a, b) = test_polynomial(&input_vector, &output_vector);
    let function = Function { constant: b, linear_coefficient: a,
                              polynomial: polynomial, error: error};
    if error == 0 {
        return function;
    } else {
        candidates.push(function);
    }

    let mut best_fit: Function = identity_function;
    for function in candidates {
        if function.error < best_fit.error {
            best_fit = function;
        }
    }
    return best_fit;
}

fn test_constant(vector: &[i32]) -> (i32, i32) {
    let first_entry = vector[0];
    for entry in vector {
        if *entry != first_entry {
            // find the average value, and create a constant vector of it
            // should represent the closest constant vector to vector
            let average = utilities::compute_mean(&vector);
            let mut constant_vector: Vec<i32> = Vec::new();
            for _count in 0..vector.len() {
                constant_vector.push(average);
            }
            let error = compute_closeness(&vector, &constant_vector);
            return (error, average)
        }
    }
    return (0, first_entry);
}

fn test_linear(input_vector: &[i32], output_vector: &[i32]) -> (i32, i32) {
    let mut is_identity = true;
    for (index, item) in input_vector.iter().enumerate() {
        if output_vector[index] != *item {
            is_identity = false;
            break;
        }
    }
    if is_identity == true {
        return (0, 1);
    }

    let sign = determine_sign(&output_vector);
    let (error, coefficient) = find_linear_approximation(&input_vector, &output_vector, sign);
    return (error, coefficient);
}

fn find_linear_approximation(input_vector: &[i32], output_vector: &[i32], sign: i32) -> (i32, i32) {
    //println!("Computing linear approximation of {:?} -> {:?} with sign {}", input_vector, output_vector, sign);
    let mut y: i32 = 2 * sign;
    let mut x: i32 = 0;
    let error = compute_closeness(&input_vector, &output_vector); // closeness to identity
    let mut candidate: (i32, i32) = (error, 1);
    let mut break_flag = false;
    let mut approximation: Vec<i32> = vec![0; input_vector.len()];
    loop {
        //println!("Test with a = {}", x + y);
        for (index, input) in input_vector.iter().enumerate() {
            approximation[index] = input * (x + y);
        }
        y *= 2;
        let new_error = compute_closeness(&output_vector, &approximation);
        if new_error < candidate.0 {
            //println!("{:?} is a better candidate than {:?}", (new_error, x + (y / 2)), candidate);
            candidate = (new_error, x + (y / 2));
            if new_error == 0 {
                //println!("Found best linear approximation: {:?}", candidate);
                return candidate;
            }
            break_flag = false;

            let look_behind = candidate.1 + (-1 * sign);
            for (index, input) in input_vector.iter().enumerate() {
                approximation[index] = input * look_behind;
            }
            let look_behind_error = compute_closeness(&output_vector, &approximation);
            if look_behind_error < candidate.0 {
                //println!("Overshot to opposite side of error curve, turning around");
                x += y / 2;
                if y > 0 {
                    y = -1;
                } else {
                    y = 1;
                }
            }
        } else { //if new_error > candidate.0 {
            if break_flag {
                break;
            }
            else {
                break_flag = true;
            }
            //println!("Overshot, resetting values to {} + {}", x + (y / 4), 1 * sign);
            x += y / 4;
            y = 1 * sign;
        }
    }
    //println!("Found best linear approximation: {:?}", candidate);
    return candidate;
}

fn determine_sign(vector: &[i32]) -> i32 {
    // starts small, gets larger -> positive a
    // starts large, gets smaller -> negative a
    // assumes vector[0] and vector[-1] contain the min/max terms (true if it is linear)
    let mut sign = 1;
    let final_entry = vector.len() - 1;
    if vector[0] > vector[final_entry] {
        sign = -1;
    } else if vector[0] == vector[final_entry] {
        if vector[0] < 0 {
            sign = -1;
        }
    }
    return sign;
}

fn test_affine(input_vector: &[i32], output_vector: &[i32]) -> (i32, i32, i32) {
    let vector2 = successive_differences(&output_vector);
    let input_vector2 = successive_differences(&input_vector);
    let (_linear_error, coefficient) = test_linear(&input_vector2, &vector2);
    let (_constant_error, constant) = determine_constant(&input_vector, &output_vector, &coefficient);

    let mut test_vector: Vec<i32> = Vec::new();
    for input in input_vector {
        test_vector.push((input * coefficient) + constant);
    }
    let error = compute_closeness(&test_vector, &output_vector);
    return (error, coefficient, constant);
}

fn successive_differences(vector: &[i32]) -> Vec<i32> {
    let mut vector2: Vec<i32> = Vec::new();
    let vector_size = vector.len();
    for index in 0..(vector_size - 1) {
        vector2.push(vector[(index + 1) % vector_size] - vector[index]);
    }
    return vector2;
}

fn determine_constant(input_vector: &[i32], output_vector: &[i32], coefficient: &i32) -> (i32, i32) {
    let output_vector_size = output_vector.len();
    let mut constant_vector: Vec<i32> = vec![0; output_vector_size];
    for index in 0..output_vector_size {
        constant_vector[index] = output_vector[index] - (input_vector[index] * coefficient);
    }
    return test_constant(&constant_vector);
}

fn test_exponential(input_vector: &[i32], output_vector: &[i32]) -> (i32, u32) {
    return find_exponent_approximation(&input_vector, &output_vector);
}

fn find_exponent_approximation(input_vector: &[i32], output_vector: &[i32]) -> (i32, u32) {
    // if negatives are in input and negatives are in output, then only test odd exponents
    // if negatives are in input but not in output, then only test even exponents
    // if negattives are not in input, then all outputs must be positive, test all exponents
    //println!("Computing exponent approximation of {:?} -> {:?}", input_vector, output_vector);
    let mut negative_input = false;
    for value in input_vector {
        if *value < 0 {
            negative_input = true;
        }
    }
    let mut negative_output = false;
    for value in output_vector {
        if *value < 0 {
            negative_output = true;
        }
    }

    let mut test_type: i32 = 0;
    if negative_input == true {
        if negative_output == true {
            test_type = 1;
        } else {
            test_type = 2;
        }
    }

    let sign: i32 = 1;
    let mut y: i32 = 2 * sign;
    if test_type == 1 {
        y = 3;
    }

    let mut x: i32 = 0;
    let error = compute_closeness(&input_vector, &output_vector); // closeness to identity
    let mut candidate: (i32, i32) = (error, 1);
    let mut break_flag = false;
    let mut approximation: Vec<i32> = vec![0; input_vector.len()];
    loop {
        //println!("Test with a = {} + {} = {}", x, y, x + y);
        for (index, input) in input_vector.iter().enumerate() {
            approximation[index] = input.pow((x + y) as u32);
        }
        y *= 2;
        if test_type == 1 {
            y -= 1;
        }
        let new_error = compute_closeness(&output_vector, &approximation);
        //println!("Error between {:?} and {:?}: {}", output_vector, approximation, new_error);
        if new_error < candidate.0 {
            //println!("{:?} is a better candidate than {:?}", (new_error, x + ((y + 1) / 2)), candidate);
            candidate = (new_error, x + ((y + 1) / 2));
            if new_error == 0 {
                //println!("Found best approximation: {:?}", candidate);
                break;
            }
            break_flag = false;
            let look_behind = candidate.1 + (-1 * sign);
            for (index, input) in input_vector.iter().enumerate() {
                approximation[index] = input.pow(look_behind as u32);
            }
            let look_behind_error = compute_closeness(&output_vector, &approximation);
            if look_behind_error < candidate.0 {
                //println!("Overshot to opposite side of error curve, turning around");
                x += (y + 1) / 2;
                if y > 0 {
                    y = -1;
                } else {
                    y = 1;
                }
                if test_type == 2 {
                    y *= 2;
                }
            }
        } else { //if new_error > candidate.0 {
            if break_flag {
                break;
            }
            else {
                break_flag = true;
            }
            //println!("Previous best: {:?}, test that overshot: {:?}", candidate, (new_error,x + (y / 2)));
            x += (y + 1) / 4;
            y = 1 * sign;
            if test_type == 2 {
                y *= 2;
            }
            //println!("Overshot, resetting values to {} + {}", x, y);
            if x + y == candidate.1 {
                y += 2;
            }
        }
    }
    //println!("Found best exponent approximation: {:?}", candidate);
    return (candidate.0, candidate.1 as u32);
}

fn test_polynomial(input_vector: &[i32], output_vector: &[i32]) -> (i32, Vec<u32>, i32, i32) {
    let mut polynomial: Vec<u32> = Vec::new();
    let length = input_vector.len();
    let mut state = output_vector.to_vec();
    let mut linear_coefficient: i32 = 0;
    let mut constant: i32 = 0;
    loop {
        let (_error, exponent) = test_exponential(&input_vector, &state);
        //println!("Found closest exponent approximation of {:?}: {}", state, exponent);
        polynomial.push(exponent);
        let mut break_flag = true;
        for index in 0..length {
            //println!("Reducing state to {} = {} - {} ** {}", state[index] - input_vector[index].pow(exponent),
            //                                                 state[index], input_vector[index], exponent);
            state[index] = state[index] - input_vector[index].pow(exponent);
            if state[index] != 0 {
                break_flag = false;
            }
        }
        //println!("State is now: {:?}; Polynomial: {:?}", state, polynomial);
        if exponent == 1 {
            break;
        } else if break_flag == true {
            break;
        }
        let (error, a, b) = test_affine(&input_vector, &state);
        if error == 0 && a != 1 {
            linear_coefficient = a;
            constant = b;
            println!("Found affine variables from {:?}: {} {}", state, a, b);
            break;
        }
    }
    let (min, max) = utilities::compute_range(&state);
    let mut error = -1;
    if min == 0 {
        error = max;
    } else {
        error = (min + max) / 2;
    }
    assert!(error != -1);
    return (error.abs(), polynomial, linear_coefficient, constant);
}

fn compute_closeness(vector1: &[i32], vector2: &[i32]) -> i32 {
    assert!(vector1.len() == vector2.len());
    let error_vector = compute_difference(&vector1, &vector2);
    //let mut sum: i32 = 0;
    //for number in error_vector {
    //    sum += number;
    //}
    //return sum;
    let (min, max) = utilities::compute_range(&error_vector);
    if min == 0 {
        return max;
    }
    assert!((min + max) / 2 > 0);
    return (min + max) / 2;
}

fn compute_difference(vector1: &[i32], vector2: &[i32]) -> Vec<i32> {
    let mut output: Vec<i32> = Vec::new();
    for (index, item) in vector1.iter().enumerate() {
        output.push((*item - vector2[index]).abs());
    }
    return output;
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_unit_test() {
        {let constant_vector = vec![10, 10, 10, 10, 10];
        let (quality, constant_value) = test_constant(&constant_vector);
        assert_eq!(quality, 0);
        assert_eq!(constant_vector[0], constant_value);}
    }

    #[test]
    fn test_linear_unit_test() {
        let input_vector: [i32; 9] = [-5, -3, -2, -1, 0, 1, 2, 3, 5];
        let tests: Vec<i32> = vec![1, 2, 3, -3, -10];
        let mut function = Function {constant: 0, linear_coefficient: 0,
                                     error: 0, polynomial: Vec::new()};
        for coefficient in tests {
            function.linear_coefficient = coefficient;
            let linear_vector = function.evaluate_at_points(&input_vector);
            let (error, a) = test_linear(&input_vector, &linear_vector);
            assert_eq!(error, 0);
            assert_eq!(a, coefficient);
        }

        let input_vector: [i32; 5] = [1, 2, 3, 4, 5];

        {let test_vector = vec![-17, -10, -7, -3, -1];
        let (error, _a) = test_linear(&input_vector, &test_vector);
        //println!("Approximation of {:?}: {} with error {}", test_vector, a, error);
        assert!(error > 0);}

        {let linear_vector = vec![1, 3, 7, 10, 17]; // test with coprime values
        let (error, _a) = test_linear(&input_vector, &linear_vector);
        //println!("Best linear approximation to {:?}: {} with error {}", linear_vector, a, error);
        assert!(error > 0);}

        {let linear_vector = vec![123, 347, 520, 797, 960];
        let (error, _a) = test_linear(&input_vector, &linear_vector);
        //println!("Best linear approximation to {:?}: {} with error {}", linear_vector, a, error);
        assert!(error > 0);}

        {let linear_vector = vec![-123, -347, -520, -797, -960];
        let (error, _a) = test_linear(&input_vector, &linear_vector);
        //println!("Best linear approximation to {:?}: {} with error {}", linear_vector, a, error);
        assert!(error > 0);}
    }

    #[test]
    fn test_affine_unit_test() {
        let input_vector: [i32; 9] = [-5, -3, -2, -1, 0, 1, 2, 3, 5];
        let mut function: Function = Function {constant: 0, linear_coefficient: 0,
                                               error: 0, polynomial: Vec::new()};

        let tests: Vec<(i32, i32)> = vec![(5, 3), (-5, 3), (5, -3), (-5, -3), (-10, 100)];
        for test in tests {
            function.constant = test.1;
            function.linear_coefficient = test.0;
            let affine_vector: Vec<i32> = function.evaluate_at_points(&input_vector);
            let (error, a, b) = test_affine(&input_vector, &affine_vector);
            assert_eq!(error, 0);
            assert_eq!(a, test.0);
            assert_eq!(b, test.1);
        }
    }

    #[test]
    fn test_exponential_unit_test() {
        let input_vector: [i32; 9] = [-10, -7, -4, -1, 0, 1, 2, 5, 8];
        let mut function: Function = Function {constant: 0, linear_coefficient: 0,
                                               error: 0, polynomial: vec![0; 1]};
        let tests: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        for test in tests {
            function.polynomial[0] = test;
            let mut vector: Vec<i32> = function.evaluate_at_points(&input_vector);
            //println!("\nComputing approximation of exponent = {}", test);
            let (error, exponent) = test_exponential(&input_vector, &vector);
            //println!("exponential approximation of {}: {} with error {}", test, exponent, error);
            assert_eq!(error, 0);
            assert_eq!(exponent, test);
            //let mut vector: Vec<i32> = function.evaluate_at_points(&input_vector);
            //for index in 0..vector.len() {
            //    vector[index] = vector[index] + index as i32;
            //}
            //let (error, exponent) = test_exponential(&input_vector, &vector);
            //println!("Exponential approximation of noisy {}: {} with error {}", test, exponent, error);
        }
    }

    #[test]
    fn test_polynomial_unit_test() {
        use std::collections::HashSet;
        let input_vector: [i32; 15] = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7];
        let mut function: Function = Function { constant: 0, linear_coefficient: 0,
                                                error: 0, polynomial: vec![1, 0, 0, 0] };
        let tests: Vec<Vec<u32>> = vec![vec![1, 0, 0, 0],
                                        vec![2, 0, 0, 0],
                                        vec![1, 2, 0, 0],
                                        vec![1, 3, 5, 0],
                                        vec![2, 4, 6, 0],
                                        vec![2, 3, 5, 7]];
        for test in tests {
            function.polynomial = test;
            let vector = function.evaluate_at_points(&input_vector);
            //println!("\nComputing polynomial approximation of {:?} -> {:?}", input_vector, vector);
            let (error, polynomial, a, b) = test_polynomial(&input_vector, &vector);
            //println!("Best polynomial approximation of {:?} -> {:?}: {:?} with error {}", input_vector, vector, polynomial, error);
            let mut test_set: HashSet<u32> = function.polynomial.into_iter().collect();
            let found_set: HashSet<u32> = polynomial.into_iter().collect();
            test_set.remove(&0);
            assert_eq!(test_set, found_set);
            assert_eq!(error, 0);
        }
    }

    #[test]
    fn test_fit_line() {
        let input_vector: [i32; 15] = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7];
        {
            for constant in vec![0, -10, 10] {
                let constant_function = Function { constant: constant, ..Default::default()};
                let samples = constant_function.evaluate_at_points(&input_vector);
                let best_fit = fit_line(&input_vector, &samples);
                //println!("Best fit: {}, True constant: {}", best_fit.constant, constant_function.constant);
                assert_eq!(best_fit.constant, constant_function.constant);
            }
        }

        {
            for linear_coefficient in vec![-10, -1, 0, 1, 10] {
                let linear_function = Function { linear_coefficient: linear_coefficient,
                                                ..Default::default()};
                let samples = linear_function.evaluate_at_points(&input_vector);
                let best_fit = fit_line(&input_vector, &samples);
                assert_eq!(best_fit.linear_coefficient, linear_function.linear_coefficient);
            }
        }

        {
            for values in vec![(0, 0), (10, 5), (10, 15), (10, 0), (10, -5),
                               (-10, -5), (-10, -15), (-10, 0), (-10, 5)] {
                let affine_function = Function { constant: values.1, linear_coefficient: values.0,
                                                 ..Default::default()};
                let samples = affine_function.evaluate_at_points(&input_vector);
                let best_fit = fit_line(&input_vector, &samples);
                //println!("Best fit for {:?} -> {:?} ({:?}): {:?}", input_vector, samples, affine_function, best_fit);
                assert_eq!(best_fit.constant, affine_function.constant);
                assert_eq!(best_fit.linear_coefficient, affine_function.linear_coefficient);
            }
        }

        {
            for power in vec![0, 2, 3, 4, 5] { // x^1 = x, so best_fit would be linear
                let power_function = Function { polynomial: vec![power], ..Default::default()};
                let samples = power_function.evaluate_at_points(&input_vector);
                let best_fit = fit_line(&input_vector, &samples);
                //println!("Best fit for {:?} -> {:?} ({:?}): {:?}", input_vector, samples,
                //                                                   power_function, best_fit);
                assert_eq!(power_function.polynomial, best_fit.polynomial);
            }
        }

        }
}

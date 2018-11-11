use std::fs;

pub fn write_samples_to_file(filename: &str, inputs: &[i32], outputs: &[i32]) -> () {
    let mut contents = String::new();
    assert_eq!(inputs.len(), outputs.len());
    for index in 0..inputs.len() {
        contents += &format!("{} {}\n", inputs[index], outputs[index]);
    }
    write_to_file(filename, &contents);
}

pub fn write_to_file(filename: &str, contents: &str) -> () {
    fs::write(filename, contents).expect(&format!("Unable to write to {}", filename));
}

pub fn compute_mean(vector: &[i32]) -> i32 {
    let mut output = 0;
    for number in vector {
        output += number;
    }
    let length = vector.len();
    return output / (length as i32);
}

pub fn compute_range(vector: &[i32]) -> (i32, i32) {
    let mut min = vector[0];
    let mut max = vector[0];
    for item in vector {
        if *item < min {
            min = *item;
        } else if *item > max {
            max = *item;
        }
    }
    return (min, max);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mean_unit_test() {
        let vector = vec![0, 1, 2, 3, 4, 5, 6, -7, -8, -9, -10, 3];
        let mode = compute_mean(&vector);
        assert_eq!(mode, 0);
    }
}

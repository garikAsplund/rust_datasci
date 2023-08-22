use std::fs::File;
use std::io::Write;

use linfa_trees::{SplitQuality, DecisionTree };
use ndarray::Array2;
use ndarray::prelude::*;
use linfa::prelude::*;

fn main() {
    let original_data: Array2<f32> = array! (
        [1., 1., 1000., 1., 10.],
        [1., 1., 1000., 1., 10.],
        [1., 1., 1000., 1., 10.],
        [1., 1., 1000., 1., 10.],
        [1., 1., 1000., 1., 10.],
        [1., 1., 1000., 1., 10.],
        [1., 1., 1000., 1., 10.],
        [1., 1., 1000., 1., 10.],
        [1., 1., 1000., 1., 10.],
        [1., 1., 1000., 1., 10.],
        [1., 1., 1000., 1., 10.],
        [1., 1., 1000., 1., 10.],
        [1., 1., 1000., 1., 10.],
        [1., 1., 1000., 1., 10.],
        [1., 1., 1000., 1., 10.],
        [1., 1., 1000., 1., 10.],
        [1., 1., 1000., 1., 10.],

    );

    let feature_names = vec!["Watched TV", "Pet Cat", "Rust LOC", "Ate Pizza"];

    let num_features = original_data.len_of(Axis((1))) - 1;
    let features = original_data.slice(s![.., 0..num_features]).to_owned();
    let labels = original_data.column(num_features).to_owned();

    let linfa_dataset = Dataset::new(features, labels)
        .map_targets(|x| match x.to_owned() as i32 {
            i32::MIN..=4 => "Sad",
            5..=7 => "Ok",
            8..=i32::MAX => "Happy",
    })
    .with_feature_names(feature_names);

    let model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .fit(&linfa_dataset)
        .unwrap();

    File::create("dt.tex")
        .unwrap()
        .write_all(model.export_to_tikz().with_legend().to_string().as_bytes())
        .unwrap();
}

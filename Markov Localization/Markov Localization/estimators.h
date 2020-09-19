#ifndef ESTIMATORS_H_
#define ESTIMATORS_H_

#include <vector>
#include <algorithm>
#include "helpers.h"



// observation model: calculate likelihood prob term based on landmark proximity
float observation_model(vector<float> landmark_positions,
    vector<float> observations, vector<float> pseudo_ranges,
    float distance_max, float observation_stdev) {
    // initialize observation probability
    float distance_prob = 1.0f;

    // run over current observation vector
    for (int z = 0; z < observations.size(); ++z) {
        // define min distance
        float pseudo_range_min;

        // check, if distance vector exists
        if (pseudo_ranges.size() > 0) {
            // set min distance
            pseudo_range_min = pseudo_ranges[0];
            // remove this entry from pseudo_ranges-vector
            pseudo_ranges.erase(pseudo_ranges.begin());
        }
        else {  // no or negative distances: set min distance to a large number
            pseudo_range_min = std::numeric_limits<const float>::infinity();
        }

        // estimate the probability for observation model, this is our likelihood 
        distance_prob *= Helpers::normpdf(observations[z], pseudo_range_min,
            observation_stdev);
    }

    return distance_prob;
}

vector<float> pseudo_range_estimator(vector<float> landmark_positions,
    float pseudo_position) {
    // define pseudo observation vector
    vector<float> pseudo_ranges;

    // loop over number of landmarks and estimate pseudo ranges
    for (int l = 0; l < landmark_positions.size(); ++l) {
        // estimate pseudo range for each single landmark 
        // and the current state position pose_i:
        float range_l = landmark_positions[l] - pseudo_position;

        // check if distances are positive: 
        if (range_l > 0.0f) {
            pseudo_ranges.push_back(range_l);
        }
    }

    // sort pseudo range vector
    sort(pseudo_ranges.begin(), pseudo_ranges.end());

    return pseudo_ranges;
}

// motion model: calculates prob of being at an estimated position at time t
float motion_model(float pseudo_position, float movement, vector<float> priors,
    int map_size, int control_stdev) {
    // initialize probability
    float position_prob = 0.0f;

    // loop over state space for all possible positions x (convolution):
    for (float j = 0; j < map_size; ++j) {
        float next_pseudo_position = j;
        // distance from i to j
        float distance_ij = pseudo_position - next_pseudo_position;

        // transition probabilities:
        float transition_prob = Helpers::normpdf(distance_ij, movement,
            control_stdev);
        // estimate probability for the motion model, this is our prior
        position_prob += transition_prob * priors[j];
    }

    return position_prob;
}

// initialize priors assuming vehicle at landmark +/- 1.0 meters position stdev
vector<float> initialize_priors(int map_size, vector<float> landmark_positions,
    float position_stdev) {
    // set all priors to 0.0
    vector<float> priors(map_size, 0.0);

    // set each landmark positon +/-1 to 1.0/9.0 (9 possible postions)
    float norm_term = landmark_positions.size() * (position_stdev * 2 + 1);
    for (int i = 0; i < landmark_positions.size(); ++i) {
        for (float j = 1; j <= position_stdev; ++j) {
            priors.at(int(j + landmark_positions[i] + map_size) % map_size) += 1.0 / norm_term;
            priors.at(int(-j + landmark_positions[i] + map_size) % map_size) += 1.0 / norm_term;
        }
        priors.at(landmark_positions[i]) += 1.0 / norm_term;
    }

    return priors;
}

vector<float> nomalized(vector<float> input) {
    vector<float> output;
    output = Helpers::normalize_vector(input);
    return output;
}
#endif ESTIMATORS_H_
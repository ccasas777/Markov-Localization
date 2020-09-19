#include <iostream>
#include <vector>

#include "estimators.h"

using std::vector;
using std::cout;
using std::endl;


int main() {
    // set standard deviation of control
    float control_stdev = 1.0f;

    // set standard deviation of position
    float position_stdev = 1.0f;

    // meters vehicle moves per time step
    float movement_per_timestep = 1.0f;

    // set observation standard deviation
    float observation_stdev = 1.0f;

    // number of x positions on map
    int map_size = 26;

    // set distance max
    float distance_max = map_size;

    // define landmarks
    vector<float> landmark_positions{ 3, 9, 14, 23 ,25};

    // define observations vector, each inner vector represents a set 
    //   of observations for a time step
    vector<vector<float> > sensor_obs{ {1,7,12,21,23},
                                       {0,6,11,20,22}, 
                                       {5,10,19,21},
                                       {4,9,18,20}, 
                                       {3,8,17,19}, 
                                       {2,7,16,18}, 
                                       {1,6,15,17},
                                       {0,5,14,16}, 
                                       {4,13,15}, 
                                       {3,12,14},
                                       {2,11,13}, 
                                       {1,10,12},
                                       {0,9,11}, 
                                       {8,10}, 
                                       {7,9}, 
                                       {6,8}, 
                                       {5,7}, 
                                       {4,6}, 
                                       {3,5},
                                       {2,4},
                                       {1,3}, 
                                       {0,2}};

    // initialize priors
    vector<float> priors = initialize_priors(map_size, landmark_positions, position_stdev);

    //initialize posteriors    
    vector<float> posteriors(map_size, 0.0);

    // specify time steps
    int time_steps = sensor_obs.size();

    // declare observations vector
    vector<float> observations;

    // cycle through time steps
    for (int t = 0; t < time_steps; ++t) {

        vector<float> true_location(map_size, 0);
        true_location[t + 2] = 1;
        //UNCOMMENT TO SEE THIS STEP OF THE FILTER
        cout << "---------------TIME STEP---------------" << endl;
        cout << "t = " << t << endl;
        cout << "index-----Motion_prob----------OBS--------------PRODUCT----" << endl;

        if (!sensor_obs[t].empty()) {
            observations = sensor_obs[t];            
        }
        else {
            observations = { float(distance_max) };
            cout << "no obervation!!" << endl;
        }

        // step through each pseudo position x (i)
        for (unsigned int i = 0; i < map_size; ++i) {
            float pseudo_position = float(i);

          
            //get the motion model probability for each x position            
            float motion_prob = motion_model(pseudo_position, movement_per_timestep,
                priors, map_size, control_stdev);
            
           //get pseudo ranges           
            vector<float> pseudo_ranges = pseudo_range_estimator(landmark_positions,
                pseudo_position);

          
            //get observation probability             
            float observation_prob = observation_model(landmark_positions, observations,
                pseudo_ranges, distance_max,
                observation_stdev);

            
            //calculate the ith posterior            
            posteriors[i] = motion_prob * observation_prob;

            //Print result and details
            cout << i << "\t" << motion_prob << "\t" << observation_prob << "\t"
                << "\t" << motion_prob * observation_prob << endl;
        }

        //normalize        
        posteriors = nomalized(posteriors);
        //print predictions
        cout << "----------NORMALIZED---------------" << endl;
        cout << "index---current_Location_prob----true_location" << endl;
        priors = posteriors;

        // print posteriors vectors to stdout
        for (int p = 0; p < posteriors.size(); ++p) {
            cout << p << "\t" << posteriors[p] << "\t \t" << true_location[p] << endl;
        }
    }

    return 0;
}

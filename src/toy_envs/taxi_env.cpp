#include "toy_envs/taxi_env.h"

using namespace std;

namespace thts {
    TaxiEnv::TaxiEnv(int width, int height) :
        ThtsEnv(true),
        width(width),
        height(height),
        cached_actions(make_shared<IntActionVector>())
    {
        cached_actions->push_back(make_shared<const IntAction>(TAXI_SOUTH));
        cached_actions->push_back(make_shared<const IntAction>(TAXI_NORTH));
        cached_actions->push_back(make_shared<const IntAction>(TAXI_EAST));
        cached_actions->push_back(make_shared<const IntAction>(TAXI_WEST));
        cached_actions->push_back(make_shared<const IntAction>(TAXI_PICKUP));
        cached_actions->push_back(make_shared<const IntAction>(TAXI_DROPOFF));
    }

    pair<int,int> TaxiEnv::get_landmark(int landmark_id) const {
        if (landmark_id == 0) return {0, 0}; // R
        if (landmark_id == 1) return {0, 4}; // G
        if (landmark_id == 2) return {4, 0}; // Y
        return {4, 3}; // B
    }

    int TaxiEnv::encode_passenger_and_destination(int passenger_status, int destination_id) const {
        return passenger_status * 4 + destination_id;
    }

    int TaxiEnv::get_passenger_status(shared_ptr<const Int3TupleState> state) const {
        int encoded = get<2>(state->state);
        return encoded / 4;
    }

    int TaxiEnv::get_destination_id(shared_ptr<const Int3TupleState> state) const {
        int encoded = get<2>(state->state);
        return encoded % 4;
    }

    bool TaxiEnv::can_move_east(int row, int col) const {
        if (col >= width-1) return false;

        // Internal vertical walls from the standard Taxi map.
        if ((row == 0 || row == 1) && col == 1) return false;
        if ((row == 3 || row == 4) && (col == 0 || col == 2)) return false;

        return true;
    }

    bool TaxiEnv::can_move_west(int row, int col) const {
        if (col <= 0) return false;
        return can_move_east(row, col-1);
    }

    shared_ptr<const Int3TupleState> TaxiEnv::get_initial_state() const {
        // Start at R, passenger at Y, destination G.
        int encoded = encode_passenger_and_destination(2, 1);
        return make_shared<const Int3TupleState>(0, 0, encoded);
    }

    bool TaxiEnv::is_sink_state(shared_ptr<const Int3TupleState> state) const {
        int row = get<0>(state->state);
        int col = get<1>(state->state);
        int passenger_status = get_passenger_status(state);
        int destination_id = get_destination_id(state);

        if (passenger_status == 4) return false;

        auto destination = get_landmark(destination_id);
        return row == destination.first && col == destination.second && passenger_status == destination_id;
    }

    shared_ptr<IntActionVector> TaxiEnv::get_valid_actions(shared_ptr<const Int3TupleState> state) const {
        if (is_sink_state(state)) return make_shared<IntActionVector>();
        return cached_actions;
    }

    shared_ptr<Int3TupleStateDistr> TaxiEnv::get_transition_distribution(
        shared_ptr<const Int3TupleState> state, shared_ptr<const IntAction> action) const
    {
        int row = get<0>(state->state);
        int col = get<1>(state->state);
        int passenger_status = get_passenger_status(state);
        int destination_id = get_destination_id(state);

        int next_row = row;
        int next_col = col;
        int next_passenger_status = passenger_status;

        if (action->action == TAXI_SOUTH) {
            if (row < height-1) next_row++;
        } else if (action->action == TAXI_NORTH) {
            if (row > 0) next_row--;
        } else if (action->action == TAXI_EAST) {
            if (can_move_east(row, col)) next_col++;
        } else if (action->action == TAXI_WEST) {
            if (can_move_west(row, col)) next_col--;
        } else if (action->action == TAXI_PICKUP) {
            if (passenger_status != 4) {
                auto passenger_loc = get_landmark(passenger_status);
                if (row == passenger_loc.first && col == passenger_loc.second) {
                    next_passenger_status = 4;
                }
            }
        } else if (action->action == TAXI_DROPOFF) {
            if (passenger_status == 4) {
                auto destination = get_landmark(destination_id);
                if (row == destination.first && col == destination.second) {
                    next_passenger_status = destination_id;
                } else {
                    // Wrong dropoff at a landmark: passenger exits taxi to current landmark.
                    for (int landmark_id=0; landmark_id<4; landmark_id++) {
                        auto landmark = get_landmark(landmark_id);
                        if (row == landmark.first && col == landmark.second) {
                            next_passenger_status = landmark_id;
                            break;
                        }
                    }
                }
            }
        }

        int encoded = encode_passenger_and_destination(next_passenger_status, destination_id);
        shared_ptr<const Int3TupleState> next_state = make_shared<const Int3TupleState>(next_row, next_col, encoded);
        shared_ptr<Int3TupleStateDistr> distr = make_shared<Int3TupleStateDistr>();
        distr->insert_or_assign(next_state, 1.0);
        return distr;
    }

    shared_ptr<const Int3TupleState> TaxiEnv::sample_transition_distribution(
        shared_ptr<const Int3TupleState> state,
        shared_ptr<const IntAction> action,
        RandManager& rand_manager) const
    {
        shared_ptr<Int3TupleStateDistr> distr = get_transition_distribution(state, action);
        return distr->begin()->first;
    }

    double TaxiEnv::get_reward(
        shared_ptr<const Int3TupleState> state,
        shared_ptr<const IntAction> action,
        shared_ptr<const Int3TupleState> observation) const
    {
        int row = get<0>(state->state);
        int col = get<1>(state->state);
        int passenger_status = get_passenger_status(state);
        int destination_id = get_destination_id(state);

        if (action->action == TAXI_PICKUP) {
            if (passenger_status == 4) return -10.0;
            auto passenger_loc = get_landmark(passenger_status);
            if (row == passenger_loc.first && col == passenger_loc.second) return -1.0;
            return -10.0;
        }

        if (action->action == TAXI_DROPOFF) {
            if (passenger_status != 4) return -10.0;
            auto destination = get_landmark(destination_id);
            if (row == destination.first && col == destination.second) return 20.0;

            for (int landmark_id=0; landmark_id<4; landmark_id++) {
                auto landmark = get_landmark(landmark_id);
                if (row == landmark.first && col == landmark.second) return -10.0;
            }
            return -10.0;
        }

        return -1.0;
    }
}

namespace thts {
    shared_ptr<Int3TupleStateDistr> TaxiEnv::get_observation_distribution(
        shared_ptr<const IntAction> action, shared_ptr<const Int3TupleState> next_state) const
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<ObservationDistr> distr_itfc = ThtsEnv::get_observation_distribution_itfc(act_itfc, next_state_itfc);
        shared_ptr<Int3TupleStateDistr> distr = make_shared<Int3TupleStateDistr>();
        for (pair<const shared_ptr<const Observation>,double> pr : *distr_itfc) {
            shared_ptr<const Int3TupleState> obsv = static_pointer_cast<const Int3TupleState>(pr.first);
            distr->insert_or_assign(obsv, pr.second);
        }
        return distr;
    }

    shared_ptr<const Int3TupleState> TaxiEnv::sample_observation_distribution(
        shared_ptr<const IntAction> action,
        shared_ptr<const Int3TupleState> next_state,
        RandManager& rand_manager) const
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<const Observation> obsv_itfc = ThtsEnv::sample_observation_distribution_itfc(
            act_itfc,
            next_state_itfc,
            rand_manager);
        return static_pointer_cast<const Int3TupleState>(obsv_itfc);
    }

    shared_ptr<ThtsEnvContext> TaxiEnv::sample_context(shared_ptr<const Int3TupleState> state) const {
        shared_ptr<const State> state_itfc = static_pointer_cast<const State>(state);
        shared_ptr<ThtsEnvContext> context = ThtsEnv::sample_context_itfc(state_itfc);
        return static_pointer_cast<ThtsEnvContext>(context);
    }
}

namespace thts {
    shared_ptr<const State> TaxiEnv::get_initial_state_itfc() const {
        shared_ptr<const Int3TupleState> init_state = get_initial_state();
        return static_pointer_cast<const State>(init_state);
    }

    bool TaxiEnv::is_sink_state_itfc(shared_ptr<const State> state) const {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        return is_sink_state(state_itfc);
    }

    shared_ptr<ActionVector> TaxiEnv::get_valid_actions_itfc(shared_ptr<const State> state) const {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        shared_ptr<IntActionVector> valid_actions_itfc = get_valid_actions(state_itfc);

        shared_ptr<ActionVector> valid_actions = make_shared<ActionVector>();
        for (shared_ptr<const IntAction> act : *valid_actions_itfc) {
            valid_actions->push_back(static_pointer_cast<const Action>(act));
        }
        return valid_actions;
    }

    shared_ptr<StateDistr> TaxiEnv::get_transition_distribution_itfc(
        shared_ptr<const State> state,
        shared_ptr<const Action> action) const
    {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<Int3TupleStateDistr> distr_itfc = get_transition_distribution(state_itfc, action_itfc);

        shared_ptr<StateDistr> distr = make_shared<StateDistr>();
        for (pair<shared_ptr<const Int3TupleState>,double> key_val_pair : *distr_itfc) {
            shared_ptr<const State> obsv = static_pointer_cast<const State>(key_val_pair.first);
            distr->insert_or_assign(obsv, key_val_pair.second);
        }
        return distr;
    }

    shared_ptr<const State> TaxiEnv::sample_transition_distribution_itfc(
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        RandManager& rand_manager) const
    {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const Int3TupleState> obsv = sample_transition_distribution(state_itfc, action_itfc, rand_manager);
        return static_pointer_cast<const State>(obsv);
    }

    shared_ptr<ObservationDistr> TaxiEnv::get_observation_distribution_itfc(
        shared_ptr<const Action> action,
        shared_ptr<const State> next_state) const
    {
        shared_ptr<const IntAction> act_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const Int3TupleState> next_state_itfc = static_pointer_cast<const Int3TupleState>(next_state);
        shared_ptr<Int3TupleStateDistr> distr_itfc = get_observation_distribution(act_itfc, next_state_itfc);
        shared_ptr<ObservationDistr> distr = make_shared<ObservationDistr>();
        for (pair<const shared_ptr<const Int3TupleState>,double> pr : *distr_itfc) {
            shared_ptr<const Observation> obsv = static_pointer_cast<const Observation>(pr.first);
            distr->insert_or_assign(obsv, pr.second);
        }
        return distr;
    }

    shared_ptr<const Observation> TaxiEnv::sample_observation_distribution_itfc(
        shared_ptr<const Action> action,
        shared_ptr<const State> next_state,
        RandManager& rand_manager) const
    {
        shared_ptr<const IntAction> act_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const Int3TupleState> next_state_itfc = static_pointer_cast<const Int3TupleState>(next_state);
        shared_ptr<const Int3TupleState> obsv_itfc = sample_observation_distribution(act_itfc, next_state_itfc, rand_manager);
        return static_pointer_cast<const Observation>(obsv_itfc);
    }

    double TaxiEnv::get_reward_itfc(
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        shared_ptr<const Observation> observation) const
    {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        shared_ptr<const IntAction> action_itfc = static_pointer_cast<const IntAction>(action);
        shared_ptr<const Int3TupleState> obsv_itfc = static_pointer_cast<const Int3TupleState>(observation);
        return get_reward(state_itfc, action_itfc, obsv_itfc);
    }

    shared_ptr<ThtsEnvContext> TaxiEnv::sample_context_itfc(shared_ptr<const State> state) const {
        shared_ptr<const Int3TupleState> state_itfc = static_pointer_cast<const Int3TupleState>(state);
        shared_ptr<ThtsEnvContext> context = sample_context(state_itfc);
        return static_pointer_cast<ThtsEnvContext>(context);
    }
}

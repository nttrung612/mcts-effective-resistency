#include "toy_envs/deep_sea_env.h"

#include <random>

using namespace std;

namespace thts {
    DeepSeaEnv::DeepSeaEnv(int N, unsigned int action_mapping_seed) :
        ThtsEnv(true),
        N(N),
        right_cost(0.01 / static_cast<double>(N)),
        action_mapping(N),
        cached_actions(make_shared<IntActionVector>())
    {
        cached_actions->push_back(make_shared<const IntAction>(0));
        cached_actions->push_back(make_shared<const IntAction>(1));

        mt19937 gen(action_mapping_seed);
        bernoulli_distribution flip(0.5);
        for (int col = 0; col < N; col++) {
            action_mapping[col] = flip(gen) ? 1 : 0;
        }
    }

    bool DeepSeaEnv::is_right_action(int col, int action) const {
        if (col < 0 || col >= N) return false;
        return action == action_mapping[col];
    }

    shared_ptr<const Int3TupleState> DeepSeaEnv::get_initial_state() const {
        return make_shared<const Int3TupleState>(0, 0, 0);
    }

    bool DeepSeaEnv::is_sink_state(shared_ptr<const Int3TupleState> state) const {
        return get<0>(state->state) >= N - 1;
    }

    shared_ptr<IntActionVector> DeepSeaEnv::get_valid_actions(shared_ptr<const Int3TupleState> state) const {
        if (is_sink_state(state)) return make_shared<IntActionVector>();
        return cached_actions;
    }

    shared_ptr<Int3TupleStateDistr> DeepSeaEnv::get_transition_distribution(
        shared_ptr<const Int3TupleState> state, shared_ptr<const IntAction> action) const
    {
        int row = get<0>(state->state);
        int col = get<1>(state->state);
        bool move_right = is_right_action(col, action->action);
        int next_row = row + 1;
        if (next_row > N - 1) next_row = N - 1;
        int next_col = move_right ? col + 1 : col;
        if (next_col > N - 1) next_col = N - 1;
        shared_ptr<const Int3TupleState> next_state = make_shared<const Int3TupleState>(next_row, next_col, 0);
        shared_ptr<Int3TupleStateDistr> distr = make_shared<Int3TupleStateDistr>();
        distr->insert_or_assign(next_state, 1.0);
        return distr;
    }

    shared_ptr<const Int3TupleState> DeepSeaEnv::sample_transition_distribution(
        shared_ptr<const Int3TupleState> state, shared_ptr<const IntAction> action, RandManager& /*rand_manager*/) const
    {
        shared_ptr<Int3TupleStateDistr> distr = get_transition_distribution(state, action);
        return distr->begin()->first;
    }

    double DeepSeaEnv::get_reward(
        shared_ptr<const Int3TupleState> state,
        shared_ptr<const IntAction> action,
        shared_ptr<const Int3TupleState> /*observation*/) const
    {
        int row = get<0>(state->state);
        int col = get<1>(state->state);
        bool move_right = is_right_action(col, action->action);
        double reward = move_right ? -right_cost : 0.0;
        if (move_right && row == N - 2 && col == N - 2) {
            reward += 1.0;
        }
        return reward;
    }

    shared_ptr<Int3TupleStateDistr> DeepSeaEnv::get_observation_distribution(
        shared_ptr<const IntAction> action, shared_ptr<const Int3TupleState> next_state) const
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<ObservationDistr> distr_itfc = ThtsEnv::get_observation_distribution_itfc(act_itfc, next_state_itfc);
        shared_ptr<Int3TupleStateDistr> distr = make_shared<Int3TupleStateDistr>();
        for (pair<const shared_ptr<const Observation>, double> pr : *distr_itfc) {
            shared_ptr<const Int3TupleState> obsv = static_pointer_cast<const Int3TupleState>(pr.first);
            distr->insert_or_assign(obsv, pr.second);
        }
        return distr;
    }

    shared_ptr<const Int3TupleState> DeepSeaEnv::sample_observation_distribution(
        shared_ptr<const IntAction> action, shared_ptr<const Int3TupleState> next_state, RandManager& rand_manager) const
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<const Observation> obsv_itfc = ThtsEnv::sample_observation_distribution_itfc(
            act_itfc, next_state_itfc, rand_manager);
        return static_pointer_cast<const Int3TupleState>(obsv_itfc);
    }

    shared_ptr<ThtsEnvContext> DeepSeaEnv::sample_context(shared_ptr<const Int3TupleState> state) const {
        shared_ptr<const State> state_itfc = static_pointer_cast<const State>(state);
        shared_ptr<ThtsEnvContext> context = ThtsEnv::sample_context_itfc(state_itfc);
        return static_pointer_cast<ThtsEnvContext>(context);
    }

    shared_ptr<const State> DeepSeaEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const State>(get_initial_state());
    }

    bool DeepSeaEnv::is_sink_state_itfc(shared_ptr<const State> state) const {
        return is_sink_state(static_pointer_cast<const Int3TupleState>(state));
    }

    shared_ptr<ActionVector> DeepSeaEnv::get_valid_actions_itfc(shared_ptr<const State> state) const {
        shared_ptr<IntActionVector> valid = get_valid_actions(static_pointer_cast<const Int3TupleState>(state));
        shared_ptr<ActionVector> out = make_shared<ActionVector>();
        for (shared_ptr<const IntAction> a : *valid) out->push_back(static_pointer_cast<const Action>(a));
        return out;
    }

    shared_ptr<StateDistr> DeepSeaEnv::get_transition_distribution_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action) const
    {
        shared_ptr<Int3TupleStateDistr> distr_itfc = get_transition_distribution(
            static_pointer_cast<const Int3TupleState>(state),
            static_pointer_cast<const IntAction>(action));
        shared_ptr<StateDistr> distr = make_shared<StateDistr>();
        for (auto& kv : *distr_itfc) {
            distr->insert_or_assign(static_pointer_cast<const State>(kv.first), kv.second);
        }
        return distr;
    }

    shared_ptr<const State> DeepSeaEnv::sample_transition_distribution_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action, RandManager& rand_manager) const
    {
        shared_ptr<const Int3TupleState> obsv = sample_transition_distribution(
            static_pointer_cast<const Int3TupleState>(state),
            static_pointer_cast<const IntAction>(action),
            rand_manager);
        return static_pointer_cast<const State>(obsv);
    }

    shared_ptr<ObservationDistr> DeepSeaEnv::get_observation_distribution_itfc(
        shared_ptr<const Action> action, shared_ptr<const State> next_state) const
    {
        shared_ptr<Int3TupleStateDistr> distr_itfc = get_observation_distribution(
            static_pointer_cast<const IntAction>(action),
            static_pointer_cast<const Int3TupleState>(next_state));
        shared_ptr<ObservationDistr> distr = make_shared<ObservationDistr>();
        for (auto& kv : *distr_itfc) {
            distr->insert_or_assign(static_pointer_cast<const Observation>(kv.first), kv.second);
        }
        return distr;
    }

    shared_ptr<const Observation> DeepSeaEnv::sample_observation_distribution_itfc(
        shared_ptr<const Action> action, shared_ptr<const State> next_state, RandManager& rand_manager) const
    {
        shared_ptr<const Int3TupleState> obsv = sample_observation_distribution(
            static_pointer_cast<const IntAction>(action),
            static_pointer_cast<const Int3TupleState>(next_state),
            rand_manager);
        return static_pointer_cast<const Observation>(obsv);
    }

    double DeepSeaEnv::get_reward_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action, shared_ptr<const Observation> observation) const
    {
        return get_reward(
            static_pointer_cast<const Int3TupleState>(state),
            static_pointer_cast<const IntAction>(action),
            static_pointer_cast<const Int3TupleState>(observation));
    }

    shared_ptr<ThtsEnvContext> DeepSeaEnv::sample_context_itfc(shared_ptr<const State> state) const {
        shared_ptr<ThtsEnvContext> context = sample_context(static_pointer_cast<const Int3TupleState>(state));
        return static_pointer_cast<ThtsEnvContext>(context);
    }
}

#include "toy_envs/nchain_env.h"

#include "helper_templates.h"

using namespace std;

namespace thts {
    NChainEnv::NChainEnv(int N,
                         double left_reward,
                         double goal_reward,
                         double right_back_prob,
                         double right_stay_prob,
                         double right_advance_prob) :
        ThtsEnv(true),
        N(N),
        left_reward(left_reward),
        goal_reward(goal_reward),
        right_back_prob(right_back_prob),
        right_stay_prob(right_stay_prob),
        right_advance_prob(right_advance_prob),
        cached_actions(make_shared<IntActionVector>())
    {
        cached_actions->push_back(make_shared<const IntAction>(0)); // LEFT
        cached_actions->push_back(make_shared<const IntAction>(1)); // RIGHT
    }

    shared_ptr<const Int3TupleState> NChainEnv::get_initial_state() const {
        return make_shared<const Int3TupleState>(0, 0, 0);
    }

    bool NChainEnv::is_sink_state(shared_ptr<const Int3TupleState> state) const {
        return get<0>(state->state) == N - 1;
    }

    shared_ptr<IntActionVector> NChainEnv::get_valid_actions(shared_ptr<const Int3TupleState> state) const {
        if (is_sink_state(state)) return make_shared<IntActionVector>();
        return cached_actions;
    }

    shared_ptr<Int3TupleStateDistr> NChainEnv::get_transition_distribution(
        shared_ptr<const Int3TupleState> state, shared_ptr<const IntAction> action) const
    {
        int pos = get<0>(state->state);
        shared_ptr<Int3TupleStateDistr> distr = make_shared<Int3TupleStateDistr>();

        if (action->action == 0) {
            int next_pos = (pos == 0) ? 0 : pos - 1;
            distr->insert_or_assign(make_shared<const Int3TupleState>(next_pos, 0, 0), 1.0);
            return distr;
        }

        // RIGHT
        if (pos == 0) {
            double stay_mass = right_back_prob + right_stay_prob;
            distr->insert_or_assign(make_shared<const Int3TupleState>(0, 0, 0), stay_mass);
            distr->insert_or_assign(make_shared<const Int3TupleState>(1, 0, 0), right_advance_prob);
            return distr;
        }
        distr->insert_or_assign(make_shared<const Int3TupleState>(pos - 1, 0, 0), right_back_prob);
        distr->insert_or_assign(make_shared<const Int3TupleState>(pos, 0, 0), right_stay_prob);
        distr->insert_or_assign(make_shared<const Int3TupleState>(pos + 1, 0, 0), right_advance_prob);
        return distr;
    }

    shared_ptr<const Int3TupleState> NChainEnv::sample_transition_distribution(
        shared_ptr<const Int3TupleState> state, shared_ptr<const IntAction> action, RandManager& rand_manager) const
    {
        shared_ptr<Int3TupleStateDistr> distr = get_transition_distribution(state, action);
        return helper::sample_from_distribution(*distr, rand_manager);
    }

    double NChainEnv::get_reward(
        shared_ptr<const Int3TupleState> state,
        shared_ptr<const IntAction> action,
        shared_ptr<const Int3TupleState> observation) const
    {
        int pos = get<0>(state->state);
        if (action->action == 0 && pos == 0) {
            return left_reward;
        }
        if (observation != nullptr) {
            int next_pos = get<0>(observation->state);
            if (next_pos == N - 1 && pos != N - 1) {
                return goal_reward;
            }
        }
        return 0.0;
    }

    shared_ptr<Int3TupleStateDistr> NChainEnv::get_observation_distribution(
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

    shared_ptr<const Int3TupleState> NChainEnv::sample_observation_distribution(
        shared_ptr<const IntAction> action, shared_ptr<const Int3TupleState> next_state, RandManager& rand_manager) const
    {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<const Observation> obsv_itfc = ThtsEnv::sample_observation_distribution_itfc(
            act_itfc, next_state_itfc, rand_manager);
        return static_pointer_cast<const Int3TupleState>(obsv_itfc);
    }

    shared_ptr<ThtsEnvContext> NChainEnv::sample_context(shared_ptr<const Int3TupleState> state) const {
        shared_ptr<const State> state_itfc = static_pointer_cast<const State>(state);
        shared_ptr<ThtsEnvContext> context = ThtsEnv::sample_context_itfc(state_itfc);
        return static_pointer_cast<ThtsEnvContext>(context);
    }

    shared_ptr<const State> NChainEnv::get_initial_state_itfc() const {
        return static_pointer_cast<const State>(get_initial_state());
    }

    bool NChainEnv::is_sink_state_itfc(shared_ptr<const State> state) const {
        return is_sink_state(static_pointer_cast<const Int3TupleState>(state));
    }

    shared_ptr<ActionVector> NChainEnv::get_valid_actions_itfc(shared_ptr<const State> state) const {
        shared_ptr<IntActionVector> valid = get_valid_actions(static_pointer_cast<const Int3TupleState>(state));
        shared_ptr<ActionVector> out = make_shared<ActionVector>();
        for (shared_ptr<const IntAction> a : *valid) out->push_back(static_pointer_cast<const Action>(a));
        return out;
    }

    shared_ptr<StateDistr> NChainEnv::get_transition_distribution_itfc(
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

    shared_ptr<const State> NChainEnv::sample_transition_distribution_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action, RandManager& rand_manager) const
    {
        shared_ptr<const Int3TupleState> obsv = sample_transition_distribution(
            static_pointer_cast<const Int3TupleState>(state),
            static_pointer_cast<const IntAction>(action),
            rand_manager);
        return static_pointer_cast<const State>(obsv);
    }

    shared_ptr<ObservationDistr> NChainEnv::get_observation_distribution_itfc(
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

    shared_ptr<const Observation> NChainEnv::sample_observation_distribution_itfc(
        shared_ptr<const Action> action, shared_ptr<const State> next_state, RandManager& rand_manager) const
    {
        shared_ptr<const Int3TupleState> obsv = sample_observation_distribution(
            static_pointer_cast<const IntAction>(action),
            static_pointer_cast<const Int3TupleState>(next_state),
            rand_manager);
        return static_pointer_cast<const Observation>(obsv);
    }

    double NChainEnv::get_reward_itfc(
        shared_ptr<const State> state, shared_ptr<const Action> action, shared_ptr<const Observation> observation) const
    {
        return get_reward(
            static_pointer_cast<const Int3TupleState>(state),
            static_pointer_cast<const IntAction>(action),
            static_pointer_cast<const Int3TupleState>(observation));
    }

    shared_ptr<ThtsEnvContext> NChainEnv::sample_context_itfc(shared_ptr<const State> state) const {
        shared_ptr<ThtsEnvContext> context = sample_context(static_pointer_cast<const Int3TupleState>(state));
        return static_pointer_cast<ThtsEnvContext>(context);
    }
}

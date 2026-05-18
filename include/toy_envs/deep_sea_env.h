#pragma once

#include "thts_env.h"
#include "thts_manager.h"
#include "thts_types.h"

#include <memory>
#include <vector>

namespace thts {
    /**
     * Deep Sea environment (bsuite-style hard exploration benchmark).
     *
     * N x N triangular grid. Agent starts at (0,0) and moves down one row per step. At each
     * column the action index that corresponds to "move right" is randomised once at env
     * construction (action_mapping[col] in {0,1}). Right moves carry a small cost 0.01/N;
     * reaching the goal (N-1, N-1) yields +1. Only the all-right trajectory reaches the goal;
     * a uniform random policy succeeds with probability 2^{-(N-1)}.
     *
     * State: (row, col, 0). Sink: row >= N-1.
     */
    class DeepSeaEnv : public ThtsEnv {
        protected:
            int N;
            double right_cost;
            std::vector<int> action_mapping;
            std::shared_ptr<IntActionVector> cached_actions;

            bool is_right_action(int col, int action) const;

        public:
            DeepSeaEnv(int N, unsigned int action_mapping_seed = 42);
            virtual ~DeepSeaEnv() = default;

            std::shared_ptr<const Int3TupleState> get_initial_state() const;
            bool is_sink_state(std::shared_ptr<const Int3TupleState> state) const;
            std::shared_ptr<IntActionVector> get_valid_actions(std::shared_ptr<const Int3TupleState> state) const;
            std::shared_ptr<Int3TupleStateDistr> get_transition_distribution(
                std::shared_ptr<const Int3TupleState> state, std::shared_ptr<const IntAction> action) const;
            std::shared_ptr<const Int3TupleState> sample_transition_distribution(
                std::shared_ptr<const Int3TupleState> state,
                std::shared_ptr<const IntAction> action,
                RandManager& rand_manager) const;
            double get_reward(
                std::shared_ptr<const Int3TupleState> state,
                std::shared_ptr<const IntAction> action,
                std::shared_ptr<const Int3TupleState> observation = nullptr) const;

            virtual std::shared_ptr<Int3TupleStateDistr> get_observation_distribution(
                std::shared_ptr<const IntAction> action, std::shared_ptr<const Int3TupleState> next_state) const;
            virtual std::shared_ptr<const Int3TupleState> sample_observation_distribution(
                std::shared_ptr<const IntAction> action,
                std::shared_ptr<const Int3TupleState> next_state,
                RandManager& rand_manager) const;
            virtual std::shared_ptr<ThtsEnvContext> sample_context(std::shared_ptr<const Int3TupleState> state) const;

            virtual std::shared_ptr<const State> get_initial_state_itfc() const;
            virtual bool is_sink_state_itfc(std::shared_ptr<const State> state) const;
            virtual std::shared_ptr<ActionVector> get_valid_actions_itfc(std::shared_ptr<const State> state) const;
            virtual std::shared_ptr<StateDistr> get_transition_distribution_itfc(
                std::shared_ptr<const State> state, std::shared_ptr<const Action> action) const;
            virtual std::shared_ptr<const State> sample_transition_distribution_itfc(
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                RandManager& rand_manager) const;
            virtual std::shared_ptr<ObservationDistr> get_observation_distribution_itfc(
                std::shared_ptr<const Action> action, std::shared_ptr<const State> next_state) const;
            virtual std::shared_ptr<const Observation> sample_observation_distribution_itfc(
                std::shared_ptr<const Action> action,
                std::shared_ptr<const State> next_state,
                RandManager& rand_manager) const;
            virtual double get_reward_itfc(
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                std::shared_ptr<const Observation> observation = nullptr) const;
            virtual std::shared_ptr<ThtsEnvContext> sample_context_itfc(std::shared_ptr<const State> state) const;
    };
}

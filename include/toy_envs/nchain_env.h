#pragma once

#include "thts_env.h"
#include "thts_manager.h"
#include "thts_types.h"

#include <memory>

namespace thts {
    /**
     * RiverSwim / NChain environment (Strehl & Littman style hard-exploration benchmark).
     *
     * N states in a chain (positions 0..N-1). Two actions:
     *   - LEFT  (action 0): "swim with the current" — deterministic and easy.
     *   - RIGHT (action 1): "swim against the current" — stochastic and hard.
     *
     * Transitions:
     *   LEFT  from pos=0:           stay at 0.
     *   LEFT  from pos>0:           deterministic move to pos-1.
     *   RIGHT from pos=0:           stay with prob (right_back_prob + right_stay_prob),
     *                               advance with prob right_advance_prob.
     *   RIGHT from pos in [1,N-2]:  back with right_back_prob, stay with right_stay_prob,
     *                               advance with right_advance_prob.
     *
     * Rewards (observation-dependent for the goal):
     *   LEFT at pos=0: +left_reward (small "exploit" trap, default 0.05).
     *   Entering pos=N-1 (sink) for the first time: +goal_reward (default 1.0).
     *
     * Classical greedy MCTS commits to LEFT for immediate small reward and rarely discovers
     * the +1 at the far end. ER's path-level signal keeps the rare RIGHT chain attractive
     * long enough for the advance probabilities to compound.
     *
     * State: (pos, 0, 0). Sink: pos == N-1.
     */
    class NChainEnv : public ThtsEnv {
        protected:
            int N;
            double left_reward;
            double goal_reward;
            double right_back_prob;
            double right_stay_prob;
            double right_advance_prob;
            std::shared_ptr<IntActionVector> cached_actions;

        public:
            NChainEnv(int N = 6,
                      double left_reward = 0.05,
                      double goal_reward = 1.0,
                      double right_back_prob = 0.05,
                      double right_stay_prob = 0.60,
                      double right_advance_prob = 0.35);
            virtual ~NChainEnv() = default;

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

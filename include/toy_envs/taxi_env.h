#pragma once

#include "thts_env.h"
#include "thts_manager.h"
#include "thts_types.h"

#include <memory>

enum TaxiAction {
    TAXI_SOUTH,
    TAXI_NORTH,
    TAXI_EAST,
    TAXI_WEST,
    TAXI_PICKUP,
    TAXI_DROPOFF,
};

namespace thts {
    /**
     * Taxi environment (5x5) similar to the classical benchmark.
     *
     * State representation: (row, col, encoded)
     * where encoded = passenger_status * 4 + destination_id.
     * passenger_status in [0,1,2,3,4], where 4 means passenger is in taxi.
     * destination_id in [0,1,2,3] for landmarks R,G,Y,B respectively.
     */
    class TaxiEnv : public ThtsEnv {
        protected:
            int width;
            int height;
            std::shared_ptr<IntActionVector> cached_actions;

            bool can_move_east(int row, int col) const;
            bool can_move_west(int row, int col) const;
            std::pair<int,int> get_landmark(int landmark_id) const;
            int encode_passenger_and_destination(int passenger_status, int destination_id) const;
            int get_passenger_status(std::shared_ptr<const Int3TupleState> state) const;
            int get_destination_id(std::shared_ptr<const Int3TupleState> state) const;

        public:
            TaxiEnv(int width=5, int height=5);
            virtual ~TaxiEnv() = default;

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
                std::shared_ptr<const Int3TupleState> observation=nullptr) const;

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
                std::shared_ptr<const Observation> observation=nullptr) const;
            virtual std::shared_ptr<ThtsEnvContext> sample_context_itfc(std::shared_ptr<const State> state) const;
    };
}

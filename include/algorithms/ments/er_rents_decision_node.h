#pragma once

#include "algorithms/ments/rents/rents_chance_node.h"
#include "algorithms/ments/rents/rents_decision_node.h"
#include "algorithms/ments/ments_manager.h"
#include "thts_types.h"

#include "thts_env.h"
#include "thts_env_context.h"
#include "thts_manager.h"

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thts {
    class ERRentsCNode;

    class ERRentsDNode : public RentsDNode {
        friend ERRentsCNode;

        protected:
            virtual void compute_action_weights(
                ActionDistr& action_weights,
                double& sum_action_weights,
                double& normalisation_term,
                ThtsEnvContext& context) const override;

            std::shared_ptr<ERRentsCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;

        public:
            ERRentsDNode(
                std::shared_ptr<MentsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const ERRentsCNode> parent=nullptr);

            virtual ~ERRentsDNode() = default;

            virtual void backup(
                const std::vector<double>& trial_rewards_before_node, 
                const std::vector<double>& trial_rewards_after_node, 
                const double trial_cumulative_return_after_node, 
                const double trial_cumulative_return,
                ThtsEnvContext& ctx) override;

            virtual std::shared_ptr<ThtsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const override;
    };

    class ERRentsCNode : public RentsCNode {
        friend ERRentsDNode;

        protected:
            std::shared_ptr<ERRentsDNode> create_child_node_helper(
                std::shared_ptr<const State> observation,
                std::shared_ptr<const State> next_state=nullptr) const;

        public:
            ERRentsCNode(
                std::shared_ptr<MentsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const ERRentsDNode> parent=nullptr);

            virtual ~ERRentsCNode() = default;

            virtual std::shared_ptr<ThtsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation,
                std::shared_ptr<const State> next_state=nullptr) const override;
    };
}
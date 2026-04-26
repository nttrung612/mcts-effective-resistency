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

            /**
             * Builds the *unaugmented* RENTS action distribution at this node (parent_distr * exp(Q/τ),
             * normalised, then mixed with the epsilon-uniform / prior-policy components from
             * MentsDNode::compute_action_distribution). Used as π_{t-1} stored in ThtsEnvContext for
             * children to read during their selection and (more importantly) backup phases -- per the
             * MCTS-ER paper the backup must use unaugmented values, so the prior we pass to children
             * must not carry the ER perturbation.
             */
            void compute_unaugmented_rents_distribution(ActionDistr& action_distr, ThtsEnvContext& ctx) const;

            std::shared_ptr<ERRentsCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;

        public:
            ERRentsDNode(
                std::shared_ptr<MentsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const ERRentsCNode> parent=nullptr);

            virtual ~ERRentsDNode() = default;

            /**
             * Selects an action using the ER-augmented RENTS policy at this node, but stores the
             * *unaugmented* RENTS distribution in ThtsEnvContext so that children read the correct
             * π_{t-1} during backup. The alias-table and max-heap selection paths are not currently
             * supported under ER-RENTS and will throw.
             */
            virtual std::shared_ptr<const Action> select_action(ThtsEnvContext& ctx) override;

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
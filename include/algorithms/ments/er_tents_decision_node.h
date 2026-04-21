#pragma once

#include "algorithms/ments/tents/tents_chance_node.h"
#include "algorithms/ments/tents/tents_decision_node.h"
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
    class ERTentsCNode;

    class ERTentsDNode : public TentsDNode {
        friend ERTentsCNode;

        protected:
            virtual void compute_action_weights(
                ActionDistr& action_weights,
                double& sum_action_weights,
                double& normalisation_term,
                ThtsEnvContext& context) const override;

            std::shared_ptr<ERTentsCNode> create_child_node_helper(std::shared_ptr<const Action> action) const;

        public:
            ERTentsDNode(
                std::shared_ptr<MentsManager> thts_manager,
                std::shared_ptr<const State> state,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const ERTentsCNode> parent=nullptr);

            virtual ~ERTentsDNode() = default;

            virtual std::shared_ptr<ThtsCNode> create_child_node_helper_itfc(
                std::shared_ptr<const Action> action) const override;
    };

    class ERTentsCNode : public TentsCNode {
        friend ERTentsDNode;

        protected:
            std::shared_ptr<ERTentsDNode> create_child_node_helper(
                std::shared_ptr<const State> observation,
                std::shared_ptr<const State> next_state=nullptr) const;

        public:
            ERTentsCNode(
                std::shared_ptr<MentsManager> thts_manager,
                std::shared_ptr<const State> state,
                std::shared_ptr<const Action> action,
                int decision_depth,
                int decision_timestep,
                std::shared_ptr<const ERTentsDNode> parent=nullptr);

            virtual ~ERTentsCNode() = default;

            virtual std::shared_ptr<ThtsDNode> create_child_node_helper_itfc(
                std::shared_ptr<const Observation> observation,
                std::shared_ptr<const State> next_state=nullptr) const override;
    };
}
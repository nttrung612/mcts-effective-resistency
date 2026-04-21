#include "algorithms/ments/er_ments_decision_node.h"

#include "helper_templates.h"

#include <cmath>
#include <limits>

using namespace std;

namespace thts {
    ERMentsDNode::ERMentsDNode(
        shared_ptr<MentsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ERMentsCNode> parent) :
            MentsDNode(
                thts_manager,
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MentsCNode>(parent))
    {
    }

    void ERMentsDNode::compute_action_weights(
        ActionDistr& action_weights,
        double& sum_action_weights,
        double& normalisation_term,
        ThtsEnvContext& context) const
    {
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        double temp = get_temp();

        normalisation_term = numeric_limits<double>::lowest();
        for (shared_ptr<const Action> action : *actions) {
            double q_value_over_temp = get_er_augmented_soft_q_value(action, opp_coeff) / temp;
            if (normalisation_term < q_value_over_temp) {
                normalisation_term = q_value_over_temp;
            }
        }

        sum_action_weights = 0.0;
        for (shared_ptr<const Action> action : *actions) {
            double soft_q_value = get_er_augmented_soft_q_value(action, opp_coeff);
            double action_weight = exp((soft_q_value / temp) - normalisation_term);
            if (isnan(action_weight)) {
                action_weight = 0.0;
            }
            action_weights[action] = action_weight;
            sum_action_weights += action_weight;
        }
    }

    shared_ptr<ERMentsCNode> ERMentsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<ERMentsCNode>(
            static_pointer_cast<MentsManager>(thts_manager),
            state,
            action,
            decision_depth,
            decision_timestep,
            static_pointer_cast<const ERMentsDNode>(shared_from_this()));
    }

    shared_ptr<ThtsCNode> ERMentsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ERMentsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }

    ERMentsCNode::ERMentsCNode(
        shared_ptr<MentsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ERMentsDNode> parent) :
            MentsCNode(
                thts_manager,
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const MentsDNode>(parent))
    {
    }

    shared_ptr<ERMentsDNode> ERMentsCNode::create_child_node_helper(
        shared_ptr<const State> observation,
        shared_ptr<const State> next_state) const
    {
        shared_ptr<const State> mdp_next_state = static_pointer_cast<const State>(observation);
        return make_shared<ERMentsDNode>(
            static_pointer_cast<MentsManager>(thts_manager),
            mdp_next_state,
            decision_depth + 1,
            decision_timestep + 1,
            static_pointer_cast<const ERMentsCNode>(shared_from_this()));
    }

    shared_ptr<ThtsDNode> ERMentsCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation,
        shared_ptr<const State> next_state) const
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<ERMentsDNode> child_node = create_child_node_helper(obsv_itfc, next_state_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
}
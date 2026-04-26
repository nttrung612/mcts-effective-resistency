#include "algorithms/ments/er_rents_decision_node.h"

#include "helper_templates.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace std;

namespace thts {
    static constexpr double ER_RENTS_NEAR_ZERO_PROB = 1e-16;

    ERRentsDNode::ERRentsDNode(
        shared_ptr<MentsManager> thts_manager,
        shared_ptr<const State> state,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ERRentsCNode> parent) :
            RentsDNode(
                thts_manager,
                state,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const RentsCNode>(parent))
    {
        ThtsEnvContext spoof_ctx;
        cached_action_distr = select_action_alias_tables_get_mixed_distr(spoof_ctx)->get_distr_map();
    }

    void ERRentsDNode::compute_action_weights(
        ActionDistr& action_weights,
        double& sum_action_weights,
        double& normalisation_term,
        ThtsEnvContext& context) const
    {
        double opp_coeff = is_opponent() ? -1.0 : 1.0;
        double temp = get_temp();

        normalisation_term = numeric_limits<double>::lowest();
        for (shared_ptr<const Action> action : *actions) {
            double q_value_over_temp = (get_soft_q_value(action, opp_coeff) + get_er_bonus(action)) / temp;
            if (normalisation_term < q_value_over_temp) {
                normalisation_term = q_value_over_temp;
            }
        }

        shared_ptr<ActionDistr> parent_distr = get_parent_distr_from_context(context);

        sum_action_weights = 0.0;
        for (shared_ptr<const Action> action : *actions) {
            double soft_q_value = get_soft_q_value(action, opp_coeff) + get_er_bonus(action);
            double action_weight = exp((soft_q_value / temp) - normalisation_term);
            action_weight *= get_parent_action_prob(parent_distr, action);
            action_weights[action] = action_weight;
            sum_action_weights += action_weight;
        }

        if (sum_action_weights < ER_RENTS_NEAR_ZERO_PROB) {
            double uniform_weight = 1.0 / actions->size();
            for (shared_ptr<const Action> action : *actions) {
                action_weights[action] = uniform_weight;
            }
            sum_action_weights = 1.0;
        }
    }

    /**
     * Mirrors MentsDNode::compute_action_distribution but explicitly invokes
     * RentsDNode::compute_action_weights (qualified, non-virtual) so that ER's compute_action_weights
     * override is bypassed. The result is the unaugmented RENTS policy at this node, used as π_{t-1}
     * by the children -- crucially during their backup phase, where the paper requires the ER bonus
     * not to leak into Q_Ω.
     */
    void ERRentsDNode::compute_unaugmented_rents_distribution(
        ActionDistr& action_distr, ThtsEnvContext& ctx) const
    {
        double sum_weights = 0.0;
        double _normalisation_term = 0.0;
        lock_all_children();
        RentsDNode::compute_action_weights(action_distr, sum_weights, _normalisation_term, ctx);
        unlock_all_children();

        MentsManager& manager = (MentsManager&) *thts_manager;
        double epsilon = manager.epsilon;
        if (is_root_node() && manager.root_node_epsilon > 0.0) epsilon = manager.root_node_epsilon;
        double lambda = epsilon / log(num_visits + 1);
        if (lambda > manager.max_explore_prob) {
            lambda = manager.max_explore_prob;
        }

        double num_actions = static_cast<double>(actions->size());
        double uniform_distr_mass = 1.0 / num_actions;
        double lambda_tilde = 0.0;
        bool use_prior_mix = manager.prior_policy_search_weight > 0.0;
        if (use_prior_mix) {
            lambda_tilde = manager.prior_policy_search_weight / log(num_visits + 3);
            if (lambda_tilde > 1.0) lambda_tilde = 1.0;
        }

        vector<shared_ptr<const Action>> near_zero_prob_actions;
        for (shared_ptr<const Action> action : *actions) {
            double prob = action_distr[action] * (1.0 - lambda) / sum_weights;
            if (use_prior_mix) {
                prob = prob * (1.0 - lambda_tilde)
                     + (1.0 - lambda) * lambda_tilde * policy_prior->at(action);
            }
            prob += lambda * uniform_distr_mass;
            action_distr[action] = prob;
            if (isnan(prob) || prob < ER_RENTS_NEAR_ZERO_PROB) {
                near_zero_prob_actions.push_back(action);
            }
        }
        for (shared_ptr<const Action> action : near_zero_prob_actions) {
            action_distr.erase(action);
        }
    }

    /**
     * ER-RENTS selection.
     *
     * 1. Compute the ER-augmented RENTS distribution via compute_action_distribution (which
     *    dispatches virtually to ERRentsDNode::compute_action_weights). This is what we *sample*
     *    from at this node -- it injects the c2/N(s,a) bonus into the logits.
     * 2. Compute the *unaugmented* RENTS distribution and store it in ThtsEnvContext under this
     *    node's depth key, overwriting the augmented copy that would otherwise have been left there
     *    by RentsDNode::select_action_rents. Children read this entry as π_{t-1} during their
     *    backup, and the MCTS-ER paper requires that the backup operate on unaugmented values.
     * 3. Sample from the augmented distribution.
     *
     * The alias-table and max-heap paths are not exercised in the toy-env runs and would each
     * require an additional cached-unaugmented-distribution to fix correctly; throw a clear error
     * if either is enabled.
     */
    shared_ptr<const Action> ERRentsDNode::select_action(ThtsEnvContext& ctx) {
        MentsManager& manager = (MentsManager&) *thts_manager;
        if (manager.alias_use_caching) {
            throw runtime_error(
                "ER-RENTS does not support alias_use_caching: the unaugmented prior fix is only "
                "implemented for the non-alias path.");
        }
        if (manager.use_max_heap) {
            throw runtime_error(
                "ER-RENTS does not support use_max_heap: backup_soft_with_max_heap reads the "
                "context prior, which would also need an unaugmented variant.");
        }

        // Distribution used to sample at this node (ER-augmented).
        ActionDistr er_action_distr;
        compute_action_distribution(er_action_distr, ctx);

        // Distribution stored as π_{t-1} for children's selection / backup (unaugmented).
        shared_ptr<ActionDistr> raw_action_distr = make_shared<ActionDistr>();
        compute_unaugmented_rents_distribution(*raw_action_distr, ctx);
        put_node_distr_in_context(raw_action_distr, ctx);

        shared_ptr<const Action> selected_action;
        while (is_nullptr_or_should_skip_under_construction_child(selected_action)) {
            selected_action = helper::sample_from_distribution(er_action_distr, *thts_manager, false);
            if (!has_child_node(selected_action)) {
                create_child_node(selected_action);
            }
        }
        return selected_action;
    }

    void ERRentsDNode::backup(
        const vector<double>& trial_rewards_before_node, 
        const vector<double>& trial_rewards_after_node, 
        const double trial_cumulative_return_after_node, 
        const double trial_cumulative_return,
        ThtsEnvContext& ctx) 
    {
        MentsManager& manager = (MentsManager&) *thts_manager;
        if (!manager.use_avg_return) {
            if (manager.use_max_heap) {
                RentsDNode::backup_soft_with_max_heap(ctx);
            } else {
                num_backups++;
                ActionDistr action_weights;
                double sum_weights;
                double normalisation_term;
                lock_all_children();
                // Explicitly call RentsDNode::compute_action_weights to avoid ER bonus leakage
                RentsDNode::compute_action_weights(action_weights, sum_weights, normalisation_term, ctx);
                unlock_all_children();

                double opp_coeff = is_opponent() ? -1.0 : 1.0;
                double temp = get_temp();
                soft_value = opp_coeff * temp * (log(sum_weights) + normalisation_term);
            }
            if (manager.alias_use_caching) {
                backup_update_alias_tables(ctx);
            }
            return;
        }

        RentsDNode::backup(trial_rewards_before_node, trial_rewards_after_node, trial_cumulative_return_after_node, trial_cumulative_return, ctx);
    }

    shared_ptr<ERRentsCNode> ERRentsDNode::create_child_node_helper(shared_ptr<const Action> action) const {
        return make_shared<ERRentsCNode>(
            static_pointer_cast<MentsManager>(thts_manager),
            state,
            action,
            decision_depth,
            decision_timestep,
            static_pointer_cast<const ERRentsDNode>(shared_from_this()));
    }

    shared_ptr<ThtsCNode> ERRentsDNode::create_child_node_helper_itfc(shared_ptr<const Action> action) const {
        shared_ptr<const Action> act_itfc = static_pointer_cast<const Action>(action);
        shared_ptr<ERRentsCNode> child_node = create_child_node_helper(act_itfc);
        return static_pointer_cast<ThtsCNode>(child_node);
    }

    ERRentsCNode::ERRentsCNode(
        shared_ptr<MentsManager> thts_manager,
        shared_ptr<const State> state,
        shared_ptr<const Action> action,
        int decision_depth,
        int decision_timestep,
        shared_ptr<const ERRentsDNode> parent) :
            RentsCNode(
                thts_manager,
                state,
                action,
                decision_depth,
                decision_timestep,
                static_pointer_cast<const RentsDNode>(parent))
    {
    }

    shared_ptr<ERRentsDNode> ERRentsCNode::create_child_node_helper(
        shared_ptr<const State> observation,
        shared_ptr<const State> next_state) const
    {
        shared_ptr<const State> mdp_next_state = static_pointer_cast<const State>(observation);
        return make_shared<ERRentsDNode>(
            static_pointer_cast<MentsManager>(thts_manager),
            mdp_next_state,
            decision_depth + 1,
            decision_timestep + 1,
            static_pointer_cast<const ERRentsCNode>(shared_from_this()));
    }

    shared_ptr<ThtsDNode> ERRentsCNode::create_child_node_helper_itfc(
        shared_ptr<const Observation> observation,
        shared_ptr<const State> next_state) const
    {
        shared_ptr<const State> obsv_itfc = static_pointer_cast<const State>(observation);
        shared_ptr<const State> next_state_itfc = static_pointer_cast<const State>(next_state);
        shared_ptr<ERRentsDNode> child_node = create_child_node_helper(obsv_itfc, next_state_itfc);
        return static_pointer_cast<ThtsDNode>(child_node);
    }
}
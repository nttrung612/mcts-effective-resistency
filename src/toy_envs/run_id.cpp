#include "toy_envs/run_id.h"

#include "toy_envs/d_chain_env.h"
#include "toy_envs/frozen_lake_env.h"
#include "toy_envs/sailing_env.h"
#include "toy_envs/taxi_env.h"
#include "toy_envs/deep_sea_env.h"
#include "toy_envs/nchain_env.h"

#include "algorithms/uct/uct_manager.h"
#include "algorithms/uct/er_uct_decision_node.h"
#include "algorithms/uct/er_fixed_depth_uct_decision_node.h"
#include "algorithms/uct/puct_manager.h"
#include "algorithms/ments/ments_manager.h"
#include "algorithms/ments/er_ments_decision_node.h"
#include "algorithms/ments/er_rents_decision_node.h"
#include "algorithms/ments/er_tents_decision_node.h"
#include "algorithms/ments/dents/dents_manager.h"

#include "algorithms/uct/uct_logger.h"
#include "algorithms/ments/ments_logger.h"
#include "algorithms/ments/dbments_logger.h"

#include "thts_env.h"

#include "algorithms/uct/uct_decision_node.h"
#include "algorithms/uct/er_uct_decision_node.h"
#include "algorithms/uct/fixed_depth_uct_decision_node.h"
#include "algorithms/uct/er_fixed_depth_uct_decision_node.h"
#include "algorithms/uct/puct_decision_node.h"
#include "algorithms/ments/ments_decision_node.h"
#include "algorithms/ments/er_ments_decision_node.h"
#include "algorithms/ments/dbments_decision_node.h"
#include "algorithms/ments/dents/dents_decision_node.h"
#include "algorithms/ments/er_rents_decision_node.h"
#include "algorithms/ments/rents/rents_decision_node.h"
#include "algorithms/ments/er_tents_decision_node.h"
#include "algorithms/ments/tents/tents_decision_node.h"
#include "algorithms/est/est_decision_node.h"

#include <stdexcept>

using namespace std;

namespace thts {
    /**
     * Default constructor
    */
    RunID::RunID() {}

    /**
     * Initialised constructor
    */
    RunID::RunID(
        string env_id,
        string env_instance_id,
        string expr_id,
        string alg_id,
        unordered_map<string, double> alg_params,
        int num_trials,
        int max_trial_length,
        int trials_log_delta,
        int mc_eval_trials_delta,
        int rollouts_per_mc_eval,
        int num_repeats,
        int num_threads,
        int eval_threads) :
            env_id(env_id),
            env_instance_id(env_instance_id),
            expr_id(expr_id),
            alg_id(alg_id),
            alg_params(alg_params),
            num_trials(num_trials),
            max_trial_length(max_trial_length),
            trials_log_delta(trials_log_delta),
            mc_eval_trials_delta(mc_eval_trials_delta),
            rollouts_per_mc_eval(rollouts_per_mc_eval),
            num_repeats(num_repeats),
            num_threads(num_threads),
            eval_threads(eval_threads)
    {
    }

    /**
     * Create and return the env
    */
    shared_ptr<ThtsEnv> RunID::get_env() {
        if (env_id == DCHAIN_ENV_ID) {
            if (env_instance_id == D_10_ID) {
                return make_shared<DChainEnv>(10, 1.0);
            } else if (env_instance_id == D_10_FF_ID) {
                return make_shared<DChainEnv>(10, 0.8);
            } else if (env_instance_id == D_10_HALF_ID) {
                return make_shared<DChainEnv>(10, 0.5);
            } else if (env_instance_id == D_20_ID) {
                return make_shared<DChainEnv>(20, 1.0);
            } else if (env_instance_id == D_20_FF_ID) {
                return make_shared<DChainEnv>(20, 0.8);
            } else if (env_instance_id == D_20_HALF_ID) {
                return make_shared<DChainEnv>(20, 0.5);
            } else {
                throw runtime_error("Invalid DChain instance id.");
            }
        }

        if (env_id == FL_ENV_ID) {
            if (env_instance_id == FL_8x8) {
                return make_shared<FrozenLakeEnv>(8, 8, FL_RAND_8X8_MAP, 0.99);
            } else if (env_instance_id == FL_8x8_TEST) {
                return make_shared<FrozenLakeEnv>(8, 8, FL_RAND_8X8_TEST_MAP, 0.99);
            } else if (env_instance_id == FL_8x12) { 
                return make_shared<FrozenLakeEnv>(8, 16, FL_RAND_8X12_MAP, 0.99);
            } else if (env_instance_id == FL_8x12_TEST) {
                return make_shared<FrozenLakeEnv>(8, 16, FL_RAND_8X12_TEST_MAP, 0.99);
            } else if (env_instance_id == FL_8x16_TEST) {
                return make_shared<FrozenLakeEnv>(8, 16, FL_RAND_8X16_TEST_MAP, 0.99);
            } else {
                throw runtime_error("Not implemented yet");
            }
        }

        if (env_id == SAILING_ENV_ID) {
            if (env_instance_id == S_6_ID) {
                return make_shared<SailingEnv>(6, 6);
            } else if (env_instance_id == S_6_TEST_ID) {
                return make_shared<SailingEnv>(6, 6, SE);
            } else {
                throw runtime_error("Not implemented yet");
            }
        }

        if (env_id == TAXI_ENV_ID) {
            if (env_instance_id == TX_5_ID || env_instance_id == TX_5_TEST_ID) {
                return make_shared<TaxiEnv>(5, 5);
            } else {
                throw runtime_error("Not implemented yet");
            }
        }

        if (env_id == DEEP_SEA_ENV_ID) {
            if (env_instance_id == DS_10_ID) {
                return make_shared<DeepSeaEnv>(10);
            } else if (env_instance_id == DS_20_ID) {
                return make_shared<DeepSeaEnv>(20);
            } else if (env_instance_id == DS_30_ID) {
                return make_shared<DeepSeaEnv>(30);
            } else {
                throw runtime_error("Not implemented yet");
            }
        }

        if (env_id == NCHAIN_ENV_ID) {
            if (env_instance_id == NC_6_ID) {
                return make_shared<NChainEnv>(6);
            } else if (env_instance_id == NC_10_ID) {
                return make_shared<NChainEnv>(10);
            } else {
                throw runtime_error("Not implemented yet");
            }
        }

        throw runtime_error("Error in RunID get_env");
    }

    /**
     * Create thts manager
    */
    shared_ptr<ThtsManager> RunID::get_thts_manager(shared_ptr<ThtsEnv> env) {
        if (alg_id == ALG_ID_UCT || alg_id == ALG_ID_ER_UCT || alg_id == ALG_ID_FIXED_DEPTH_UCT || alg_id == ALG_ID_ER_FIXED_DEPTH_UCT) {
            UctManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.bias = alg_params.at(PARAMS_ID_UCT_BIAS);
            if (alg_params.find(PARAMS_ID_UCT_ER_C2) != alg_params.end()) {
                manager_args.er_c2 = alg_params.at(PARAMS_ID_UCT_ER_C2);
            }
            if (alg_params.find(PARAMS_ID_UCT_POWER_MEAN_P) != alg_params.end()) {
                manager_args.power_mean_p = alg_params.at(PARAMS_ID_UCT_POWER_MEAN_P);
            }
            return make_shared<UctManager>(manager_args);
        } 
        if (alg_id == ALG_ID_PUCT) {
            PuctManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.bias = alg_params.at(PARAMS_ID_UCT_BIAS);
            return make_shared<PuctManager>(manager_args);
        }
        if (alg_id == ALG_ID_MENTS || 
            alg_id == ALG_ID_ER_MENTS ||
            alg_id == ALG_ID_RENTS || 
            alg_id == ALG_ID_ER_RENTS ||
            alg_id == ALG_ID_TENTS ||
            alg_id == ALG_ID_ER_TENTS) 
        {
            MentsManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.temp = alg_params.at(PARAMS_ID_MENTS_TEMP);
            manager_args.epsilon = alg_params.at(PARAMS_ID_MENTS_EPSILON);
            if (alg_params.find(PARAMS_ID_MENTS_DEFAULT_Q_VALUE) != alg_params.end()) {
                manager_args.default_q_value = alg_params.at(PARAMS_ID_MENTS_DEFAULT_Q_VALUE);
            }
            if (alg_params.find(PARAMS_ID_UCT_ER_C2) != alg_params.end()) {
                manager_args.er_c2 = alg_params.at(PARAMS_ID_UCT_ER_C2);
            }
            if (alg_params.find(PARAMS_ID_MENTS_POWER_MEAN_P) != alg_params.end()) {
                manager_args.power_mean_p = alg_params.at(PARAMS_ID_MENTS_POWER_MEAN_P);
                manager_args.use_power_mean_backup = true;
            }
            if (alg_id == ALG_ID_ER_MENTS || alg_id == ALG_ID_ER_RENTS || alg_id == ALG_ID_ER_TENTS) {
                manager_args.use_power_mean_backup = true;
            }
            return make_shared<MentsManager>(manager_args);
        }
        if (alg_id == ALG_ID_DENTS ||
            alg_id == ALG_ID_EST) {
            DentsManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.temp = alg_params.at(PARAMS_ID_MENTS_TEMP);
            manager_args.value_temp_init = alg_params.at(PARAMS_ID_MENTS_TEMP);
            manager_args.epsilon = alg_params.at(PARAMS_ID_MENTS_EPSILON);
            if (alg_params.find(PARAMS_ID_MENTS_DEFAULT_Q_VALUE) != alg_params.end()) {
                manager_args.default_q_value = alg_params.at(PARAMS_ID_MENTS_DEFAULT_Q_VALUE);
            }
            if (alg_params.find(PARAMS_ID_DENTS_IGNORE_DECAY) != alg_params.end()) {
                manager_args.value_temp_decay_fn = decayed_temp_no_decay;
            }
            if (alg_params.find(PARAMS_ID_DENTS_TEMP) != alg_params.end()) {
                manager_args.value_temp_init = alg_params.at(PARAMS_ID_DENTS_TEMP);
            }
            return make_shared<DentsManager>(manager_args);
        }
        if (alg_id == ALG_ID_DBMENTS) {
            DentsManagerArgs manager_args(env);
            manager_args.max_depth = max_trial_length;
            manager_args.mcts_mode = false;
            manager_args.temp = alg_params.at(PARAMS_ID_MENTS_TEMP);
            manager_args.value_temp_init = alg_params.at(PARAMS_ID_MENTS_TEMP);
            manager_args.epsilon = alg_params.at(PARAMS_ID_MENTS_EPSILON);
            manager_args.value_temp_decay_fn = decayed_temp_no_decay;
            if (alg_params.find(PARAMS_ID_MENTS_DEFAULT_Q_VALUE) != alg_params.end()) {
                manager_args.default_q_value = alg_params.at(PARAMS_ID_MENTS_DEFAULT_Q_VALUE);
            }
            return make_shared<DentsManager>(manager_args);
        }

        throw runtime_error("Error in RunID get_thts_manager");
    }

    /**
     * Return a root search node
    */
    shared_ptr<ThtsDNode> RunID::get_root_search_node(shared_ptr<ThtsEnv> env, shared_ptr<ThtsManager> manager) {
        if (alg_id == ALG_ID_UCT) {
            shared_ptr<UctManager> uct_manager = static_pointer_cast<UctManager>(manager);
            return make_shared<UctDNode>(uct_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_ER_UCT) {
            shared_ptr<UctManager> uct_manager = static_pointer_cast<UctManager>(manager);
            return make_shared<ERUctDNode>(uct_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_FIXED_DEPTH_UCT) {
            shared_ptr<UctManager> uct_manager = static_pointer_cast<UctManager>(manager);
            return make_shared<FixedDepthUctDNode>(uct_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_ER_FIXED_DEPTH_UCT) {
            shared_ptr<UctManager> uct_manager = static_pointer_cast<UctManager>(manager);
            return make_shared<ERFixedDepthUctDNode>(uct_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_PUCT) {
            shared_ptr<PuctManager> puct_manager = static_pointer_cast<PuctManager>(manager);
            return make_shared<PuctDNode>(puct_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_MENTS) {
            shared_ptr<MentsManager> ments_manager = static_pointer_cast<MentsManager>(manager);
            return make_shared<MentsDNode>(ments_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_ER_MENTS) {
            shared_ptr<MentsManager> ments_manager = static_pointer_cast<MentsManager>(manager);
            return make_shared<ERMentsDNode>(ments_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_RENTS) {
            shared_ptr<MentsManager> ments_manager = static_pointer_cast<MentsManager>(manager);
            return make_shared<RentsDNode>(ments_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_ER_RENTS) {
            shared_ptr<MentsManager> ments_manager = static_pointer_cast<MentsManager>(manager);
            return make_shared<ERRentsDNode>(ments_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_TENTS) {
            shared_ptr<MentsManager> ments_manager = static_pointer_cast<MentsManager>(manager);
            return make_shared<TentsDNode>(ments_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_ER_TENTS) {
            shared_ptr<MentsManager> ments_manager = static_pointer_cast<MentsManager>(manager);
            return make_shared<ERTentsDNode>(ments_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_DENTS || alg_id == ALG_ID_DBMENTS) {
            shared_ptr<DentsManager> dents_manager = static_pointer_cast<DentsManager>(manager);
            return make_shared<DentsDNode>(dents_manager, env->get_initial_state_itfc(), 0, 0);
        }
        if (alg_id == ALG_ID_EST) {
            shared_ptr<DentsManager> ments_manager = static_pointer_cast<DentsManager>(manager);
            return make_shared<EstDNode>(ments_manager, env->get_initial_state_itfc(), 0, 0);
        }

        throw runtime_error("Error in RunID get_root_search_node");
    }

    /**
     * Returns a logger to use with this run
    */
    shared_ptr<ThtsLogger> RunID::get_logger() {
        if (alg_id == ALG_ID_UCT || alg_id == ALG_ID_ER_UCT || alg_id == ALG_ID_FIXED_DEPTH_UCT || alg_id == ALG_ID_ER_FIXED_DEPTH_UCT || alg_id == ALG_ID_PUCT) {
            shared_ptr<ThtsLogger> logger = make_shared<UctLogger>();
            logger->set_trials_delta(trials_log_delta);
            return logger;
        } 
        if (alg_id == ALG_ID_MENTS || alg_id == ALG_ID_ER_MENTS || alg_id == ALG_ID_DENTS || alg_id == ALG_ID_DBMENTS || alg_id == ALG_ID_RENTS || alg_id == ALG_ID_ER_RENTS || alg_id == ALG_ID_TENTS || alg_id == ALG_ID_ER_TENTS) {
            shared_ptr<ThtsLogger> logger = make_shared<MentsLogger>();
            logger->set_trials_delta(trials_log_delta);
            return logger;
        }
        if (alg_id == ALG_ID_EST) {
            shared_ptr<ThtsLogger> logger = make_shared<DBMentsLogger>();
            logger->set_trials_delta(trials_log_delta);
            return logger;
        }

        throw runtime_error("Error in RunID get_logger");
    }


    /**
     * Gets a list of RunID objects from a given expr id
    */
    shared_ptr<vector<RunID>> get_run_ids_from_expr_id(string expr_id) {
        shared_ptr<vector<RunID>> run_ids = make_shared<vector<RunID>>();

        // debug expr id for debugging
        if (expr_id == DEBUG_EXPR_ID) {
            string env_id = SAILING_ENV_ID;
            string env_instance_id = S_6_ID;
            int num_trials = 150000;
            int max_trial_length = 50;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 250;
            int num_repeats = 5;
            int num_threads = 16;
            int eval_threads = 32;

            double default_q_value = -20.0;

            vector<string> alg_ids = { ALG_ID_RENTS };
            vector<double> temps = { 0.01 };
            for (string alg_id : alg_ids) {
                for (double temp : temps) {
                    unordered_map<string,double> alg_params = {
                            {PARAMS_ID_MENTS_TEMP, temp}, 
                            {PARAMS_ID_MENTS_EPSILON, 0.1}, 
                            {PARAMS_ID_MENTS_DEFAULT_Q_VALUE, default_q_value},
                        };
                    run_ids->push_back(RunID(
                        env_id,
                        env_instance_id,
                        expr_id,
                        alg_id,
                        alg_params,
                        num_trials,
                        max_trial_length,
                        trials_log_delta,
                        mc_eval_trials_delta,
                        rollouts_per_mc_eval,
                        num_repeats,
                        num_threads,
                        eval_threads));
                }
            }

            return run_ids;
        }


        // expr_id: D001_LEN10
        // Runs algorithms with varying parameters on the 10-chain
        if (expr_id == D001_LEN10) {
            string env_id = DCHAIN_ENV_ID;
            vector<string> env_instance_ids = {D_10_ID, D_10_HALF_ID};
            int num_trials = 10000;
            int max_trial_length = 100; 
            int trials_log_delta = 10;
            int mc_eval_trials_delta = 10;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 32;
            int eval_threads = 32;

            for (string env_instance_id : env_instance_ids) {

                vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_ER_UCT, ALG_ID_FIXED_DEPTH_UCT, ALG_ID_ER_FIXED_DEPTH_UCT, ALG_ID_PUCT};
                vector<double> uct_biases = { UctManagerArgs::USE_AUTO_BIAS, 0.1, 1.0, 10.0, 100.0 };
                for (string alg_id : alg_ids) {
                    for (double bias : uct_biases) {
                        unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, bias}};
                        if (alg_id == ALG_ID_ER_UCT || alg_id == ALG_ID_ER_FIXED_DEPTH_UCT) {
                            alg_params[PARAMS_ID_UCT_ER_C2] = 1.0;
                        }
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }


                alg_ids = {ALG_ID_MENTS};
                vector<double> temps = { 1.0, 0.5, 0.2, 0.15, 0.10, 0.05, 0.01 };
                vector<double> epss = { 10.0, 1.0, 0.1, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        for (double eps : epss) {
                            unordered_map<string,double> alg_params = 
                                {
                                    {PARAMS_ID_MENTS_TEMP, temp}, 
                                    {PARAMS_ID_MENTS_EPSILON, eps}
                                };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }

                alg_ids = {ALG_ID_RENTS};
                temps = { 1000.0, 100.0, 10.0, 1.0, 0.1, 0.01 };
                epss = { 10.0, 1.0, 0.1, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        for (double eps : epss) {
                            unordered_map<string,double> alg_params = 
                                {
                                    {PARAMS_ID_MENTS_TEMP, temp}, 
                                    {PARAMS_ID_MENTS_EPSILON, eps}
                                };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }

                alg_ids = {ALG_ID_TENTS};
                temps = { 10.0, 1.0, 0.7, 0.5, 0.3, 0.1, 0.01 };
                epss = { 10.0, 1.0, 0.1, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        for (double eps : epss) {
                            unordered_map<string,double> alg_params = 
                                {
                                    {PARAMS_ID_MENTS_TEMP, temp}, 
                                    {PARAMS_ID_MENTS_EPSILON, eps}
                                };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }

                alg_ids = {ALG_ID_DENTS, ALG_ID_EST};
                temps = { 1.0, 0.5, 0.2, 0.15, 0.10, 0.05, 0.01 };
                epss = { 10.0, 1.0, 0.1, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        for (double eps : epss) {
                            unordered_map<string,double> alg_params = 
                                {
                                    {PARAMS_ID_MENTS_TEMP, temp}, 
                                    {PARAMS_ID_MENTS_EPSILON, eps}
                                };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }

            }

            return run_ids;
        }


        // expr_id: D003_LEN20
        // Runs algorithms with varying parameters on the 20-chain
        if (expr_id == D003_LEN20) {
            string env_id = DCHAIN_ENV_ID;
            vector<string> env_instance_ids = {D_20_ID, D_20_HALF_ID};
            int num_trials = 10000;
            int max_trial_length = 100; 
            int trials_log_delta = 10;
            int mc_eval_trials_delta = 10;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 32;
            int eval_threads = 32;

            for (string env_instance_id : env_instance_ids) {
                vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_ER_UCT, ALG_ID_FIXED_DEPTH_UCT, ALG_ID_ER_FIXED_DEPTH_UCT, ALG_ID_PUCT};
                vector<double> uct_biases = { UctManagerArgs::USE_AUTO_BIAS, 0.1, 1.0, 10.0, 100.0 };
                for (string alg_id : alg_ids) {
                    for (double bias : uct_biases) {
                        unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, bias}};
                        if (alg_id == ALG_ID_ER_UCT || alg_id == ALG_ID_ER_FIXED_DEPTH_UCT) {
                            alg_params[PARAMS_ID_UCT_ER_C2] = 1.0;
                        }
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }


                alg_ids = {ALG_ID_MENTS};
                vector<double> temps = { 1.0, 0.5, 0.2, 0.15, 0.10, 0.05, 0.01 };
                vector<double> epss = { 10.0, 1.0, 0.1, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        for (double eps : epss) {
                            unordered_map<string,double> alg_params = 
                                {
                                    {PARAMS_ID_MENTS_TEMP, temp}, 
                                    {PARAMS_ID_MENTS_EPSILON, eps}
                                };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }

                alg_ids = {ALG_ID_RENTS};
                temps = { 1000.0, 100.0, 10.0, 1.0, 0.1, 0.01 };
                epss = { 10.0, 1.0, 0.1, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        for (double eps : epss) {
                            unordered_map<string,double> alg_params = 
                                {
                                    {PARAMS_ID_MENTS_TEMP, temp}, 
                                    {PARAMS_ID_MENTS_EPSILON, eps}
                                };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }

                alg_ids = {ALG_ID_TENTS};
                temps = { 10.0, 1.0, 0.7, 0.5, 0.3, 0.1, 0.01 };
                epss = { 10.0, 1.0, 0.1, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        for (double eps : epss) {
                            unordered_map<string,double> alg_params = 
                                {
                                    {PARAMS_ID_MENTS_TEMP, temp}, 
                                    {PARAMS_ID_MENTS_EPSILON, eps}
                                };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }

                alg_ids = {ALG_ID_DENTS, ALG_ID_EST};
                temps = { 1.0, 0.5, 0.2, 0.15, 0.10, 0.05, 0.01 };
                epss = { 10.0, 1.0, 0.1, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        for (double eps : epss) {
                            unordered_map<string,double> alg_params = 
                                {
                                    {PARAMS_ID_MENTS_TEMP, temp}, 
                                    {PARAMS_ID_MENTS_EPSILON, eps}
                                };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }
            }

            return run_ids;
        }


        // expr_id: D005_LEN20
        // Runs algorithms with varying parameters on the 20-chain
        if (expr_id == D005_LEN20) {
            string env_id = DCHAIN_ENV_ID;
            vector<string> env_instance_ids = {D_20_ID, D_20_HALF_ID};
            int num_trials = 25000;
            int max_trial_length = 100; 
            int trials_log_delta = 10;
            int mc_eval_trials_delta = 10;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 32;
            int eval_threads = 32;

            for (string env_instance_id : env_instance_ids) {
                
                vector<string> alg_ids = {ALG_ID_MENTS};
                vector<double> temps = { 10.0, 1.0, 0.5, 0.2, 0.15, 0.10, 0.05 };
                vector<double> epss = { 10.0, 1.0, 0.1, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        for (double eps : epss) {
                            unordered_map<string,double> alg_params = 
                                {
                                    {PARAMS_ID_MENTS_TEMP, temp}, 
                                    {PARAMS_ID_MENTS_EPSILON, eps}
                                };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }
            }

            string env_instance_id = D_20_ID;

            vector<string> alg_ids = {ALG_ID_DENTS, ALG_ID_EST};
            vector<double> temps = { 10.0, 1.0, 0.5, 0.2, 0.15, 0.10, 0.05 };
            vector<double> epss = { 10.0, 1.0, 0.1, 0.01 };
            for (string alg_id : alg_ids) {
                for (double temp : temps) {
                    for (double eps : epss) {
                        unordered_map<string,double> alg_params = 
                            {
                                {PARAMS_ID_MENTS_TEMP, temp}, 
                                {PARAMS_ID_MENTS_EPSILON, eps}
                            };
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }
            }

            return run_ids;
        }


        // expr_id: D006_LEN20
        // 20-chain with DENTS without decaying temp
        if (expr_id == D006_LEN20) {
            string env_id = DCHAIN_ENV_ID;
            string env_instance_id = D_20_ID;
            int num_trials = 25000;
            int max_trial_length = 100; 
            int trials_log_delta = 10;
            int mc_eval_trials_delta = 10;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 32;
            int eval_threads = 32;

            vector<string> alg_ids = {ALG_ID_DENTS};
            vector<double> temps = { 10.0, 1.0, 0.5, 0.2, 0.15, 0.10, 0.05 };
            vector<double> epss = { 10.0, 1.0, 0.1, 0.01 };
            for (string alg_id : alg_ids) {
                for (double temp : temps) {
                    for (double eps : epss) {
                        unordered_map<string,double> alg_params = 
                            {
                                {PARAMS_ID_MENTS_TEMP, temp}, 
                                {PARAMS_ID_MENTS_EPSILON, eps},
                                {PARAMS_ID_DENTS_IGNORE_DECAY, 1.0},
                            };
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }
            }

            return run_ids;
        }


        // expr_id: D006_LEN20
        // 20-chain with DENTS tuned
        if (expr_id == D007_LEN20) {
            string env_id = DCHAIN_ENV_ID;
            string env_instance_id = D_20_ID;
            int num_trials = 25000;
            int max_trial_length = 100; 
            int trials_log_delta = 10;
            int mc_eval_trials_delta = 10;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 32;
            int eval_threads = 32;

            string alg_id = ALG_ID_DENTS;
            unordered_map<string,double> alg_params = 
                {
                    {PARAMS_ID_MENTS_TEMP, 0.5}, 
                    {PARAMS_ID_MENTS_EPSILON, 0.01},
                    {PARAMS_ID_DENTS_TEMP, 10.0},
                };
            run_ids->push_back(RunID(
                env_id,
                env_instance_id,
                expr_id,
                alg_id,
                alg_params,
                num_trials,
                max_trial_length,
                trials_log_delta,
                mc_eval_trials_delta,
                rollouts_per_mc_eval,
                num_repeats,
                num_threads,
                eval_threads));
                
            return run_ids;
        }

        // expr id: D021_LEN10_PAPER = "100_len_10_main_paper"
        // rerunning with specific parameters with more replicates to make curves smoother for nice plots
        if (expr_id == D021_LEN10_PAPER) {
            string env_id = DCHAIN_ENV_ID;
            vector<string> env_instance_ids = {D_10_ID, D_10_HALF_ID};
            int num_trials = 10000;
            int max_trial_length = 100; 
            int trials_log_delta = 10;
            int mc_eval_trials_delta = 10;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 100;
            int num_threads = 16;
            int eval_threads = 32;

            for (string env_instance_id : env_instance_ids) {

                vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_ER_UCT, ALG_ID_FIXED_DEPTH_UCT, ALG_ID_ER_FIXED_DEPTH_UCT, ALG_ID_PUCT};
                vector<double> uct_biases = { UctManagerArgs::USE_AUTO_BIAS };
                for (string alg_id : alg_ids) {
                    for (double bias : uct_biases) {
                        unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, bias}};
                        if (alg_id == ALG_ID_ER_UCT || alg_id == ALG_ID_ER_FIXED_DEPTH_UCT) {
                            alg_params[PARAMS_ID_UCT_ER_C2] = 1.0;
                        }
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }

                alg_ids = {ALG_ID_MENTS};
                vector<double> temps = { 1.0, 0.01 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        unordered_map<string,double> alg_params = 
                            {
                                {PARAMS_ID_MENTS_TEMP, temp}, 
                                {PARAMS_ID_MENTS_EPSILON, 0.1}
                            };
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }

                alg_ids = {ALG_ID_DENTS};
                temps = { 1.0 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        unordered_map<string,double> alg_params = 
                            {
                                {PARAMS_ID_MENTS_TEMP, temp}, 
                                {PARAMS_ID_MENTS_EPSILON, 0.1}
                            };
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }

                // adding DENTS with fixed beta(m)=alpha to demonstrate dents mimicing ments 
                // (and just copied MENTS params without tunring)
                alg_ids = {ALG_ID_DBMENTS};
                temps = { 1.0 };
                for (string alg_id : alg_ids) {
                    for (double temp : temps) {
                        unordered_map<string,double> alg_params = 
                            {
                                {PARAMS_ID_MENTS_TEMP, temp}, 
                                {PARAMS_ID_MENTS_EPSILON, 0.1}
                            };
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }

            }

            return run_ids;
        }










        // expr id: FL12_051_HPS
        // Runs a hyperparameter search on all algos for frozen lake
        if (expr_id == FL12_051_HPS) {
            string env_id = FL_ENV_ID;
            string env_instance_id = FL_8x12;
            int num_trials = 300000;
            int max_trial_length = 100;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 500;
            int rollouts_per_mc_eval = 50;
            int num_repeats = 5;
            int num_threads = 32;
            int eval_threads = 32;

            vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_ER_UCT, ALG_ID_FIXED_DEPTH_UCT, ALG_ID_ER_FIXED_DEPTH_UCT, ALG_ID_PUCT};
            vector<double> uct_biases = { UctManagerArgs::USE_AUTO_BIAS, 0.1, 1.0, 10.0, 100.0 };
            for (string alg_id : alg_ids) {
                for (double bias : uct_biases) {
                    unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, bias}};
                    if (alg_id == ALG_ID_ER_UCT || alg_id == ALG_ID_ER_FIXED_DEPTH_UCT) {
                        alg_params[PARAMS_ID_UCT_ER_C2] = 1.0;
                    }
                    run_ids->push_back(RunID(
                        env_id,
                        env_instance_id,
                        expr_id,
                        alg_id,
                        alg_params,
                        num_trials,
                        max_trial_length,
                        trials_log_delta,
                        mc_eval_trials_delta,
                        rollouts_per_mc_eval,
                        num_repeats,
                        num_threads,
                        eval_threads));
                }
            }

            alg_ids = {ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS, ALG_ID_EST, ALG_ID_DENTS };
            vector<double> temps = {100.0, 10.0, 1.0, 0.1, 0.01, 0.001};
            vector<double> epss = {2.0, 1.0, 0.3, 0.1, 0.03, 0.01};
            for (string alg_id : alg_ids) {
                for (double temp : temps) {
                    for (double eps : epss) {
                        unordered_map<string,double> alg_params = 
                            {
                                {PARAMS_ID_MENTS_TEMP, temp}, 
                                {PARAMS_ID_MENTS_EPSILON, eps},
                            };
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));       
                    }
                }
            }
            return run_ids;
        }

        // expr id: FL12_051A_HPS
        // Tunes the dents temperature after the rest of the params are set
        if (expr_id == FL12_051A_HPS) {
            string env_id = FL_ENV_ID;
            string env_instance_id = FL_8x12;
            int num_trials = 300000;
            int max_trial_length = 100;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 500;
            int rollouts_per_mc_eval = 50;
            int num_repeats = 5;
            int num_threads = 32;
            int eval_threads = 32;

            string alg_id = ALG_ID_DENTS;
            double m_temp = 0.1;
            double eps = 1.0;
            vector<double> d_temps = {100.0, 10.0, 1.0, 0.1};

            for (double d_temp : d_temps) {
                unordered_map<string,double> alg_params = 
                    {
                        {PARAMS_ID_MENTS_TEMP, m_temp}, 
                        {PARAMS_ID_MENTS_EPSILON, eps},
                        {PARAMS_ID_DENTS_TEMP, d_temp}, 
                    };
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));    
            }

            return run_ids;
        }

        // expr id: FL12_052_TEST
        // Test envs for hps selected params
        if (expr_id == FL12_052_TEST) {
            string env_id = FL_ENV_ID;
            string env_instance_id = FL_8x12_TEST;
            int num_trials = 500000;
            int max_trial_length = 100; 
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 250;
            int num_repeats = 25;
            int num_threads = 32;
            int eval_threads = 32;

            vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_ER_UCT, ALG_ID_FIXED_DEPTH_UCT, ALG_ID_ER_FIXED_DEPTH_UCT, ALG_ID_PUCT};
            for (string alg_id : alg_ids) {
                double bias = UctManagerArgs::USE_AUTO_BIAS;
                unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, bias}};
                if (alg_id == ALG_ID_ER_UCT || alg_id == ALG_ID_ER_FIXED_DEPTH_UCT) {
                    alg_params[PARAMS_ID_UCT_ER_C2] = 1.0;
                }
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));
            }

            alg_ids = {ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS, ALG_ID_EST, ALG_ID_DENTS};
            for (string alg_id : alg_ids) {
                double temp = 1.0;
                double eps = 1.0;
                if (alg_id == ALG_ID_MENTS) {
                    temp = 0.001;
                    eps = 1.0;
                } else if (alg_id == ALG_ID_RENTS) {
                    temp = 0.001;
                    eps = 2.0;
                } else if (alg_id == ALG_ID_TENTS) {
                    temp = 0.001;
                    eps = 1.0;
                } else if (alg_id == ALG_ID_EST) {
                    temp = 0.1;
                    eps = 2.0;
                } else if (alg_id == ALG_ID_DENTS) {
                    temp = 0.1;
                    eps = 1.0;
                } else {
                    throw runtime_error("error in FL12_052_TEST");
                }
                unordered_map<string,double> alg_params = 
                    {
                        {PARAMS_ID_MENTS_TEMP, temp}, 
                        {PARAMS_ID_MENTS_EPSILON, eps}
                    };
                if (alg_id == ALG_ID_DENTS) {
                    alg_params.insert_or_assign(PARAMS_ID_DENTS_TEMP, 1.0);
                }
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));
            }

            // adding DENTS with fixed beta(m)=alpha to demonstrate dents mimicing ments 
            // (and just copied MENTS params without tunring)
            alg_ids = {ALG_ID_DBMENTS};
            vector<double> temps = { 0.01 };
            vector<double> epss = { 1.0 };
            for (string alg_id : alg_ids) {
                for (double temp : temps) {
                    for (double eps : epss) {
                        unordered_map<string,double> alg_params = 
                            {
                                {PARAMS_ID_MENTS_TEMP, temp}, 
                                {PARAMS_ID_MENTS_EPSILON, eps}
                            };
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }
            }
            return run_ids;
        }

        // expr id: FL12_060_ER_UCT_SMOKE
        // Small Frozen Lake smoke test for UCT vs ER-UCT
        if (expr_id == FL12_060_ER_UCT_SMOKE) {
            string env_id = FL_ENV_ID;
            string env_instance_id = FL_8x12_TEST;
            int num_trials = 50000;
            int max_trial_length = 100;
            int trials_log_delta = 100;
            int mc_eval_trials_delta = 500;
            int rollouts_per_mc_eval = 50;
            int num_repeats = 1;
            int num_threads = 4;
            int eval_threads = 4;

            vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_ER_UCT};
            for (string alg_id : alg_ids) {
                unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, UctManagerArgs::USE_AUTO_BIAS}};
                if (alg_id == ALG_ID_ER_UCT) {
                    alg_params[PARAMS_ID_UCT_ER_C2] = 1.0;
                }
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));
            }

            return run_ids;
        }

        // expr id: FL12_061_ER_MENTS_SMOKE
        // Small Frozen Lake smoke test for MENTS-family ER variants
        if (expr_id == FL12_061_ER_MENTS_SMOKE) {
            string env_id = FL_ENV_ID;
            string env_instance_id = FL_8x12_TEST;
            int num_trials = 5000;
            int max_trial_length = 100;
            int trials_log_delta = 100;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 50;
            int num_repeats = 1;
            int num_threads = 4;
            int eval_threads = 4;

            vector<string> alg_ids = {ALG_ID_MENTS, ALG_ID_ER_MENTS, ALG_ID_RENTS, ALG_ID_ER_RENTS, ALG_ID_TENTS, ALG_ID_ER_TENTS};
            for (string alg_id : alg_ids) {
                double temp = 1.0;
                double eps = 1.0;
                if (alg_id == ALG_ID_MENTS || alg_id == ALG_ID_ER_MENTS) {
                    temp = 0.001;
                    eps = 1.0;
                } else if (alg_id == ALG_ID_RENTS || alg_id == ALG_ID_ER_RENTS) {
                    temp = 0.001;
                    eps = 2.0;
                } else if (alg_id == ALG_ID_TENTS || alg_id == ALG_ID_ER_TENTS) {
                    temp = 0.001;
                    eps = 1.0;
                } else {
                    throw runtime_error("error in FL12_061_ER_MENTS_SMOKE");
                }

                unordered_map<string,double> alg_params = {
                    {PARAMS_ID_MENTS_TEMP, temp},
                    {PARAMS_ID_MENTS_EPSILON, eps},
                    {PARAMS_ID_MENTS_POWER_MEAN_P, 1.0},
                    {PARAMS_ID_UCT_ER_C2, 1.0}
                };

                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));
            }

            return run_ids;
        }

        // expr id: FL12_062_REQUESTED_ALGS
        // Frozen Lake preset for all supported algorithms.
        if (expr_id == FL12_062_REQUESTED_ALGS) {
            string env_id = FL_ENV_ID;
            string env_instance_id = FL_8x12_TEST;
            int num_trials = 250000;
            int max_trial_length = 100;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 500;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            vector<string> alg_ids = {
                ALG_ID_UCT,
                ALG_ID_ER_UCT,
                ALG_ID_FIXED_DEPTH_UCT,
                ALG_ID_ER_FIXED_DEPTH_UCT,
                ALG_ID_PUCT,
                ALG_ID_MENTS,
                ALG_ID_ER_MENTS,
                ALG_ID_RENTS,
                ALG_ID_ER_RENTS,
                ALG_ID_TENTS,
                ALG_ID_ER_TENTS,
                ALG_ID_DENTS,
                ALG_ID_DBMENTS,
                ALG_ID_EST
            };
            for (string alg_id : alg_ids) {
                unordered_map<string,double> alg_params;

                if (alg_id == ALG_ID_UCT || alg_id == ALG_ID_ER_UCT || alg_id == ALG_ID_FIXED_DEPTH_UCT || alg_id == ALG_ID_ER_FIXED_DEPTH_UCT || alg_id == ALG_ID_PUCT) {
                    alg_params[PARAMS_ID_UCT_BIAS] = UctManagerArgs::USE_AUTO_BIAS;
                    if (alg_id == ALG_ID_ER_UCT || alg_id == ALG_ID_ER_FIXED_DEPTH_UCT) {
                        alg_params[PARAMS_ID_UCT_ER_C2] = 1.0;
                    }
                } else if (alg_id == ALG_ID_EST) {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.1;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = 2.0;
                } else if (alg_id == ALG_ID_DENTS || alg_id == ALG_ID_DBMENTS) {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.1;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = 1.0;
                    if (alg_id == ALG_ID_DENTS) {
                        alg_params[PARAMS_ID_DENTS_TEMP] = 1.0;
                    }
                } else {
                    double temp = 0.001;
                    double eps = 1.0;
                    if (alg_id == ALG_ID_RENTS || alg_id == ALG_ID_ER_RENTS) {
                        eps = 2.0;
                    }
                    alg_params[PARAMS_ID_MENTS_TEMP] = temp;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = eps;
                    alg_params[PARAMS_ID_MENTS_POWER_MEAN_P] = 1.0;
                }

                if (alg_id == ALG_ID_ER_MENTS || alg_id == ALG_ID_ER_RENTS || alg_id == ALG_ID_ER_TENTS) {
                    alg_params[PARAMS_ID_UCT_ER_C2] = 1.0;
                }

                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));
            }

            return run_ids;
        }

        // expr id: FL12_063_ER_TUNE
        // Frozen Lake 8x12 grid search for ER-UCT, ER-fixed-depth-UCT, ER-MENTS, ER-RENTS, and ER-TENTS.
        // Tunes on the *train* map (FL_8x12) to match the FL12_051_HPS convention; baselines have
        // already been tuned on the same instance, so picking ER hyperparameters here keeps the
        // train/test split consistent with the published methodology.
        if (expr_id == FL12_063_ER_TUNE) {
            string env_id = FL_ENV_ID;
            string env_instance_id = FL_8x12;
            int num_trials = 150000;
            int max_trial_length = 100;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            // power_mean_p applies only to UctDNode::backup_average_return; MentsManager.power_mean_p
            // is currently unread, so we don't sweep it for the MENTS-family below.
            vector<double> power_mean_ps = {1.0, 2.0, 4.0, std::numeric_limits<double>::infinity()};

            vector<string> uct_alg_ids = {ALG_ID_ER_UCT, ALG_ID_ER_FIXED_DEPTH_UCT};
            vector<double> uct_biases = {UctManagerArgs::USE_AUTO_BIAS, 0.1, 0.3, 1.0, 3.0, 10.0};
            vector<double> er_c2s = {0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 50.0, 100.0};
            for (string alg_id : uct_alg_ids) {
                for (double bias : uct_biases) {
                    for (double er_c2 : er_c2s) {
                        for (double power_mean_p : power_mean_ps) {
                            unordered_map<string,double> alg_params = {
                                {PARAMS_ID_UCT_BIAS, bias},
                                {PARAMS_ID_UCT_ER_C2, er_c2},
                                {PARAMS_ID_UCT_POWER_MEAN_P, power_mean_p}
                            };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }
            }

            vector<string> ments_alg_ids = {ALG_ID_ER_MENTS, ALG_ID_ER_RENTS, ALG_ID_ER_TENTS};
            vector<double> temps = {0.001, 0.01, 0.05, 0.1, 0.5};
            vector<double> epss = {0.1, 0.3, 1.0, 2.0, 5.0};
            vector<double> er_c2s_ments = {0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 50.0, 100.0};
            for (string alg_id : ments_alg_ids) {
                for (double temp : temps) {
                    for (double eps : epss) {
                        for (double er_c2 : er_c2s_ments) {
                            unordered_map<string,double> alg_params = {
                                {PARAMS_ID_MENTS_TEMP, temp},
                                {PARAMS_ID_MENTS_EPSILON, eps},
                                {PARAMS_ID_UCT_ER_C2, er_c2}
                            };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }
            }

            return run_ids;
        }

        // expr id: FL16_050_TEST
        // Test envs for hps selected params
        if (expr_id == FL16_050_TEST) {
            string env_id = FL_ENV_ID;
            string env_instance_id = FL_8x16_TEST;
            int num_trials = 500000;
            int max_trial_length = 100; 
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 250;
            int num_repeats = 25;
            int num_threads = 32;
            int eval_threads = 32;

            vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_ER_UCT, ALG_ID_FIXED_DEPTH_UCT, ALG_ID_ER_FIXED_DEPTH_UCT, ALG_ID_PUCT};
            for (string alg_id : alg_ids) {
                double bias = UctManagerArgs::USE_AUTO_BIAS;
                unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, bias}};
                if (alg_id == ALG_ID_ER_UCT || alg_id == ALG_ID_ER_FIXED_DEPTH_UCT) {
                    alg_params[PARAMS_ID_UCT_ER_C2] = 1.0;
                }
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));
            }

            alg_ids = {ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS, ALG_ID_EST, ALG_ID_DENTS};
            for (string alg_id : alg_ids) {
                double temp = 1.0;
                double eps = 1.0;
                if (alg_id == ALG_ID_MENTS) {
                    temp = 0.001;
                    eps = 1.0;
                } else if (alg_id == ALG_ID_RENTS) {
                    temp = 0.001;
                    eps = 2.0;
                } else if (alg_id == ALG_ID_TENTS) {
                    temp = 0.001;
                    eps = 1.0;
                } else if (alg_id == ALG_ID_EST) {
                    temp = 0.1;
                    eps = 2.0;
                } else if (alg_id == ALG_ID_DENTS) {
                    temp = 0.1;
                    eps = 1.0;
                } else {
                    throw runtime_error("error in FL12_052_TEST");
                }
                unordered_map<string,double> alg_params = 
                    {
                        {PARAMS_ID_MENTS_TEMP, temp}, 
                        {PARAMS_ID_MENTS_EPSILON, eps}
                    };
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));
            }

            return run_ids;
        }

        // expr id: FL8_05X_SENS
        // Experiments showing sensitivity of ments temperature
        // Runs a hyperparameter search on all algos for frozen lake
        if (expr_id == FL8_053_SENS ||
            expr_id == FL8_054_SENS ||
            expr_id == FL8_055_SENS ||
            expr_id == FL8_056_SENS ||
            expr_id == FL8_057_SENS ||
            expr_id == FL8_058_SENS ||
            expr_id == FL8_059_SENS)
        {
            string env_id = FL_ENV_ID;
            string env_instance_id = FL_8x8;
            int num_trials = 300000;
            int max_trial_length = 100;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 250;
            int num_repeats = 15;
            int num_threads = 32;
            int eval_threads = 32;

            vector<string> alg_ids = {ALG_ID_MENTS, ALG_ID_EST, ALG_ID_DENTS, ALG_ID_RENTS, ALG_ID_TENTS};
            alg_ids = {ALG_ID_RENTS, ALG_ID_TENTS};
            double temp = 1.0;
            double eps = 1.0;
            if (expr_id == FL8_053_SENS) {
                temp = 1.0;
            } else if (expr_id == FL8_054_SENS) {
                temp = 0.5;
            } else if (expr_id == FL8_055_SENS) {
                temp = 0.1;
            } else if (expr_id == FL8_056_SENS) {
                temp = 0.05;
            } else if (expr_id == FL8_057_SENS) {
                temp = 0.01;
            } else if (expr_id == FL8_058_SENS) {
                temp = 0.005;
            } else if (expr_id == FL8_059_SENS) {
                temp = 0.001;
            } else {
                throw runtime_error("something happened");
            }
            
            for (string alg_id : alg_ids) {
                unordered_map<string,double> alg_params = 
                    {
                        {PARAMS_ID_MENTS_TEMP, temp}, 
                        {PARAMS_ID_MENTS_EPSILON, eps},
                    };
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));  
            }

            return run_ids;
        }










        

        // expr id: S6_091_HPS
        // Runs a hyperparameter search on all algos for sailing 6x6 env
        if (expr_id == S6_091_HPS) {
            string env_id = SAILING_ENV_ID;
            string env_instance_id = S_6_ID;
            int num_trials = 250000;
            int max_trial_length = 50;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 250;
            int num_repeats = 5;
            int num_threads = 32;
            int eval_threads = 32;

            double default_q_value = -200.0;

            vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_ER_UCT, ALG_ID_FIXED_DEPTH_UCT, ALG_ID_ER_FIXED_DEPTH_UCT, ALG_ID_PUCT};
            vector<double> uct_biases = { UctManagerArgs::USE_AUTO_BIAS, 0.1, 1.0, 10.0, 100.0 };
            for (string alg_id : alg_ids) {
                for (double bias : uct_biases) {
                    unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, bias}};
                    if (alg_id == ALG_ID_ER_UCT || alg_id == ALG_ID_ER_FIXED_DEPTH_UCT) {
                        alg_params[PARAMS_ID_UCT_ER_C2] = 1.0;
                    }
                    run_ids->push_back(RunID(
                        env_id,
                        env_instance_id,
                        expr_id,
                        alg_id,
                        alg_params,
                        num_trials,
                        max_trial_length,
                        trials_log_delta,
                        mc_eval_trials_delta,
                        rollouts_per_mc_eval,
                        num_repeats,
                        num_threads,
                        eval_threads));
                }
            }

            alg_ids = {ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS, ALG_ID_EST, ALG_ID_DENTS };
            vector<double> temps = {100.0, 10.0, 1.0, 0.1, 0.01, 0.001};
            vector<double> epss = {2.0, 1.0, 0.3, 0.1}; //, 0.03, 0.01};
            for (string alg_id : alg_ids) {
                for (double temp : temps) {
                    for (double eps : epss) {
                        unordered_map<string,double> alg_params = 
                            {
                                {PARAMS_ID_MENTS_TEMP, temp}, 
                                {PARAMS_ID_MENTS_EPSILON, eps},
                                {PARAMS_ID_MENTS_DEFAULT_Q_VALUE, default_q_value},
                            };
                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));       
                    }
                }
            }
            return run_ids;
        }
        
        // expr id: S6_092_TEST
        // Tests sailing 6x6 env, using a different starting direction
        if (expr_id == S6_092_TEST) {
            string env_id = SAILING_ENV_ID;
            string env_instance_id = S_6_TEST_ID;
            int num_trials = 300000;
            int max_trial_length = 50;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 250;
            int num_repeats = 25;
            int num_threads = 32;
            int eval_threads = 32;

            double default_q_value = -200.0;

            vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_ER_UCT, ALG_ID_FIXED_DEPTH_UCT, ALG_ID_ER_FIXED_DEPTH_UCT, ALG_ID_PUCT};
            for (string alg_id : alg_ids) {
                double bias = UctManagerArgs::USE_AUTO_BIAS;
                unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, bias}};
                if (alg_id == ALG_ID_ER_UCT || alg_id == ALG_ID_ER_FIXED_DEPTH_UCT) {
                    alg_params[PARAMS_ID_UCT_ER_C2] = 1.0;
                }
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));
            }

            alg_ids = {ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS, ALG_ID_EST, ALG_ID_DENTS};
            for (string alg_id : alg_ids) {
                double temp = 1.0;
                double eps = 1.0;
                if (alg_id == ALG_ID_MENTS) {
                    temp = 10.0;
                    eps = 1.0;
                } else if (alg_id == ALG_ID_RENTS) {
                    temp = 10.0;
                    eps = 1.0;
                } else if (alg_id == ALG_ID_TENTS) {
                    temp = 0.1;
                    eps = 2.0;
                } else if (alg_id == ALG_ID_EST) {
                    temp = 10.0;
                    eps = 1.0;
                } else if (alg_id == ALG_ID_DENTS) {
                    temp = 10.0;
                    eps = 1.0;
                } else {
                    throw runtime_error("error in S001_5");
                }
                unordered_map<string,double> alg_params = 
                        {
                            {PARAMS_ID_MENTS_TEMP, temp}, 
                            {PARAMS_ID_MENTS_EPSILON, eps},
                            {PARAMS_ID_MENTS_DEFAULT_Q_VALUE, default_q_value},
                        };
                if (alg_id == ALG_ID_DENTS) {
                    alg_params.insert_or_assign(PARAMS_ID_DENTS_TEMP, 10.0);
                }
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));
            }
            return run_ids;
        }

        // expr id: S6_093_ER_TUNE
        // Sailing 6x6 grid search for ER-UCT, ER-fixed-depth-UCT, ER-MENTS, ER-RENTS, and ER-TENTS.
        // MENTS-family algorithms get the same default_q_value = -200 used by S6_091_HPS / S6_092_TEST,
        // since sailing has only negative rewards: a default of 0 leaves unvisited actions wildly
        // optimistic and drives MENTS into BFS-like behaviour over unvisited children.
        if (expr_id == S6_093_ER_TUNE) {
            string env_id = SAILING_ENV_ID;
            string env_instance_id = S_6_ID;
            int num_trials = 150000;
            int max_trial_length = 50;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 250;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            double default_q_value = -200.0;

            // power_mean_p applies only to UctDNode::backup_average_return; MentsManager.power_mean_p
            // is currently unread, so we don't sweep it for the MENTS-family below.
            vector<double> power_mean_ps = {1.0, 2.0, 4.0, std::numeric_limits<double>::infinity()};

            vector<string> uct_alg_ids = {ALG_ID_ER_UCT, ALG_ID_ER_FIXED_DEPTH_UCT};
            vector<double> uct_biases = {UctManagerArgs::USE_AUTO_BIAS, 0.1, 0.3, 1.0, 3.0, 10.0};
            vector<double> er_c2s = {0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 50.0, 100.0};
            for (string alg_id : uct_alg_ids) {
                for (double bias : uct_biases) {
                    for (double er_c2 : er_c2s) {
                        for (double power_mean_p : power_mean_ps) {
                            unordered_map<string,double> alg_params = {
                                {PARAMS_ID_UCT_BIAS, bias},
                                {PARAMS_ID_UCT_ER_C2, er_c2},
                                {PARAMS_ID_UCT_POWER_MEAN_P, power_mean_p}
                            };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }
            }

            vector<string> ments_alg_ids = {ALG_ID_ER_MENTS, ALG_ID_ER_RENTS, ALG_ID_ER_TENTS};
            vector<double> temps = {0.001, 0.01, 0.05, 0.1, 0.5};
            vector<double> epss = {0.1, 0.3, 1.0, 2.0, 5.0};
            vector<double> er_c2s_ments = {0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 50.0, 100.0};
            for (string alg_id : ments_alg_ids) {
                for (double temp : temps) {
                    for (double eps : epss) {
                        for (double er_c2 : er_c2s_ments) {
                            unordered_map<string,double> alg_params = {
                                {PARAMS_ID_MENTS_TEMP, temp},
                                {PARAMS_ID_MENTS_EPSILON, eps},
                                {PARAMS_ID_MENTS_DEFAULT_Q_VALUE, default_q_value},
                                {PARAMS_ID_UCT_ER_C2, er_c2}
                            };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }
            }

            return run_ids;
        }

        // expr id: TX5_101_HPS
        // Hyperparameter search on Taxi 5x5.
        if (expr_id == TX5_101_HPS) {
            string env_id = TAXI_ENV_ID;
            string env_instance_id = TX_5_ID;
            int num_trials = 200000;
            int max_trial_length = 50;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 5;
            int num_threads = 16;
            int eval_threads = 16;

            vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_ER_UCT, ALG_ID_FIXED_DEPTH_UCT, ALG_ID_ER_FIXED_DEPTH_UCT, ALG_ID_PUCT};
            vector<double> uct_biases = {UctManagerArgs::USE_AUTO_BIAS, 0.1, 1.0, 10.0};
            for (string alg_id : alg_ids) {
                for (double bias : uct_biases) {
                    unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, bias}};
                    if (alg_id == ALG_ID_ER_UCT || alg_id == ALG_ID_ER_FIXED_DEPTH_UCT) {
                        alg_params[PARAMS_ID_UCT_ER_C2] = 1.0;
                        alg_params[PARAMS_ID_UCT_POWER_MEAN_P] = 2.0;
                    }
                    run_ids->push_back(RunID(
                        env_id,
                        env_instance_id,
                        expr_id,
                        alg_id,
                        alg_params,
                        num_trials,
                        max_trial_length,
                        trials_log_delta,
                        mc_eval_trials_delta,
                        rollouts_per_mc_eval,
                        num_repeats,
                        num_threads,
                        eval_threads));
                }
            }

            alg_ids = {ALG_ID_MENTS, ALG_ID_ER_MENTS, ALG_ID_RENTS, ALG_ID_ER_RENTS, ALG_ID_TENTS, ALG_ID_ER_TENTS, ALG_ID_EST, ALG_ID_DENTS, ALG_ID_DBMENTS};
            vector<double> temps = {1.0, 0.1, 0.01, 0.001};
            vector<double> epss = {2.0, 1.0, 0.3, 0.1};
            for (string alg_id : alg_ids) {
                for (double temp : temps) {
                    for (double eps : epss) {
                        unordered_map<string,double> alg_params = {
                            {PARAMS_ID_MENTS_TEMP, temp},
                            {PARAMS_ID_MENTS_EPSILON, eps}
                        };
                        if (alg_id == ALG_ID_DENTS) {
                            alg_params[PARAMS_ID_DENTS_TEMP] = 1.0;
                        }
                        if (alg_id == ALG_ID_ER_MENTS || alg_id == ALG_ID_ER_RENTS || alg_id == ALG_ID_ER_TENTS) {
                            alg_params[PARAMS_ID_UCT_ER_C2] = 1.0;
                            alg_params[PARAMS_ID_MENTS_POWER_MEAN_P] = 2.0;
                        }

                        run_ids->push_back(RunID(
                            env_id,
                            env_instance_id,
                            expr_id,
                            alg_id,
                            alg_params,
                            num_trials,
                            max_trial_length,
                            trials_log_delta,
                            mc_eval_trials_delta,
                            rollouts_per_mc_eval,
                            num_repeats,
                            num_threads,
                            eval_threads));
                    }
                }
            }

            return run_ids;
        }

        // expr id: TX5_102_TEST
        // Taxi 5x5 test run with selected baseline params.
        if (expr_id == TX5_102_TEST) {
            string env_id = TAXI_ENV_ID;
            string env_instance_id = TX_5_TEST_ID;
            int num_trials = 250000;
            int max_trial_length = 50;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 200;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            vector<string> alg_ids = {ALG_ID_UCT, ALG_ID_ER_UCT, ALG_ID_FIXED_DEPTH_UCT, ALG_ID_ER_FIXED_DEPTH_UCT, ALG_ID_PUCT};
            for (string alg_id : alg_ids) {
                unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, UctManagerArgs::USE_AUTO_BIAS}};
                if (alg_id == ALG_ID_ER_UCT || alg_id == ALG_ID_ER_FIXED_DEPTH_UCT) {
                    alg_params[PARAMS_ID_UCT_ER_C2] = 1.0;
                    alg_params[PARAMS_ID_UCT_POWER_MEAN_P] = 2.0;
                }
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));
            }

            alg_ids = {ALG_ID_MENTS, ALG_ID_ER_MENTS, ALG_ID_RENTS, ALG_ID_ER_RENTS, ALG_ID_TENTS, ALG_ID_ER_TENTS, ALG_ID_EST, ALG_ID_DENTS, ALG_ID_DBMENTS};
            for (string alg_id : alg_ids) {
                double temp = 0.01;
                double eps = (alg_id == ALG_ID_RENTS || alg_id == ALG_ID_ER_RENTS || alg_id == ALG_ID_EST) ? 2.0 : 1.0;
                unordered_map<string,double> alg_params = {
                    {PARAMS_ID_MENTS_TEMP, temp},
                    {PARAMS_ID_MENTS_EPSILON, eps}
                };
                if (alg_id == ALG_ID_DENTS) {
                    alg_params[PARAMS_ID_DENTS_TEMP] = 1.0;
                }
                if (alg_id == ALG_ID_ER_MENTS || alg_id == ALG_ID_ER_RENTS || alg_id == ALG_ID_ER_TENTS) {
                    alg_params[PARAMS_ID_UCT_ER_C2] = 1.0;
                    alg_params[PARAMS_ID_MENTS_POWER_MEAN_P] = 2.0;
                }
                run_ids->push_back(RunID(
                    env_id,
                    env_instance_id,
                    expr_id,
                    alg_id,
                    alg_params,
                    num_trials,
                    max_trial_length,
                    trials_log_delta,
                    mc_eval_trials_delta,
                    rollouts_per_mc_eval,
                    num_repeats,
                    num_threads,
                    eval_threads));
            }

            return run_ids;
        }

        // expr id: TX5_103_ER_TUNE
        // Taxi 5x5 grid search for ER-UCT, ER-fixed-depth-UCT, ER-MENTS, ER-RENTS, and ER-TENTS.
        if (expr_id == TX5_103_ER_TUNE) {
            string env_id = TAXI_ENV_ID;
            string env_instance_id = TX_5_ID;
            int num_trials = 150000;
            int max_trial_length = 50;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            // power_mean_p applies only to UctDNode::backup_average_return; MentsManager.power_mean_p
            // is currently unread, so we don't sweep it for the MENTS-family below.
            vector<double> power_mean_ps = {1.0, 2.0, 4.0, std::numeric_limits<double>::infinity()};

            vector<string> uct_alg_ids = {ALG_ID_ER_UCT, ALG_ID_ER_FIXED_DEPTH_UCT};
            vector<double> uct_biases = {UctManagerArgs::USE_AUTO_BIAS, 0.1, 0.3, 1.0, 3.0, 10.0};
            vector<double> er_c2s = {0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 50.0, 100.0};
            for (string alg_id : uct_alg_ids) {
                for (double bias : uct_biases) {
                    for (double er_c2 : er_c2s) {
                        for (double power_mean_p : power_mean_ps) {
                            unordered_map<string,double> alg_params = {
                                {PARAMS_ID_UCT_BIAS, bias},
                                {PARAMS_ID_UCT_ER_C2, er_c2},
                                {PARAMS_ID_UCT_POWER_MEAN_P, power_mean_p}
                            };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }
            }

            vector<string> ments_alg_ids = {ALG_ID_ER_MENTS, ALG_ID_ER_RENTS, ALG_ID_ER_TENTS};
            vector<double> temps = {0.001, 0.01, 0.05, 0.1, 0.5};
            vector<double> epss = {0.1, 0.3, 1.0, 2.0, 5.0};
            vector<double> er_c2s_ments = {0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 50.0, 100.0};
            for (string alg_id : ments_alg_ids) {
                for (double temp : temps) {
                    for (double eps : epss) {
                        for (double er_c2 : er_c2s_ments) {
                            unordered_map<string,double> alg_params = {
                                {PARAMS_ID_MENTS_TEMP, temp},
                                {PARAMS_ID_MENTS_EPSILON, eps},
                                {PARAMS_ID_UCT_ER_C2, er_c2}
                            };
                            run_ids->push_back(RunID(
                                env_id,
                                env_instance_id,
                                expr_id,
                                alg_id,
                                alg_params,
                                num_trials,
                                max_trial_length,
                                trials_log_delta,
                                mc_eval_trials_delta,
                                rollouts_per_mc_eval,
                                num_repeats,
                                num_threads,
                                eval_threads));
                        }
                    }
                }
            }

            return run_ids;
        }

        // expr id: FL12_064_BASELINES
        // Baselines on the FL12 train instance, sized to match FL12_063_ER_TUNE for fair plotting
        // (same num_trials, num_repeats, num_threads). Both write under the same instance directory
        // so a single comparison plot can read from both folders.
        if (expr_id == FL12_064_BASELINES) {
            string env_id = FL_ENV_ID;
            string env_instance_id = FL_8x12;
            int num_trials = 150000;
            int max_trial_length = 100;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            vector<string> alg_ids = { ALG_ID_UCT, ALG_ID_FIXED_DEPTH_UCT, ALG_ID_PUCT, ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS, ALG_ID_DENTS, ALG_ID_DBMENTS, ALG_ID_EST };
            for (string alg_id : alg_ids) {
                unordered_map<string,double> alg_params;
                if (alg_id == ALG_ID_UCT || alg_id == ALG_ID_FIXED_DEPTH_UCT || alg_id == ALG_ID_PUCT) {
                    alg_params[PARAMS_ID_UCT_BIAS] = UctManagerArgs::USE_AUTO_BIAS;
                    alg_params[PARAMS_ID_UCT_POWER_MEAN_P] = 1.0;
                } else if (alg_id == ALG_ID_EST) {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.1;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = 2.0;
                } else if (alg_id == ALG_ID_DENTS || alg_id == ALG_ID_DBMENTS) {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.1;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = 1.0;
                    if (alg_id == ALG_ID_DENTS) alg_params[PARAMS_ID_DENTS_TEMP] = 1.0;
                } else {
                    double temp = 0.001;
                    double eps = 1.0;
                    if (alg_id == ALG_ID_RENTS) eps = 2.0;
                    alg_params[PARAMS_ID_MENTS_TEMP] = temp;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = eps;
                    alg_params[PARAMS_ID_MENTS_POWER_MEAN_P] = 1.0;
                }
                run_ids->push_back(RunID(env_id, env_instance_id, expr_id, alg_id, alg_params, num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta, rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));
            }
            return run_ids;
        }

        // expr id: S6_094_BASELINES
        // Baselines on the sailing train instance, sized to match S6_093_ER_TUNE for fair plotting.
        // MENTS-family algorithms set default_q_value = -200 to match the published S6_091/S6_092
        // tuning convention -- without it the previously selected (temp, eps) values are operating
        // against a different optimisation surface (default_q_value=0 makes unvisited actions look
        // optimistic compared with any visited child whose Q is necessarily negative).
        if (expr_id == S6_094_BASELINES) {
            string env_id = SAILING_ENV_ID;
            string env_instance_id = S_6_ID;
            int num_trials = 150000;
            int max_trial_length = 50;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 250;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            double default_q_value = -200.0;

            vector<string> alg_ids = { ALG_ID_UCT, ALG_ID_FIXED_DEPTH_UCT, ALG_ID_PUCT, ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS, ALG_ID_DENTS, ALG_ID_DBMENTS, ALG_ID_EST };
            for (string alg_id : alg_ids) {
                unordered_map<string,double> alg_params;
                if (alg_id == ALG_ID_UCT || alg_id == ALG_ID_FIXED_DEPTH_UCT || alg_id == ALG_ID_PUCT) {
                    alg_params[PARAMS_ID_UCT_BIAS] = UctManagerArgs::USE_AUTO_BIAS;
                    alg_params[PARAMS_ID_UCT_POWER_MEAN_P] = 1.0;
                } else if (alg_id == ALG_ID_EST) {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.1;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = 2.0;
                    alg_params[PARAMS_ID_MENTS_DEFAULT_Q_VALUE] = default_q_value;
                } else if (alg_id == ALG_ID_DENTS || alg_id == ALG_ID_DBMENTS) {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.1;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = 1.0;
                    alg_params[PARAMS_ID_MENTS_DEFAULT_Q_VALUE] = default_q_value;
                    if (alg_id == ALG_ID_DENTS) alg_params[PARAMS_ID_DENTS_TEMP] = 1.0;
                } else {
                    double temp = 0.001;
                    double eps = 1.0;
                    if (alg_id == ALG_ID_RENTS) eps = 2.0;
                    alg_params[PARAMS_ID_MENTS_TEMP] = temp;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = eps;
                    alg_params[PARAMS_ID_MENTS_DEFAULT_Q_VALUE] = default_q_value;
                }
                run_ids->push_back(RunID(env_id, env_instance_id, expr_id, alg_id, alg_params, num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta, rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));
            }
            return run_ids;
        }

        // expr id: TX5_104_BASELINES
        if (expr_id == TX5_104_BASELINES) {
            string env_id = TAXI_ENV_ID;
            string env_instance_id = TX_5_ID;
            int num_trials = 150000;
            int max_trial_length = 50;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            vector<string> alg_ids = { ALG_ID_UCT, ALG_ID_FIXED_DEPTH_UCT, ALG_ID_PUCT, ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS, ALG_ID_DENTS, ALG_ID_DBMENTS, ALG_ID_EST };
            for (string alg_id : alg_ids) {
                unordered_map<string,double> alg_params;
                if (alg_id == ALG_ID_UCT || alg_id == ALG_ID_FIXED_DEPTH_UCT || alg_id == ALG_ID_PUCT) {
                    alg_params[PARAMS_ID_UCT_BIAS] = UctManagerArgs::USE_AUTO_BIAS;
                    alg_params[PARAMS_ID_UCT_POWER_MEAN_P] = 1.0;
                } else if (alg_id == ALG_ID_EST) {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.1;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = 2.0;
                } else if (alg_id == ALG_ID_DENTS || alg_id == ALG_ID_DBMENTS) {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.1;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = 1.0;
                    if (alg_id == ALG_ID_DENTS) alg_params[PARAMS_ID_DENTS_TEMP] = 1.0;
                } else {
                    double temp = 0.001;
                    double eps = 1.0;
                    if (alg_id == ALG_ID_RENTS) eps = 2.0;
                    alg_params[PARAMS_ID_MENTS_TEMP] = temp;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = eps;
                    alg_params[PARAMS_ID_MENTS_POWER_MEAN_P] = 1.0;
                }
                run_ids->push_back(RunID(env_id, env_instance_id, expr_id, alg_id, alg_params, num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta, rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));
            }
            return run_ids;
        }

        // expr id: FL12_065_ER_TEST
        // Evaluates the top-1 tuned ER hyperparameters from 063_fl12_er_tune on the held-out
        // FL_8x12_test instance. One fixed config per algorithm -- no grid sweep.
        // Best configs (from find_best_hyperparams.py on 063):
        //   ER-UCT:             power_mean_p=inf, er_c2=1,    bias=-1 (auto)
        //   ER-FIXED-DEPTH-UCT: power_mean_p=1,  er_c2=50,   bias=-1 (auto)
        //   ER-MENTS:           temp=0.001, epsilon=2,   er_c2=0.05
        //   ER-RENTS:           temp=0.001, epsilon=0.3, er_c2=0.3
        //   ER-TENTS:           temp=0.01,  epsilon=2,   er_c2=0.1
        if (expr_id == FL12_065_ER_TEST) {
            string env_id = FL_ENV_ID;
            string env_instance_id = FL_8x12_TEST;
            int num_trials = 150000;
            int max_trial_length = 100;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            // ER-UCT
            run_ids->push_back(RunID(env_id, env_instance_id, expr_id, ALG_ID_ER_UCT,
                {{PARAMS_ID_UCT_BIAS,         UctManagerArgs::USE_AUTO_BIAS},
                 {PARAMS_ID_UCT_POWER_MEAN_P, std::numeric_limits<double>::infinity()},
                 {PARAMS_ID_UCT_ER_C2,        1.0}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            // ER-FIXED-DEPTH-UCT
            run_ids->push_back(RunID(env_id, env_instance_id, expr_id, ALG_ID_ER_FIXED_DEPTH_UCT,
                {{PARAMS_ID_UCT_BIAS,         UctManagerArgs::USE_AUTO_BIAS},
                 {PARAMS_ID_UCT_POWER_MEAN_P, 1.0},
                 {PARAMS_ID_UCT_ER_C2,        50.0}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            // ER-MENTS
            run_ids->push_back(RunID(env_id, env_instance_id, expr_id, ALG_ID_ER_MENTS,
                {{PARAMS_ID_MENTS_TEMP,    0.001},
                 {PARAMS_ID_MENTS_EPSILON, 2.0},
                 {PARAMS_ID_UCT_ER_C2,     0.05}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            // ER-RENTS
            run_ids->push_back(RunID(env_id, env_instance_id, expr_id, ALG_ID_ER_RENTS,
                {{PARAMS_ID_MENTS_TEMP,    0.001},
                 {PARAMS_ID_MENTS_EPSILON, 0.3},
                 {PARAMS_ID_UCT_ER_C2,     0.3}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            // ER-TENTS
            run_ids->push_back(RunID(env_id, env_instance_id, expr_id, ALG_ID_ER_TENTS,
                {{PARAMS_ID_MENTS_TEMP,    0.01},
                 {PARAMS_ID_MENTS_EPSILON, 2.0},
                 {PARAMS_ID_UCT_ER_C2,     0.1}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            return run_ids;
        }

        // expr id: FL12_066_BASELINES_TEST
        // Baselines on the FL_8x12_test held-out instance, sized to match FL12_065_ER_TEST
        // for fair apples-to-apples comparison (same num_trials, num_repeats, num_threads).
        if (expr_id == FL12_066_BASELINES_TEST) {
            string env_id = FL_ENV_ID;
            string env_instance_id = FL_8x12_TEST;
            int num_trials = 150000;
            int max_trial_length = 100;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            vector<string> alg_ids = { ALG_ID_UCT, ALG_ID_FIXED_DEPTH_UCT, ALG_ID_PUCT, ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS, ALG_ID_DENTS, ALG_ID_DBMENTS, ALG_ID_EST };
            for (string alg_id : alg_ids) {
                unordered_map<string,double> alg_params;
                if (alg_id == ALG_ID_UCT || alg_id == ALG_ID_FIXED_DEPTH_UCT || alg_id == ALG_ID_PUCT) {
                    alg_params[PARAMS_ID_UCT_BIAS] = UctManagerArgs::USE_AUTO_BIAS;
                    alg_params[PARAMS_ID_UCT_POWER_MEAN_P] = 1.0;
                } else if (alg_id == ALG_ID_EST) {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.1;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = 2.0;
                } else if (alg_id == ALG_ID_DENTS || alg_id == ALG_ID_DBMENTS) {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.1;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = 1.0;
                    if (alg_id == ALG_ID_DENTS) alg_params[PARAMS_ID_DENTS_TEMP] = 1.0;
                } else {
                    double temp = 0.001;
                    double eps = 1.0;
                    if (alg_id == ALG_ID_RENTS) eps = 2.0;
                    alg_params[PARAMS_ID_MENTS_TEMP] = temp;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = eps;
                    alg_params[PARAMS_ID_MENTS_POWER_MEAN_P] = 1.0;
                }
                run_ids->push_back(RunID(env_id, env_instance_id, expr_id, alg_id, alg_params, num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta, rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));
            }
            return run_ids;
        }

        // expr id: S6_095_ER_TEST
        // Evaluates the top-1 tuned ER hyperparameters from 093_s6_er_tune on the held-out
        // S_6_test instance. One fixed config per algorithm -- no grid sweep.
        // TODO: replace placeholder hyperparameters with the winning configs from
        //       `python find_best_hyperparams.py results/sailing_env/6/093_s6_er_tune` once tuning finishes.
        if (expr_id == S6_095_ER_TEST) {
            string env_id = SAILING_ENV_ID;
            string env_instance_id = S_6_TEST_ID;
            int num_trials = 150000;
            int max_trial_length = 50;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 250;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            double default_q_value = -200.0;

            // ER-UCT (placeholder hyperparameters)
            run_ids->push_back(RunID(env_id, env_instance_id, expr_id, ALG_ID_ER_UCT,
                {{PARAMS_ID_UCT_BIAS,         UctManagerArgs::USE_AUTO_BIAS},
                 {PARAMS_ID_UCT_POWER_MEAN_P, 1.0},
                 {PARAMS_ID_UCT_ER_C2,        0.1}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            // ER-FIXED-DEPTH-UCT (placeholder hyperparameters)
            run_ids->push_back(RunID(env_id, env_instance_id, expr_id, ALG_ID_ER_FIXED_DEPTH_UCT,
                {{PARAMS_ID_UCT_BIAS,         UctManagerArgs::USE_AUTO_BIAS},
                 {PARAMS_ID_UCT_POWER_MEAN_P, 4.0},
                 {PARAMS_ID_UCT_ER_C2,        3.0}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            // ER-MENTS (placeholder hyperparameters)
            run_ids->push_back(RunID(env_id, env_instance_id, expr_id, ALG_ID_ER_MENTS,
                {{PARAMS_ID_MENTS_TEMP,             0.01},
                 {PARAMS_ID_MENTS_EPSILON,          2.0},
                 {PARAMS_ID_MENTS_DEFAULT_Q_VALUE,  default_q_value},
                 {PARAMS_ID_UCT_ER_C2,              100.0}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            // ER-RENTS (placeholder hyperparameters)
            run_ids->push_back(RunID(env_id, env_instance_id, expr_id, ALG_ID_ER_RENTS,
                {{PARAMS_ID_MENTS_TEMP,             0.01},
                 {PARAMS_ID_MENTS_EPSILON,          2.0},
                 {PARAMS_ID_MENTS_DEFAULT_Q_VALUE,  default_q_value},
                 {PARAMS_ID_UCT_ER_C2,              0.01}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            // ER-TENTS (placeholder hyperparameters)
            run_ids->push_back(RunID(env_id, env_instance_id, expr_id, ALG_ID_ER_TENTS,
                {{PARAMS_ID_MENTS_TEMP,             0.5},
                 {PARAMS_ID_MENTS_EPSILON,          2.0},
                 {PARAMS_ID_MENTS_DEFAULT_Q_VALUE,  default_q_value},
                 {PARAMS_ID_UCT_ER_C2,              0.3}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            return run_ids;
        }

        // expr id: S6_096_BASELINES_TEST
        // Baselines on the S_6_test held-out instance, sized to match S6_095_ER_TEST
        // for fair apples-to-apples comparison (same num_trials, num_repeats, num_threads).
        // MENTS-family algorithms set default_q_value = -200 to match the S6_094_BASELINES convention.
        if (expr_id == S6_096_BASELINES_TEST) {
            string env_id = SAILING_ENV_ID;
            string env_instance_id = S_6_TEST_ID;
            int num_trials = 150000;
            int max_trial_length = 50;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 250;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            double default_q_value = -200.0;

            vector<string> alg_ids = { ALG_ID_UCT, ALG_ID_FIXED_DEPTH_UCT, ALG_ID_PUCT, ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS, ALG_ID_DENTS, ALG_ID_DBMENTS, ALG_ID_EST };
            for (string alg_id : alg_ids) {
                unordered_map<string,double> alg_params;
                if (alg_id == ALG_ID_UCT || alg_id == ALG_ID_FIXED_DEPTH_UCT || alg_id == ALG_ID_PUCT) {
                    alg_params[PARAMS_ID_UCT_BIAS] = UctManagerArgs::USE_AUTO_BIAS;
                    alg_params[PARAMS_ID_UCT_POWER_MEAN_P] = 1.0;
                } else if (alg_id == ALG_ID_EST) {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.1;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = 2.0;
                    alg_params[PARAMS_ID_MENTS_DEFAULT_Q_VALUE] = default_q_value;
                } else if (alg_id == ALG_ID_DENTS || alg_id == ALG_ID_DBMENTS) {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.1;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = 1.0;
                    alg_params[PARAMS_ID_MENTS_DEFAULT_Q_VALUE] = default_q_value;
                    if (alg_id == ALG_ID_DENTS) alg_params[PARAMS_ID_DENTS_TEMP] = 1.0;
                } else {
                    double temp = 0.001;
                    double eps = 1.0;
                    if (alg_id == ALG_ID_RENTS) eps = 2.0;
                    alg_params[PARAMS_ID_MENTS_TEMP] = temp;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = eps;
                    alg_params[PARAMS_ID_MENTS_DEFAULT_Q_VALUE] = default_q_value;
                }
                run_ids->push_back(RunID(env_id, env_instance_id, expr_id, alg_id, alg_params, num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta, rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));
            }
            return run_ids;
        }

        // expr id: TX5_105_ER_TEST
        // Evaluates the top-1 tuned ER hyperparameters from 103_tx5_er_tune on the held-out
        // TX_5_test instance. One fixed config per algorithm -- no grid sweep.
        // TODO: replace placeholder hyperparameters with the winning configs from
        //       `python find_best_hyperparams.py results/taxi_env/5/103_tx5_er_tune` once tuning finishes.
        if (expr_id == TX5_105_ER_TEST) {
            string env_id = TAXI_ENV_ID;
            string env_instance_id = TX_5_TEST_ID;
            int num_trials = 150000;
            int max_trial_length = 50;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            // ER-UCT (placeholder hyperparameters)
            run_ids->push_back(RunID(env_id, env_instance_id, expr_id, ALG_ID_ER_UCT,
                {{PARAMS_ID_UCT_BIAS,         UctManagerArgs::USE_AUTO_BIAS},
                 {PARAMS_ID_UCT_POWER_MEAN_P, 1.0},
                 {PARAMS_ID_UCT_ER_C2,        1.0}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            // ER-FIXED-DEPTH-UCT (placeholder hyperparameters)
            run_ids->push_back(RunID(env_id, env_instance_id, expr_id, ALG_ID_ER_FIXED_DEPTH_UCT,
                {{PARAMS_ID_UCT_BIAS,         UctManagerArgs::USE_AUTO_BIAS},
                 {PARAMS_ID_UCT_POWER_MEAN_P, 1.0},
                 {PARAMS_ID_UCT_ER_C2,        1.0}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            // ER-MENTS (placeholder hyperparameters)
            run_ids->push_back(RunID(env_id, env_instance_id, expr_id, ALG_ID_ER_MENTS,
                {{PARAMS_ID_MENTS_TEMP,    0.001},
                 {PARAMS_ID_MENTS_EPSILON, 1.0},
                 {PARAMS_ID_UCT_ER_C2,     1.0}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            // ER-RENTS (placeholder hyperparameters)
            run_ids->push_back(RunID(env_id, env_instance_id, expr_id, ALG_ID_ER_RENTS,
                {{PARAMS_ID_MENTS_TEMP,    0.001},
                 {PARAMS_ID_MENTS_EPSILON, 2.0},
                 {PARAMS_ID_UCT_ER_C2,     1.0}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            // ER-TENTS (placeholder hyperparameters)
            run_ids->push_back(RunID(env_id, env_instance_id, expr_id, ALG_ID_ER_TENTS,
                {{PARAMS_ID_MENTS_TEMP,    0.001},
                 {PARAMS_ID_MENTS_EPSILON, 1.0},
                 {PARAMS_ID_UCT_ER_C2,     1.0}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            return run_ids;
        }

        // expr id: TX5_106_BASELINES_TEST
        // Baselines on the TX_5_test held-out instance, sized to match TX5_105_ER_TEST
        // for fair apples-to-apples comparison (same num_trials, num_repeats, num_threads).
        if (expr_id == TX5_106_BASELINES_TEST) {
            string env_id = TAXI_ENV_ID;
            string env_instance_id = TX_5_TEST_ID;
            int num_trials = 150000;
            int max_trial_length = 50;
            int trials_log_delta = 250;
            int mc_eval_trials_delta = 250;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            vector<string> alg_ids = { ALG_ID_UCT, ALG_ID_FIXED_DEPTH_UCT, ALG_ID_PUCT, ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS, ALG_ID_DENTS, ALG_ID_DBMENTS, ALG_ID_EST };
            for (string alg_id : alg_ids) {
                unordered_map<string,double> alg_params;
                if (alg_id == ALG_ID_UCT || alg_id == ALG_ID_FIXED_DEPTH_UCT || alg_id == ALG_ID_PUCT) {
                    alg_params[PARAMS_ID_UCT_BIAS] = UctManagerArgs::USE_AUTO_BIAS;
                    alg_params[PARAMS_ID_UCT_POWER_MEAN_P] = 1.0;
                } else if (alg_id == ALG_ID_EST) {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.1;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = 2.0;
                } else if (alg_id == ALG_ID_DENTS || alg_id == ALG_ID_DBMENTS) {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.1;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = 1.0;
                    if (alg_id == ALG_ID_DENTS) alg_params[PARAMS_ID_DENTS_TEMP] = 1.0;
                } else {
                    double temp = 0.001;
                    double eps = 1.0;
                    if (alg_id == ALG_ID_RENTS) eps = 2.0;
                    alg_params[PARAMS_ID_MENTS_TEMP] = temp;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = eps;
                    alg_params[PARAMS_ID_MENTS_POWER_MEAN_P] = 1.0;
                }
                run_ids->push_back(RunID(env_id, env_instance_id, expr_id, alg_id, alg_params, num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta, rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));
            }
            return run_ids;
        }

        // ============================================================================
        // Deep Sea (bsuite-style hard exploration): random policy succeeds with prob 2^{-(N-1)}.
        // Goal +1 only at all-right path; small cost 0.01/N per right move. Optimal value
        // upper bound is 1 - 0.01*(N-1)/N ≈ 0.99; classical UCT typically stuck near 0.
        // ER's path-level signal keeps the all-right prefix attractive long enough to be
        // reinforced.
        // ============================================================================

        auto run_deep_sea_baselines = [&run_ids, &expr_id](
            const string& env_instance_id, int num_trials, int max_trial_length)
        {
            int trials_log_delta = 100;
            int mc_eval_trials_delta = 100;
            int rollouts_per_mc_eval = 50;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            vector<string> alg_ids = { ALG_ID_UCT, ALG_ID_FIXED_DEPTH_UCT, ALG_ID_PUCT, ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS, ALG_ID_DENTS, ALG_ID_DBMENTS, ALG_ID_EST };
            for (string alg_id : alg_ids) {
                unordered_map<string,double> alg_params;
                if (alg_id == ALG_ID_UCT || alg_id == ALG_ID_FIXED_DEPTH_UCT || alg_id == ALG_ID_PUCT) {
                    alg_params[PARAMS_ID_UCT_BIAS] = UctManagerArgs::USE_AUTO_BIAS;
                    alg_params[PARAMS_ID_UCT_POWER_MEAN_P] = 1.0;
                } else if (alg_id == ALG_ID_EST) {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.1;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = 2.0;
                } else if (alg_id == ALG_ID_DENTS || alg_id == ALG_ID_DBMENTS) {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.1;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = 1.0;
                    if (alg_id == ALG_ID_DENTS) alg_params[PARAMS_ID_DENTS_TEMP] = 1.0;
                } else {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.01;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = (alg_id == ALG_ID_RENTS) ? 2.0 : 1.0;
                    alg_params[PARAMS_ID_MENTS_POWER_MEAN_P] = 1.0;
                }
                run_ids->push_back(RunID(DEEP_SEA_ENV_ID, env_instance_id, expr_id, alg_id, alg_params,
                    num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                    rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));
            }
        };

        auto run_deep_sea_er = [&run_ids, &expr_id](
            const string& env_instance_id, int num_trials, int max_trial_length)
        {
            int trials_log_delta = 100;
            int mc_eval_trials_delta = 100;
            int rollouts_per_mc_eval = 50;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            run_ids->push_back(RunID(DEEP_SEA_ENV_ID, env_instance_id, expr_id, ALG_ID_ER_UCT,
                {{PARAMS_ID_UCT_BIAS,         UctManagerArgs::USE_AUTO_BIAS},
                 {PARAMS_ID_UCT_POWER_MEAN_P, std::numeric_limits<double>::infinity()},
                 {PARAMS_ID_UCT_ER_C2,        1.0}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            run_ids->push_back(RunID(DEEP_SEA_ENV_ID, env_instance_id, expr_id, ALG_ID_ER_FIXED_DEPTH_UCT,
                {{PARAMS_ID_UCT_BIAS,         UctManagerArgs::USE_AUTO_BIAS},
                 {PARAMS_ID_UCT_POWER_MEAN_P, 1.0},
                 {PARAMS_ID_UCT_ER_C2,        50.0}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            run_ids->push_back(RunID(DEEP_SEA_ENV_ID, env_instance_id, expr_id, ALG_ID_ER_MENTS,
                {{PARAMS_ID_MENTS_TEMP,    0.001},
                 {PARAMS_ID_MENTS_EPSILON, 2.0},
                 {PARAMS_ID_UCT_ER_C2,     0.05}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            run_ids->push_back(RunID(DEEP_SEA_ENV_ID, env_instance_id, expr_id, ALG_ID_ER_RENTS,
                {{PARAMS_ID_MENTS_TEMP,    0.001},
                 {PARAMS_ID_MENTS_EPSILON, 0.3},
                 {PARAMS_ID_UCT_ER_C2,     0.3}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            run_ids->push_back(RunID(DEEP_SEA_ENV_ID, env_instance_id, expr_id, ALG_ID_ER_TENTS,
                {{PARAMS_ID_MENTS_TEMP,    0.01},
                 {PARAMS_ID_MENTS_EPSILON, 2.0},
                 {PARAMS_ID_UCT_ER_C2,     0.1}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));
        };

        if (expr_id == DS20_201_BASELINES) {
            run_deep_sea_baselines(DS_20_ID, /*num_trials=*/50000, /*max_trial_length=*/25);
            return run_ids;
        }
        if (expr_id == DS20_202_ER) {
            run_deep_sea_er(DS_20_ID, /*num_trials=*/50000, /*max_trial_length=*/25);
            return run_ids;
        }
        if (expr_id == DS30_203_BASELINES) {
            run_deep_sea_baselines(DS_30_ID, /*num_trials=*/100000, /*max_trial_length=*/35);
            return run_ids;
        }
        if (expr_id == DS30_204_ER) {
            run_deep_sea_er(DS_30_ID, /*num_trials=*/100000, /*max_trial_length=*/35);
            return run_ids;
        }

        // ============================================================================
        // NChain / RiverSwim (greedy-trap exploration benchmark).
        // LEFT at pos=0 gives +0.05 (greedy trap). Goal reward +1 at pos=N-1, reached only
        // by repeated RIGHT moves against the current (advance prob = 0.35). Classical UCT
        // commits to LEFT for its immediate small reward; ER lifts the rarely-visited
        // RIGHT chain enough for the goal to be discovered.
        // ============================================================================

        auto run_nchain_baselines = [&run_ids, &expr_id](
            const string& env_instance_id, int num_trials, int max_trial_length)
        {
            int trials_log_delta = 100;
            int mc_eval_trials_delta = 100;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            vector<string> alg_ids = { ALG_ID_UCT, ALG_ID_FIXED_DEPTH_UCT, ALG_ID_PUCT, ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS, ALG_ID_DENTS, ALG_ID_DBMENTS, ALG_ID_EST };
            for (string alg_id : alg_ids) {
                unordered_map<string,double> alg_params;
                if (alg_id == ALG_ID_UCT || alg_id == ALG_ID_FIXED_DEPTH_UCT || alg_id == ALG_ID_PUCT) {
                    alg_params[PARAMS_ID_UCT_BIAS] = UctManagerArgs::USE_AUTO_BIAS;
                    alg_params[PARAMS_ID_UCT_POWER_MEAN_P] = 1.0;
                } else if (alg_id == ALG_ID_EST) {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.1;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = 2.0;
                } else if (alg_id == ALG_ID_DENTS || alg_id == ALG_ID_DBMENTS) {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.1;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = 1.0;
                    if (alg_id == ALG_ID_DENTS) alg_params[PARAMS_ID_DENTS_TEMP] = 1.0;
                } else {
                    alg_params[PARAMS_ID_MENTS_TEMP] = 0.01;
                    alg_params[PARAMS_ID_MENTS_EPSILON] = (alg_id == ALG_ID_RENTS) ? 2.0 : 1.0;
                    alg_params[PARAMS_ID_MENTS_POWER_MEAN_P] = 1.0;
                }
                run_ids->push_back(RunID(NCHAIN_ENV_ID, env_instance_id, expr_id, alg_id, alg_params,
                    num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                    rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));
            }
        };

        auto run_nchain_er = [&run_ids, &expr_id](
            const string& env_instance_id, int num_trials, int max_trial_length)
        {
            int trials_log_delta = 100;
            int mc_eval_trials_delta = 100;
            int rollouts_per_mc_eval = 100;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            run_ids->push_back(RunID(NCHAIN_ENV_ID, env_instance_id, expr_id, ALG_ID_ER_UCT,
                {{PARAMS_ID_UCT_BIAS,         UctManagerArgs::USE_AUTO_BIAS},
                 {PARAMS_ID_UCT_POWER_MEAN_P, std::numeric_limits<double>::infinity()},
                 {PARAMS_ID_UCT_ER_C2,        1.0}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            run_ids->push_back(RunID(NCHAIN_ENV_ID, env_instance_id, expr_id, ALG_ID_ER_FIXED_DEPTH_UCT,
                {{PARAMS_ID_UCT_BIAS,         UctManagerArgs::USE_AUTO_BIAS},
                 {PARAMS_ID_UCT_POWER_MEAN_P, 1.0},
                 {PARAMS_ID_UCT_ER_C2,        50.0}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            run_ids->push_back(RunID(NCHAIN_ENV_ID, env_instance_id, expr_id, ALG_ID_ER_MENTS,
                {{PARAMS_ID_MENTS_TEMP,    0.001},
                 {PARAMS_ID_MENTS_EPSILON, 2.0},
                 {PARAMS_ID_UCT_ER_C2,     0.05}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            run_ids->push_back(RunID(NCHAIN_ENV_ID, env_instance_id, expr_id, ALG_ID_ER_RENTS,
                {{PARAMS_ID_MENTS_TEMP,    0.001},
                 {PARAMS_ID_MENTS_EPSILON, 0.3},
                 {PARAMS_ID_UCT_ER_C2,     0.3}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));

            run_ids->push_back(RunID(NCHAIN_ENV_ID, env_instance_id, expr_id, ALG_ID_ER_TENTS,
                {{PARAMS_ID_MENTS_TEMP,    0.01},
                 {PARAMS_ID_MENTS_EPSILON, 2.0},
                 {PARAMS_ID_UCT_ER_C2,     0.1}},
                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));
        };

        if (expr_id == NC6_211_BASELINES) {
            run_nchain_baselines(NC_6_ID, /*num_trials=*/30000, /*max_trial_length=*/30);
            return run_ids;
        }
        if (expr_id == NC6_212_ER) {
            run_nchain_er(NC_6_ID, /*num_trials=*/30000, /*max_trial_length=*/30);
            return run_ids;
        }
        if (expr_id == NC10_213_BASELINES) {
            run_nchain_baselines(NC_10_ID, /*num_trials=*/80000, /*max_trial_length=*/60);
            return run_ids;
        }
        if (expr_id == NC10_214_ER) {
            run_nchain_er(NC_10_ID, /*num_trials=*/80000, /*max_trial_length=*/60);
            return run_ids;
        }

        // ============================================================================
        // Hyperparameter search blocks for Deep Sea and NChain.
        //   *_HPS:     sweeps bias for UCT-family (including ER variants at fixed er_c2=1)
        //              and (temp, eps) for MENTS-family baselines, mirroring FL12_051_HPS.
        //   *_ER_TUNE: sweeps the ER hyperparameters (bias, er_c2, power_mean_p) for
        //              ER-UCT/ER-FIXED-DEPTH-UCT and (temp, eps, er_c2) for
        //              ER-MENTS/ER-RENTS/ER-TENTS, mirroring FL12_063_ER_TUNE.
        // ============================================================================

        auto add_hps = [&run_ids, &expr_id](
            const string& env_id,
            const string& env_instance_id,
            int num_trials,
            int max_trial_length,
            int rollouts_per_mc_eval)
        {
            int trials_log_delta = 200;
            int mc_eval_trials_delta = 200;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            // UCT family (incl. ER variants at fixed er_c2 = 1.0)
            vector<string> uct_alg_ids = {ALG_ID_UCT, ALG_ID_ER_UCT, ALG_ID_FIXED_DEPTH_UCT,
                                          ALG_ID_ER_FIXED_DEPTH_UCT, ALG_ID_PUCT};
            vector<double> uct_biases = {UctManagerArgs::USE_AUTO_BIAS, 0.1, 1.0, 10.0};
            for (const string& alg_id : uct_alg_ids) {
                for (double bias : uct_biases) {
                    unordered_map<string,double> alg_params = {{PARAMS_ID_UCT_BIAS, bias}};
                    if (alg_id == ALG_ID_ER_UCT || alg_id == ALG_ID_ER_FIXED_DEPTH_UCT) {
                        alg_params[PARAMS_ID_UCT_ER_C2] = 1.0;
                    }
                    run_ids->push_back(RunID(env_id, env_instance_id, expr_id, alg_id, alg_params,
                        num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                        rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));
                }
            }

            // MENTS family (baselines only; ER variants are tuned in *_ER_TUNE)
            vector<string> ments_alg_ids = {ALG_ID_MENTS, ALG_ID_RENTS, ALG_ID_TENTS,
                                            ALG_ID_DENTS, ALG_ID_DBMENTS, ALG_ID_EST};
            vector<double> temps = {0.001, 0.01, 0.1, 1.0};
            vector<double> epss = {0.1, 0.3, 1.0, 2.0};
            for (const string& alg_id : ments_alg_ids) {
                for (double temp : temps) {
                    for (double eps : epss) {
                        unordered_map<string,double> alg_params = {
                            {PARAMS_ID_MENTS_TEMP, temp},
                            {PARAMS_ID_MENTS_EPSILON, eps}
                        };
                        if (alg_id == ALG_ID_DENTS) {
                            alg_params[PARAMS_ID_DENTS_TEMP] = 1.0;
                        }
                        run_ids->push_back(RunID(env_id, env_instance_id, expr_id, alg_id, alg_params,
                            num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                            rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));
                    }
                }
            }
        };

        auto add_er_tune = [&run_ids, &expr_id](
            const string& env_id,
            const string& env_instance_id,
            int num_trials,
            int max_trial_length,
            int rollouts_per_mc_eval)
        {
            int trials_log_delta = 200;
            int mc_eval_trials_delta = 200;
            int num_repeats = 10;
            int num_threads = 16;
            int eval_threads = 16;

            // ER-UCT family
            vector<string> er_uct_alg_ids = {ALG_ID_ER_UCT, ALG_ID_ER_FIXED_DEPTH_UCT};
            vector<double> uct_biases = {UctManagerArgs::USE_AUTO_BIAS, 1.0, 10.0};
            vector<double> er_c2s = {0.05, 0.1, 1.0, 10.0, 50.0};
            vector<double> power_mean_ps = {1.0, 2.0, std::numeric_limits<double>::infinity()};
            for (const string& alg_id : er_uct_alg_ids) {
                for (double bias : uct_biases) {
                    for (double er_c2 : er_c2s) {
                        for (double pmp : power_mean_ps) {
                            unordered_map<string,double> alg_params = {
                                {PARAMS_ID_UCT_BIAS, bias},
                                {PARAMS_ID_UCT_ER_C2, er_c2},
                                {PARAMS_ID_UCT_POWER_MEAN_P, pmp}
                            };
                            run_ids->push_back(RunID(env_id, env_instance_id, expr_id, alg_id, alg_params,
                                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));
                        }
                    }
                }
            }

            // ER-MENTS family
            vector<string> er_ments_alg_ids = {ALG_ID_ER_MENTS, ALG_ID_ER_RENTS, ALG_ID_ER_TENTS};
            vector<double> temps = {0.001, 0.01, 0.1};
            vector<double> epss = {0.3, 1.0, 2.0};
            vector<double> er_c2s_ments = {0.05, 0.1, 0.5, 5.0};
            for (const string& alg_id : er_ments_alg_ids) {
                for (double temp : temps) {
                    for (double eps : epss) {
                        for (double er_c2 : er_c2s_ments) {
                            unordered_map<string,double> alg_params = {
                                {PARAMS_ID_MENTS_TEMP, temp},
                                {PARAMS_ID_MENTS_EPSILON, eps},
                                {PARAMS_ID_UCT_ER_C2, er_c2}
                            };
                            run_ids->push_back(RunID(env_id, env_instance_id, expr_id, alg_id, alg_params,
                                num_trials, max_trial_length, trials_log_delta, mc_eval_trials_delta,
                                rollouts_per_mc_eval, num_repeats, num_threads, eval_threads));
                        }
                    }
                }
            }
        };

        if (expr_id == DS20_205_HPS) {
            add_hps(DEEP_SEA_ENV_ID, DS_20_ID, /*num_trials=*/50000, /*max_trial_length=*/25, /*rollouts=*/50);
            return run_ids;
        }
        if (expr_id == DS20_206_ER_TUNE) {
            add_er_tune(DEEP_SEA_ENV_ID, DS_20_ID, /*num_trials=*/50000, /*max_trial_length=*/25, /*rollouts=*/50);
            return run_ids;
        }
        if (expr_id == DS30_207_HPS) {
            add_hps(DEEP_SEA_ENV_ID, DS_30_ID, /*num_trials=*/100000, /*max_trial_length=*/35, /*rollouts=*/50);
            return run_ids;
        }
        if (expr_id == DS30_208_ER_TUNE) {
            add_er_tune(DEEP_SEA_ENV_ID, DS_30_ID, /*num_trials=*/100000, /*max_trial_length=*/35, /*rollouts=*/50);
            return run_ids;
        }
        if (expr_id == NC6_215_HPS) {
            add_hps(NCHAIN_ENV_ID, NC_6_ID, /*num_trials=*/30000, /*max_trial_length=*/30, /*rollouts=*/100);
            return run_ids;
        }
        if (expr_id == NC6_216_ER_TUNE) {
            add_er_tune(NCHAIN_ENV_ID, NC_6_ID, /*num_trials=*/30000, /*max_trial_length=*/30, /*rollouts=*/100);
            return run_ids;
        }
        if (expr_id == NC10_217_HPS) {
            add_hps(NCHAIN_ENV_ID, NC_10_ID, /*num_trials=*/80000, /*max_trial_length=*/60, /*rollouts=*/100);
            return run_ids;
        }
        if (expr_id == NC10_218_ER_TUNE) {
            add_er_tune(NCHAIN_ENV_ID, NC_10_ID, /*num_trials=*/80000, /*max_trial_length=*/60, /*rollouts=*/100);
            return run_ids;
        }

        throw runtime_error("Error in get_run_ids_from_expr_id");
    }

}
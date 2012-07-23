#ifndef PARAMS_DEF_HPP
#define PARAMS_DEF_HPP

// The Params structure defines the parameters of the algorithm. This particular way of
// setting them allows the compiler to propagate constants to the whole program
// (i.e.it replaces the parameters in the code by their values), allowing some optimizations.

struct Params {
		struct evo_float {
				// The crossover rate
				// (see: http://en.wikipedia.org/wiki/Crossover_(genetic_algorithm) )
				// Set the crossover rate to 0.0 to disable crossover, or use
				// "static const cross_over_t cross_over_type = no_cross_over"
				static const float cross_rate = 0.0f;//0.5f;
				// the mutation rate of the real-valued vector
				// (see: http://en.wikipedia.org/wiki/Mutation_(genetic_algorithm) )
				static const float mutation_rate = 0.1f;
				// a parameter of the polynomial mutation
				static const float eta_m = 15.0f;
				// a parameter of the polynomial cross-over
				static const float eta_c = 10.0f;
				// we choose the polynomial mutation type
				static const sferes::gen::evo_float::mutation_t mutation_type =
						sferes::gen::evo_float::polynomial;
				// we choose the polynomial cross-over type.
				// SBX stands for simulated binary crossover.
				// replace by ... = no_cross_over; if you want no cross over at all
				// Another solution would be to set cross_rate = 0.0f;
				static const sferes::gen::evo_float::cross_over_t cross_over_type =
						sferes::gen::evo_float::sbx;
		};
		struct pop {
				// size of the population
				// Note: we use the NSGA-II multiobjective evolutionary algorithm,
				//       therefore population size must be divisible by 4.
				// For each generation, size new individuals will be generated
				static const unsigned size = 500; //500
				// number of generations
				static const unsigned nb_gen = 5000;
				// how often should the result file be written (each X generation)?
				static const int dump_period = 1;
				// how many individuals should be created during the random generation process?
				static const int initial_aleat = 4;
				// keep_rate is only used when using rank_simple, i.e. only when
				// HUMPHRIES_MODEL_2006 is defined.
				static const float keep_rate = 0.4f;
				// coeff is only used when using rank_simple, i.e. only when
				// HUMPHRIES_MODEL_2006 is defined.
				static const float coeff = 1.1f;
		};
		struct parameters {
				static const float min = 0.0f;
				static const float max = 1.0f;
				// for weight lpds dnn (see roadsign.cpp)
				static const float min_2 = 0;
				static const float max_2 = 5;
		};
		//  struct nn
		//    {
		//      static const int nb_inputs = 4;
		//      static const int nb_outputs = 4;
		//    };
		struct dnn {
				// number of inputs in the individuals of the initial population:
				// in other words, this is the number of inputs each chip receive.
				// this number remains constant throughout the evolution.
#ifdef EXP_6_VECT
				static const size_t nb_inputs = 3;
#endif
#ifdef EXP_1155_VECT
				static const size_t nb_inputs = 3;
#endif
#ifdef EXP_SURVIVAL
				static const size_t nb_inputs = 4;
				//static const size_t nb_inputs = 8;
#endif
				// number of outputs in the individuals of the initial population:
				// in other words, this is the number of outputs each chip receive.
				// this number remains constant throughout the evolution.
#ifdef EXP_6_VECT
				static const size_t nb_outputs = nb_inputs;
#endif
#ifdef EXP_1155_VECT
				static const size_t nb_outputs = nb_inputs;
#endif
#ifdef EXP_SURVIVAL
				static const size_t nb_outputs = 4;
#endif
				// minimum number of neurons in the individuals of the initial population:
				// the evolution does not take it into account.
				// Should be equal or less than 3, otherwise it'll crash at start (TODO: why?)
				static const size_t min_nb_neurons = 3;
				// maximum number of neurons in 1 chip of the individuals of the initial population:
				// the evolution does not take it into account.
				// The mRF has ca. 0.75 millions neuron in frog, 2 millions in man (p9 McCulloch 1969)
				// This parameter must be at least 1, otherwise sferes won't be happy
				static const size_t max_nb_neurons = 10;
				// minimum number of connections in 1 chip of the individuals of the initial population:
				// the evolution does not take it into account.
				// Usually we take min_nb_conns = number of neurons
				static const size_t min_nb_conns = 0;
				// maximum number of connections in 1 chip of the individuals of the initial population:
				// the evolution does not take it into account.
				// Usually we take min_nb_conns = number of neurons * 2
				static const size_t max_nb_conns = 64;
				//static const weight_t weight_type	= continuous;
				//static const float weight_sigma	= 0.5f; // ??
				//static const float vect_sigma	= 0.25f; // ??
				static const float max_weight = 2.0f;
				static const float max_bias = 2.0f;

				// Constants for mutation of the genotype
				static const float m_rate_add_conn = 0.05f; //1.0f;
				static const float m_rate_del_conn = 0.05f; //1.0f;
				static const float m_rate_change_conn = 3.0f; //1.0f;
				static const float m_rate_add_neuron = 0.05f; //1.0f;
				static const float m_rate_del_neuron = 0.05f; //1.0f;

				static const int io_param_evolving = true; // ??
				static const sferes::gen::dnn::init_t init =
						sferes::gen::dnn::random_topology;

#ifdef EXP_SURVIVAL
				// The convergence threshold is a value under which if
				// the nn's outputs didn't vary more, then it is regarded
				// as having converged.
				// At each behavioral update the salience values are
				// calculated from the state variables and presented
				// at the appropriate outputs of the sensory systems,
				// as described above. The model is then run for
				// "max_nb_time_steps" time-steps or to convergence,
				// whichever is sooner.
				static const float convergence_threshold = 1;//0.001f;
				// The convergence number is the number of the last outputs
				// we analyze to see if there is convergence.
				static const int nb_convergence = 50;
#endif
		};
		// The struct dnn_mrf is specific to this experiment. It contains
		// values which define the global mRF, which is basically
		// a neural network made up with smaller neural networks.
		// (=small-world neural network)
		// interconnection = connection between 2 different chips
		struct dnn_mrf {
				// minimum number of chips in the individuals of the initial population:
				// the evolution does not take it into account.
				// According to literature, the number of chips is between 35 and 75.
				// It should be at least 2 (otherwise disable interconnections)
				static const size_t initial_min_nb_chips = 4;
				// maximum number of chips in the individuals of the initial population:
				// the evolution does not take it into account.
				static const size_t initial_max_nb_chips = initial_min_nb_chips;
				// minimum number of interconnections in the individuals of the initial population:
				// the evolution does not take it into account.
				static const size_t initial_min_nb_interconn = 0;
				// maximum number of interconnections in the individuals of the initial population:
				// the evolution does not take it into account.
				static const size_t initial_max_nb_interconn = 0;
				// minimum number of interconnections during evolution
				static const size_t min_nb_interconn = 0;
				// minimum number of interconnections during evolution
				static const size_t max_nb_interconn = 0;
				// Mutation rate of the interconnections' weight of the mRF
				static const float m_rate_change_interconn = 0.1f;//3.0f;
				// Mutation rate of adding interconnections in the mRF
				static const float m_rate_add_interconn = 0.05f;
				// Mutation rate of deleting interconnections from the mRF
				static const float m_rate_del_interconn = 0.05f;

				// Anatomical constraints:
				// p ->  % of projection neurons (therefore 1 - p is the % of interneurons)
				// (p=0.8 known in literature (p25 Humphries 2007)) ;
				// Note: CONSTRAINT_INHIB_AND_EXCIT_GOOD_RATIO must be defined
				// In the individuals of the initial population
				static const float initial_inhib_ratio = 0.2f;
				// During evolution
				static const float inhib_ratio = 0.2f;
				// P(c) -> probability of each projection neuron contacting a given cluster
				// (P(c) = 0.25 p25 Humphries 2007 known in literature) ;
				static const float interconn_other_chip_proba = 0.25f;
		};
		struct fit {
				// Number of iterations of for the propagation of the inputs in the
				// neural network phenotype (here this is the mRF).
				// Humphries 2005b p8 took the value 30, but his network was smaller than ours.
				static const size_t nb_steps = 100;
				// Number of iterations of objectives for the fitness function. Should be 1, 2 or 3.
#ifdef ANAT_CONSTRAINTS_DURING_FITNESS
				static const size_t nb_objectives = 3;
#else
				static const size_t nb_objectives = 2;
#endif
#ifdef EXP_SURVIVAL
				// Maximum number of steps of the survival task evaluation.
				// Humphries 2005b p10 took the value 3000.
				static const size_t max_nb_time_steps = 3000;
				// Number of lives the robot will be tested.
				// Humphries 2005b p8 took the value 20.
				static const size_t nb_lives = 1;
				// Number of lives the robot will be tested if we check its generalization.
				static const size_t nb_lives_generalization = 100;
				// Minimum average survival time to check generalization.
				// Set it to max_nb_time_steps + 1 if you want to disable the
				// generalization check
				// static const float min_score_for_generalization = max_nb_time_steps + 10;
				static const float min_score_for_generalization = 2000;
				// Type of controller for the robot
				// 0 is mRF ; 1 is WTA ; 2 is random ; 3 is mRF Humphries model
				// WARNING: if you choose controller_type = 3, you MUST
				// define HUMPHRIES_MODEL_2006, otherwise you musn't define it.
				static const int controller_type = 0;
//#define HUMPHRIES_MODEL_2006
				// selection_type defines how we'll selection actions
				// 0 is unique action (only if controller_type == {0, 1, 2} )
				// 1 is multiple actions (only if controller_type == 3)
				static const int selection_type = 0;
				// Type of pre-computation for the inputs of the mRF
				// 0 is computation of saliences (Humphries 2005a p7)
				// 1 is direct internal/external data (Humphries 2006 p5)
				// WARNING: set Params::dnn:nb_inputs accordingly!
				// input_format = 0 --> Params::dnn:nb_inputs = 4
				// input_format = 1 --> Params::dnn:nb_inputs = 8
				static const int input_format = 0;
				// The output vector c of the mRF computational model is converted to a
				// spinal-command vector by m_k = M(c_k), where M is one of the output transfer
				// functions, "dual" or "triangle" (Humphries 2006 p7)
				// 0 is contrast isn't taken into account, actions are always fully activated
				// 1 is actions are activated proportionally to the contrast
				// 2 is actions are activated proportionally to the contrast square root
				// WARNING: if input_format != 0, the only possible choice is output_transfer_function = 0
				static const int output_transfer_function = 0;
				// The action_selection_type parameter defines
				// 0 is one action is always selected
				// 1 is one action is selected if and only if 1 action's output is least
				//      0.8 and the others less than 0.2
				static const int action_selection_type = 0;
				// dumm_actions_forbidden allows you to kill the robot prematurely if
				// he makes dummy decisions (e.g. selection to reload E whereas he's
				// not on a light area, since he can't reload E there)
				// 0 is we never kill the robot
				// 1 is we kill the robot if he makes dummy decisions
				static const int dumm_actions_forbidden = 0;
#endif
		};
#ifdef EXP_SURVIVAL
		struct simu {
			SFERES_STRING(map_name, "../src/exp/arena_square.pbm");
		};
#endif
};

#endif

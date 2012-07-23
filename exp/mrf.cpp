/******** mRF Model - main file *********
 * This program needs the following library:
 * Boost (Sferes Core requires Boost)
 * Eigen v2 (Sferes Module nn2 requires Eigen v2)
 * Sferes Core
 * Sferes Module evoneuro (LPDS)
 * Sferes Module nn2 (DNN)
 * TBB (http://threadingbuildingblocks.org/ - Multi-threading library)
 ****************************************/

#ifndef MRF_CPP
#define MRF_CPP

/******** TO BE CONFIGURED before each compilation! *********/
// Define the log folder. Choose only one among the following.
// expXX corresponds to the folder name in the cluster of the mRF program.
// #define FOLDER_NAME "contr_3obj_1155" // exp01
// #define FOLDER_NAME "contr_2obj_1155" // exp02
// #define FOLDER_NAME "nocontr_3obj_1155" // exp03
// #define FOLDER_NAME "nocontr_2obj_1155" // exp04
//#define FOLDER_NAME "rob_mrf_5m_5fwd_rnd_dirL" // exp50
//#define FOLDER_NAME "rob_mrf_1m_5fwd_rnd_dirLC"
//#define FOLDER_NAME "rob_nocontr2c_1m_5fwd_rndL"
//#define FOLDER_NAME "rob_nocontr2c_20m_5fwd_rndL"
//#define FOLDER_NAME "test_generalisation_nocontr2"
//#define FOLDER_NAME "rob_wta_5m_5fwd_rndLC"
//#define FOLDER_NAME "rob_mrf_1m_5fwd_rndLC"
#define FOLDER_NAME "rob_mrf_1m_100m_5fwd_rndLC"
//#define FOLDER_NAME "rob_mrf_1m_5fwd_rndLCT"
//#define FOLDER_NAME "rob_mrf_1m_5fwd_rndLCTh"
//#define FOLDER_NAME "rob_nocontr2c_1m_5fwd_Th"
//#define FOLDER_NAME "rob_rnd_1m_5fwd_rndLC"
//#define FOLDER_NAME "rob_rnd_5m_5fwd_rndLC"
//#define FOLDER_NAME "rob_wta_1m_5fwd_rndLC"
//#define FOLDER_NAME "rob_wta_5m_5fwd"
//#define FOLDER_NAME "rob_rnd_5m_5fwd_rndL" // Don't forget to comment the line "selected_action = rest;"
// Link foldername with ANAT_CONSTRAINTS_DURING_EVOLUTION and ANAT_CONSTRAINTS_DURING_FITNESS

// Choose the task
//#define EXP_6_VECT
//#define EXP_1155_VECT
#define EXP_SURVIVAL

// To activate constraints on the mRF during evolution
#define ANAT_CONSTRAINTS_DURING_EVOLUTION

// To activate constraints on the mRF during evaluation (fitness).
#define ANAT_CONSTRAINTS_DURING_FITNESS

// To shuffle the IO vectors all_inputs_variables, outputs variables, as well
// as compute_vector_contrast.
#define SHUFFLE_RANDOM_IO_VECTOR

#ifdef EXP_SURVIVAL
// If you define/undefine USE_SDL, you must do the same in modules/fastsim/display.hpp
//#define USE_SDL
#endif

/************************************************/

// nn2 module needs Eigen2
#define EIGEN2_ENABLED
// If you comment #define NO_PARALLEL, the program will be multi-threaded.
// NOTE: there seems to be a bug when multi-threading is activated (crash after
// a random number of generations, but we didn't bother debugging it since many
// runs will be launched concurrently.
#define NO_PARALLEL

/******** Anatomical constraints for the mRF *********/
// A connection must link an inhibitory neuron with an excitatory neuron
// if the connection is within a chip (i.e. interconnections can be between
// 2 neurons of the same type)
// Note: this constraint doesn't seem to be right : "They make connections
//       with both other interneurons and projection neurons within their
//       cluster. (p3 Humphries 2005b).
// #define CONSTRAINT_EDGE_ALWAYS_INHIB_AND_EXCIT

#ifdef ANAT_CONSTRAINTS_DURING_EVOLUTION
// Respect "projection neurons make no connections within their own
// clusters." (p3 Humphries 2005b).
// Note: you can't define CONSTRAINT_EDGE_ALWAYS_INHIB_AND_EXCIT
//       and CONSTRAINT_PROJECTIONS_NEVER_INTRA_CHIP at the same time

#define CONSTRAINT_PROJECTIONS_NEVER_INTRA_CHIP //c
// Respect ratio of inhibitory/excitatory neurons within a chip
#define CONSTRAINT_INHIB_AND_EXCIT_GOOD_RATIO //c
// Respect probability of each projection neuron contacting a
// given cluster (P(c) = 0.25 p25 Humphries 2007 known in literature) ;
#define CONSTRAINT_PROJECTION_INTERCONNECTIONS //c
//  Delete all isolated neurons (except input/output)
#define DEL_ISOLATED_NEURONS

#endif

// The output neurons are projection neurons by definition, therefore
// they must be excitatory.
#define OUTPUT_ARE_EXCIT

// Fix LPDS neuron's tau at value. If not defined, tau can have several values,
// see in lpds.hpp, SFERES_ARRAY(float, tau, value 1, value 2, ..., value n);
#define LPDS_FIXED_TAU

//  Fix LPDS neuron's LPDS threshold at value. If not defined, threshold can have several values,
// see in lpds.hpp, SFERES_ARRAY(float, threshold, value 1, value 2, ..., value n);
#define LPDS_FIXED_THRESHOLD
/*******************************************************/

// STL includes
#include <algorithm> // std::max_element
#include <iostream>
#include <math.h> // pow, sqrt
#include <limits> // std::numeric_limits<int>::max()
// External libraries includes
#include <boost/program_options.hpp>

// Sferes core includes
#include <sferes/phen/parameters.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/ea/nsga2.hpp>
#include <sferes/eval/eval.hpp>
#include <sferes/stat/pareto_front.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/run.hpp>

// Sferes modules includes
#include <modules/nn2/gen_dnn.hpp>
#include <modules/nn2/phen_dnn.hpp>

#ifdef EXP_SURVIVAL
#include <sferes/fit/fitness_simu.hpp>
//#include <sferes/simu/simu.hpp>
#include <modules/fastsim/fastsim.hpp>
#endif

// Defines the Params structure, which defines the parameters of the algorithm.
#include "params_def.hpp"

// This program's own files includes
#include "genotype.hpp"
#include "mrf_humphries2006.hpp"
#include "phenotype.hpp"
#include "stat.hpp"

using namespace sferes;

// Debug function: display a vector
template<typename t>
void display_io_vector(std::vector<t> vector_to_display, bool stop = false,
		int new_line = 0) {
	BOOST_FOREACH(t value, vector_to_display)
				{
					std::cout << value << " ; ";
				}
	if (stop)
		std::cin.get();

	for (int i = 0; i < new_line; ++i) {
		std::cout << std::endl;
	}
}

// Store random input-outputs pairs, which will be learned by the mRF
// We put these pairs in global variables so as to speed up the execution.
// It's initialized by init_random_io() or init_random_io2()
std::vector<std::vector<float> > all_inputs_variables; //io_t is float
std::vector<std::vector<float> > all_outputs_variables;
// Store random input-outputs pairs contrast
// It's initialized by compute_all_input_best_contrast()
std::vector<float> all_inputs_variables_contrast;
// Best contrast possible, i.e. contrast between all_inputs_variables
// and all_outputs_variables
// It's initialized by compute_all_input_best_contrast()
float contrast_best_possible_average;

// Compute a vector contrast
float compute_vector_contrast(std::vector<float> vect) {

	float constrast = 0;

	std::vector<float>::iterator max_value_it = std::max_element(vect.begin(),
			vect.end());
	int max_position = std::distance(vect.begin(), max_value_it);

	//display_io_vector(vect);
	//std::cout <<"max_position"<<max_position<<";";
	//std::cout <<"vect[max_position]="<<vect[max_position]<<";";
	BOOST_FOREACH(float value, vect)
				{
					//std::cout << "output_vector_average[max_position]=" << output_vector_average[max_position] << ";";
					//std::cout << "value=" << value << ";";
					assert(vect[max_position] - value >= 0);
					constrast += sferes::misc::put_in_range(
							pow(vect[max_position] - value, 2), 0.0f, 1.0f);
					//std::cout <<"constrastT="<<constrast<<";";
				}

	if (vect.size() <= 1) {
		constrast = 0;
	} else {
		constrast /= (vect.size() - 1);
	}
	assert(constrast >= 0 && constrast <= (vect.size() - 1));
	constrast = sqrt(constrast);

	//std::cout <<"constrast="<<constrast<<";";
	//std::cout <<"vect[max_position]="<<vect[max_position]<<";";
	//std::cout <<"max_position"<<max_position<<";";
	//std::cout << std::endl;

	return constrast;
}

// Compute all vector contrast
void compute_all_input_contrasts() {
	BOOST_FOREACH(std::vector<float> input, all_inputs_variables)
				{
					std::vector<float> temp_vect(Params::dnn::nb_inputs);
					// We only compute the contrast of the output of one chip (since expected output
					// is the same for all chips)
					copy(input.begin(), input.begin() + Params::dnn::nb_inputs,
							temp_vect.begin());
					std::cout << "temp_vect.size()=" << temp_vect.size();
					all_inputs_variables_contrast.push_back(
							compute_vector_contrast(temp_vect));
				}
}

// Compute the best contrast average possible
void compute_all_input_best_contrast() {
	int count = 0;
	contrast_best_possible_average = 0;
	BOOST_FOREACH(float value, all_inputs_variables_contrast)
				{

					display_io_vector(all_outputs_variables[count]);
					std::cout << "compute_vector_contrast(all_outputs_variables[count])="
							<< compute_vector_contrast(all_outputs_variables[count]) << ";";
					std::cout << "value=" << value << ";";
					std::vector<float> temp_vect(Params::dnn::nb_outputs);
					copy(all_outputs_variables[count].begin(),
							all_outputs_variables[count].begin() + Params::dnn::nb_inputs,
							temp_vect.begin());
					if (value > 0) {
						contrast_best_possible_average += (compute_vector_contrast(
								temp_vect) / value);
					}
					count++;
				}
	std::cout << "contrast_best_possible_average="
			<< contrast_best_possible_average << ";";
	contrast_best_possible_average /= all_inputs_variables_contrast.size();
	std::cout << "contrast_best_possible_average="
			<< contrast_best_possible_average << ";";
}

// Generate Params::dnn::nb_inputs * 2 random input-outputs pairs, *
// which will be learned by the mRF
void init_random_io() {
	typedef float io_t;
	for (size_t i = 0; i < Params::dnn::nb_inputs * 2; ++i) {
		// Initialize vectors
		std::vector<io_t> input_variables(Params::dnn::nb_inputs);
		std::vector<io_t> output_variables(Params::dnn::nb_outputs);
		std::fill(input_variables.begin(), input_variables.end(), 0.0f);
		std::fill(output_variables.begin(), output_variables.end(), 0.0f);
		input_variables[i % Params::dnn::nb_inputs] = 1;
		output_variables[i % Params::dnn::nb_outputs] = 1;
		if (i >= Params::dnn::nb_inputs) {
			input_variables[(i + 1) % Params::dnn::nb_inputs] = 0.5f;
			output_variables[(i + 1) % Params::dnn::nb_outputs] = 0.0f;
		}
		// Be careful with chip_number. We won't be able to change it during evolution
		// if it's used like this.
		size_t nb_chips = Params::dnn_mrf::initial_min_nb_chips;
		// Duplicate vectors so that each chip receive the same information
		for (size_t chip_number = 1; chip_number < nb_chips; ++chip_number) {
			for (size_t j = 0; j < Params::dnn::nb_inputs; ++j) {
				input_variables.push_back(input_variables[j]);
			}
			for (size_t j = 0; j < Params::dnn::nb_outputs; ++j) {
				output_variables.push_back(output_variables[j]);
			}
		}
		//
		all_inputs_variables.push_back(input_variables);
		all_outputs_variables.push_back(output_variables);
	}
	assert(all_inputs_variables.size() == all_outputs_variables.size());
	compute_all_input_contrasts();
	compute_all_input_best_contrast();
}

// Humphries 2007 p15
void init_random_io2() {
	typedef float io_t;
	//float interval = 0.2f;
	int number_of_values = 11;
	int number_of_pairs = pow(number_of_values, Params::dnn::nb_inputs);
	std::vector<io_t> current_value;
	std::fill(current_value.begin(), current_value.end(), 0.0f);
	for (int i = 0; i < number_of_pairs; ++i) {
		// Initialize new vectors
		std::vector<io_t> input_variables(Params::dnn::nb_inputs);
		std::vector<io_t> output_variables(Params::dnn::nb_outputs);
		std::fill(input_variables.begin(), input_variables.end(), 0.0f);
		std::fill(output_variables.begin(), output_variables.end(), 0.0f);

		for (size_t cursor = 0; cursor < Params::dnn::nb_inputs; ++cursor) {
			input_variables[cursor] = ((float) ((int) (floor(
					i / pow(number_of_values, cursor))) % number_of_values))
					/ (number_of_values - 1);
		}

		float max_input = *(std::max_element(input_variables.begin(),
				input_variables.end()));
		int nb_of_max = 0;
		int max_position = -1;
		int count_position = 0;
		BOOST_FOREACH(io_t value, input_variables)
					{
						if (value == max_input) {
							max_position = count_position;
							nb_of_max++;
						}
						count_position++;
					}

		// We want to have only one max !
		if (nb_of_max > 1)
			continue;

		// Now we compute the output
		assert(max_input >= 0 && max_input <= 1);
		assert(max_position >= 0 && max_position < output_variables.size());
		output_variables[max_position] = 1;

		std::cout << std::endl;
		BOOST_FOREACH(io_t value, input_variables)
					{
						std::cout << value << ";";
					}
		std::cout << "nb_of_max=" << nb_of_max << ";";

		BOOST_FOREACH(io_t value, output_variables)
					{
						std::cout << value << ";";
					}
		//std::cin.get();

		// Be careful with chip_number. We won't be able to change it during evolution
		// if it's used like this.
		size_t nb_chips = Params::dnn_mrf::initial_min_nb_chips;
		// Duplicate vectors so that each chip receive the same information
		for (size_t chip_number = 1; chip_number < nb_chips; ++chip_number) {
			for (size_t j = 0; j < Params::dnn::nb_inputs; ++j) {
				input_variables.push_back(input_variables[j]);
			}
			for (size_t j = 0; j < Params::dnn::nb_outputs; ++j) {
				output_variables.push_back(output_variables[j]);
			}
		}
		//
		all_inputs_variables.push_back(input_variables);
		all_outputs_variables.push_back(output_variables);
	}
	assert(all_inputs_variables.size() == all_outputs_variables.size());
	std::cout << all_inputs_variables.size();
	compute_all_input_contrasts();
	compute_all_input_best_contrast();
}

// To shuffle the IO vectors all_inputs_variables, outputs variables, as well
// as compute_vector_contrast.
void shuffle_io_vectors() {

}
// This is the fitness function. It's a special class with an "eval()" function which
// derives from fit::Fitness. It has to fill this->_value in single-objective optimization
// and this->_objs in multiobjective optimization.
// A fitness is a "normal" class and consequently we can add other methods
// or attributes to suit our needs.
// Note that evolutionary algorithms maximize the fitness (whereas most
// optimization algorithms minimize the cost function).
int counter_individuals = 0;
SFERES_FITNESS(MrfFit, sferes::fit::Fitness){
public:

// These typedefs could be avoided by using auto of C++0x
		// Yet, TBB have trouble with C++0x.
		// Error in  tbb_exception.h -> ‘exception_ptr’ in namespace ‘std’
		//                               does not name a type.
		// This issue can be solved by commenting the class in tbb_exception.h
		// class tbb_exception_ptr {} -> /*class tbb_exception_ptr {} */
		// but this solution makes TBB non standard...
		typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>, Params>
		weight_t;
		typedef phen::Parameters<gen::EvoFloat<4, Params>, fit::FitDummy<>, Params>
		node_label_t;
		typedef nn::PfWSum<weight_t> potential_t;
		typedef nn::AfLpds<node_label_t> activation_t;
		typedef nn::Neuron<potential_t, activation_t> neuron_t;
		typedef nn::Connection<weight_t> connection_t;
		typedef nn::NN<neuron_t, connection_t> nn_t;
		typedef gen::Dnn<neuron_t, connection_t, Params> genotype_t;
		typedef typename nn_t::vertex_desc_t vertex_desc_t;
		typedef typename nn_t::edge_desc_t edge_desc_t;

#ifdef EXP_SURVIVAL
		//typedef sferes::simu::Fastsim<Params> simu_t;

		// typedef simu::Fastsim<Params> simu_t;
		//typedef fastsim::simu::Fastsim<Params> simu_t3;
#endif

		MrfFit() {
		}

		typedef float io_t;

		// anatomical_coherence() computes the anatomical plausibility score
		// of an individual.
		// Setting ignore_useless_neurons to true will compute the anatomical
		// plausibility without taking into account the useless neurons, which
		// may prevent the genetic algorithm from cheating.
		template<typename Indiv>
		float anatomical_plausibility(Indiv& ind,
				bool ignore_useless_neurons = false) {
			// If c++0x were enabled, we'd use auto and typeof, so as not to
			// rely on typedef defined in the class MrfFit
			//auto graph_src = ind.nn_mrf().get_graph();

			// “The interneurons are assumed to project only within their parent
			// cluster, due to their limited projection radius (1 mm)”
			// Not implemented because there interconnections were implemented so
			// that only projection neurons can be interconnections
			size_t nb_interneuron_not_within_chip = 0;

			// Roughly 45% of the synapses on a projection neuron are GABAergic
			// (i.e. inhibitory) (p14 Humphries 2007).
			size_t nb_inhib_synapses_on_projection = 0;
			size_t nb_excit_synapses_on_projection = 0;
			float proba_gabaergic_projection = 0.0f;

			// “projection neurons make no connections within their own clusters.”
			// (p3 Humphries 2005b).
			// You can use #define CONSTRAINT_PROJECTIONS_NEVER_INTRA_CHIP to
			// make sure mRF always respect this constraint.
			size_t nb_excit = 0;
			size_t nb_inhib = 0;
			size_t nb_projection_within_chip = 0;
			// float proba_projection_within_chip = 0.0f;

			size_t nb_clusters = ind.nb_chips();

			// Respect inhibitory/excitatory neurons ratio within a chip.
			// There should be ca 80% of excitatory neurons (=projection neurons),
			//                    20% of inhibitory neurons (=interneurons)
			float ratio_inhib = ind.nn_mrf().ratio_inhib(ignore_useless_neurons);

			// P(c) -> probability of each projection neuron contacting a given
			// cluster. There are two models known in literature:
			// * Model 1: P(c) = 0.25 .(p4 Humphries 2005b: "Data from Grantyn
			//            et al. (1987)	suggest a spatially uniform model for
			//            which we assigned P(c) = 0.25 for all clusters."
			// * Model 2: P(c) = d^(-a) .(p4 Humphries 2005b:In contrast, McCulloch
			//            and colleague’s RF model (Kilmer et al. 1969) used a
			//            distance-dependent distribution (William Kilmer, personal
			//            communication),	typical of models of neural connectivity
			//            (Hellwig 2000). Thus, if there are d intervening clusters
			//            between the projection neuron node and the target cluster
			//            (so for adjacency dZ1), then P(c) = d^(-a); we use a = 1
			//            throughout.
			float proba_c = 0.0f;

			// P(p) -> probability of the projection neuron forming a connection
			// with any given neuron in that cluster. P(p) is unknown in literature,
			// but likely to be low: Humphries 2007 p28: " P(l) = P(p) = 0.1,
			// as arbitrarily chosen neuron pairs are likely to have low connection
			// probabilities (Schuz, 1995);".)
			float proba_p = 0.0f;

			// P(l) -> probability of an inter-neuron forming a connection with any
			// other given neuron in its own cluster. P(l) is unknown in literature,
			// but likely to be low: Humphries 2007 p28: " P(l) = P(p) = 0.1,
			// as arbitrarily chosen neuron pairs are likely to have low connection
			// probabilities (Schuz, 1995);".)
			// Note: we've proved that P(l) > 45*P(p) (see e-mails)
			float proba_l = 0.0f;

			typename nn_t::graph_t graph_src = ind.nn_mrf().get_graph();

			// Compute data
			int nb_useless_neurons = 0;
			BGL_FORALL_VERTICES_T(vertex, graph_src, typename nn_t::graph_t)
			{
				if (graph_src[vertex].useless()) {
					nb_useless_neurons++;
				}
				if (ignore_useless_neurons && graph_src[vertex].useless()) {
					continue;
				}
				if (!graph_src[vertex].is_input()) { // Don't take input into account
					graph_src[vertex].inhib() ? nb_inhib++ : nb_excit++;
				}
			}

			// projection_chip_map stores on which chips a projection neuron projects to.
			std::map<vertex_desc_t, std::vector<int> > projection_chip_map;
			// projection_chip_map stores on which chips a projection neuron projects to.
			std::map<vertex_desc_t, int> inhib_within_chip_map;

			int nb_useless_connections = 0;
			BGL_FORALL_EDGES_T(edge, graph_src, typename nn_t::graph_t)
			{
				// A useless connection is a connection whose target is a useless
				// neuron.
				if (graph_src[target(edge, graph_src)].useless()) {
					nb_useless_connections++;
				}

				if (ignore_useless_neurons
						&& (graph_src[source(edge, graph_src)].useless()
								|| graph_src[target(edge, graph_src)].useless())) {
					continue;
				}

				if (graph_src[target(edge, graph_src)].inhib()) {
					// TODO: exclude inputs from stat?
					graph_src[source(edge, graph_src)].inhib() ? nb_inhib_synapses_on_projection++
					: nb_excit_synapses_on_projection++;
				}

				if (!graph_src[source(edge, graph_src)].is_input()
						&& graph_src[source(edge, graph_src)].inhib()
						&& (graph_src[source(edge, graph_src)].numchip()
								!= graph_src[target(edge, graph_src)].numchip())) {
					nb_interneuron_not_within_chip++;
				}

				if (!graph_src[source(edge, graph_src)].is_input()
						&& !graph_src[source(edge, graph_src)].inhib()
						&& (graph_src[source(edge, graph_src)].numchip()
								== graph_src[target(edge, graph_src)].numchip())) {
					nb_projection_within_chip++;
				}

				if (!graph_src[source(edge, graph_src)].inhib() && (graph_src[source(
										edge, graph_src)].numchip()
								!= graph_src[target(edge, graph_src)].numchip())) {
					// If this is the first we encounter this projection neuron,
					// construct the std::vector<bool> and resize it
					if (projection_chip_map.count(source(edge, graph_src)) <= 0) {
						projection_chip_map[source(edge, graph_src)].resize(nb_clusters);
					}
					projection_chip_map[source(edge, graph_src)][graph_src[target(edge,
							graph_src)].numchip()]++;
				}

				if (graph_src[source(edge, graph_src)].inhib() && (graph_src[source(
										edge, graph_src)].numchip()
								== graph_src[target(edge, graph_src)].numchip())) {
					// If this is the first we encounter this interneuron,
					// construct the std::vector<bool> and resize it
					if (projection_chip_map.count(source(edge, graph_src)) <= 0) {
						inhib_within_chip_map[source(edge, graph_src)] = 0;
					}
					inhib_within_chip_map[source(edge, graph_src)]++;
				}

			}

			/// Compute scores
			// proba_gabaergic_projection_score
			float proba_gabaergic_projection_score = 0.0f;
			if ((nb_inhib_synapses_on_projection + nb_excit_synapses_on_projection)
					<= 0) {
				//std::cout << "nb_inhib_synapses_on_projection=" << nb_inhib_synapses_on_projection << ";";
				//std::cout << "nb_excit_synapses_on_projection=" << nb_excit_synapses_on_projection << ";";
				return -100.0f;
			}
			if ((nb_inhib_synapses_on_projection + nb_excit_synapses_on_projection)
					== 0) {
				proba_gabaergic_projection = 0;
			} else {
				proba_gabaergic_projection = ((float) nb_inhib_synapses_on_projection)
				/ ((float) (nb_inhib_synapses_on_projection
								+ nb_excit_synapses_on_projection));
			}

			proba_gabaergic_projection_score = (-1) * pow(
					(0.45 - proba_gabaergic_projection), 2);

			// ratio_inhib_score
			float ratio_inhib_score = (-1) * pow((0.2 - ratio_inhib), 2);

			// proba_c_score and proba_p
			typename std::map<vertex_desc_t, std::vector<int> >::iterator it;
			std::vector<int>::iterator it_vector;
			proba_c = 0.0f;
			proba_p = 0.0f;
			int nb_projection_chip = 0;
			float sum_proba_p = 0.0f;
			for (it = projection_chip_map.begin(); it != projection_chip_map.end(); it++) {
				float proba_c_neuron = 0.0f;
				int chip_number = 0;
				for (it_vector = (*it).second.begin(); it_vector != (*it).second.end(); it_vector++) {
					if (*it_vector >= 1) { // If the projection neuron project on this chip
						// proba_c
						proba_c_neuron = proba_c_neuron + 1;
						// proba_p
						nb_projection_chip++;
						assert(
								ind.chips_gen(chip_number).get_nb_neurons() - ind.chips_gen(
										chip_number).get_nb_inputs() > 0);
						sum_proba_p += *it_vector
						/ ((float) (ind.chips_gen(chip_number).get_nb_neurons()
										- ind.chips_gen(chip_number).get_nb_inputs()));
						//std::cout << sum_proba_p << ";";
					}
					chip_number++;
				}
				proba_c_neuron /= (*it).second.size();
				proba_c += proba_c_neuron;
			}

			if (projection_chip_map.size() == 0) {
				proba_c = 0.0f;
				proba_p = 0.0f;
			} else {
				proba_c /= projection_chip_map.size();
				if (nb_projection_chip == 0) {
					proba_p = 0.0f;
				} else {
					proba_p = sum_proba_p / ((float) nb_projection_chip);
				}
			}

			float proba_c_score = (-1) * pow((0.25 - proba_c), 2);

			// proba_l
			proba_l = 0.0f;
			typename std::map<vertex_desc_t, int>::iterator it2;
			for (it2 = inhib_within_chip_map.begin(); it2
					!= inhib_within_chip_map.end(); it2++) {
				assert(
						ind.chips_gen((graph_src[(*it2).first].numchip())).get_nb_neurons()
						- ind.chips_gen((graph_src[(*it2).first].numchip())).get_nb_inputs()
						> 0);
				proba_l
				+= ((float) ((*it2).second))
				/ (ind.chips_gen((graph_src[(*it2).first].numchip())).get_nb_neurons()
						- ind.chips_gen((graph_src[(*it2).first].numchip())).get_nb_inputs());
			}

			if (inhib_within_chip_map.size() == 0) {
				proba_l = 0.0f;
			} else {
				proba_l /= inhib_within_chip_map.size();
			}

			//std::cout << "SCORES:";
			//std::cout << proba_gabaergic_projection_score << ";";
			//std::cout << nb_interneuron_not_within_chip << ";";
			//std::cout << nb_projection_within_chip << ";";
			//std::cout << nb_clusters << ";";
			//std::cout << proba_c_score << ";";
			//std::cout << proba_p << ";";
			//std::cout << proba_l << ";";


			// Save all data stats
			if (!ignore_useless_neurons) {
				ind.set_anatomical_stats("proba_gabaergic_projection",
						proba_gabaergic_projection);
				ind.set_anatomical_stats("nb_interneuron_not_within_chip",
						nb_interneuron_not_within_chip);
				ind.set_anatomical_stats("nb_projection_within_chip",
						nb_projection_within_chip);
				ind.set_anatomical_stats("nb_neurons", ind.nn_mrf().get_nb_neurons());
				ind.set_anatomical_stats("nb_useless_neurons", nb_useless_neurons);
				ind.set_anatomical_stats("nb_connections",
						ind.nn_mrf().get_nb_connections());
				ind.set_anatomical_stats("nb_useless_connections",
						nb_useless_connections);
				ind.set_anatomical_stats("nb_clusters", nb_clusters);
				ind.set_anatomical_stats("ratio_inhib", ratio_inhib);
				ind.set_anatomical_stats("proba_c", proba_c);
				ind.set_anatomical_stats("proba_p", proba_p);
				ind.set_anatomical_stats("proba_l", proba_l);
			} else {
				ind.set_anatomical_real_stats("proba_gabaergic_projection",
						proba_gabaergic_projection);
				ind.set_anatomical_real_stats("nb_interneuron_not_within_chip",
						nb_interneuron_not_within_chip);
				ind.set_anatomical_real_stats("nb_projection_within_chip",
						nb_projection_within_chip);
				ind.set_anatomical_real_stats("nb_neurons",
						ind.nn_mrf().get_nb_neurons_no_useless());
				ind.set_anatomical_real_stats("nb_useless_neurons", nb_useless_neurons);
				ind.set_anatomical_real_stats("nb_connections",
						ind.nn_mrf().get_nb_connections_no_useless());
				ind.set_anatomical_real_stats("nb_useless_connections",
						nb_useless_connections);
				ind.set_anatomical_real_stats("nb_clusters", nb_clusters);
				ind.set_anatomical_real_stats("ratio_inhib", ratio_inhib);
				ind.set_anatomical_real_stats("proba_c", proba_c);
				ind.set_anatomical_real_stats("proba_p", proba_p);
				ind.set_anatomical_real_stats("proba_l", proba_l);
			}

			// Save all plausibility scores
			if (!ignore_useless_neurons) {
				ind.set_anatomical_plausibility_scores(
						"proba_gabaergic_projection_score",
						proba_gabaergic_projection_score);
				ind.set_anatomical_plausibility_scores(
						"nb_interneuron_not_within_chip", nb_interneuron_not_within_chip);
				ind.set_anatomical_plausibility_scores("nb_projection_within_chip",
						nb_projection_within_chip);
				ind.set_anatomical_plausibility_scores("nb_clusters", nb_clusters);
				ind.set_anatomical_plausibility_scores("ratio_inhib_score",
						ratio_inhib_score);
				ind.set_anatomical_plausibility_scores("proba_c_score", proba_c_score);
			} else {
				ind.set_anatomical_plausibility_real_scores(
						"proba_gabaergic_projection_score",
						proba_gabaergic_projection_score);
				ind.set_anatomical_plausibility_real_scores(
						"nb_interneuron_not_within_chip", nb_interneuron_not_within_chip);
				ind.set_anatomical_plausibility_real_scores(
						"nb_projection_within_chip", nb_projection_within_chip);
				ind.set_anatomical_plausibility_real_scores("nb_clusters", nb_clusters);
				ind.set_anatomical_plausibility_real_scores("ratio_inhib_score",
						ratio_inhib_score);
				ind.set_anatomical_plausibility_real_scores("proba_c_score",
						proba_c_score);
			}

			// Ponderation : a * score1 ^ b
			return (8 * proba_gabaergic_projection_score) - (1
					* nb_interneuron_not_within_chip) - (0.5 * nb_projection_within_chip)
			+ (0 * nb_clusters) + (8 * ratio_inhib_score) + (8 * proba_c_score);
		}

#ifdef EXP_SURVIVAL

		// Since C++ doesn't have a round function...
		double round(double d) {
			return floor(d + 0.5);
		}

		enum color_t {
			grey, white, black
		};
		enum action_t {
			wander, avoid, reload_on_dark, reload_on_light, rest
		};

		// Return the tile color where the robot is
		int get_tile_color(std::vector<std::vector<int> > tiles_pos,
				float tile_size_real_w, float tile_size_real_h, float robot_x,
				float robot_y) {

			int x = (int) floor(robot_x / tile_size_real_w);
			int y = (int) floor(robot_y / tile_size_real_h);
			//std::cout << "robot_x=" << robot_x;
			//std::cout << "map_size_x=" << map_size_h;
			//std::cout << "nb_tiles_x=" << nb_tiles_x;
			//std::cout << "x=" << x;
			//std::cout << "robot_y=" << robot_y;
			//std::cout << "map_size_y=" << map_size_w;
			//std::cout << "nb_tiles_y=" << nb_tiles_y;
			//std::cout << "y=" << y;
			if (x < tiles_pos.size() && y < tiles_pos[x].size()) {
				return tiles_pos[x][y];
			} else {
				return grey;
			}

			//return grey;
		}

		// Duplicate vectors so that each chip receive the same information
		std::vector<float> duplicate_inputs(size_t nb_chips,
				std::vector<float> vect) {
			std::vector<float> vect_dup;
			for (size_t chip_number = 0; chip_number < nb_chips; ++chip_number) {
				for (size_t j = 0; j < Params::dnn::nb_inputs; ++j) {
					vect_dup.push_back(vect[j]);
				}
			}
			return vect_dup;
		}

		// Utility function to normalize a vector, i.e. force its values to be
		// between min and max.
		std::vector<float> normalize_vector(std::vector<float> vect, float min,
				float max) {
			std::vector<float> out_vect;
			std::vector<float>::iterator it;
			for (it = vect.begin(); it < vect.end(); it++) {
				out_vect.push_back(sferes::misc::put_in_range<float>(*it, min, max));
			}
			return out_vect;
		}

		// Utility function to add an element to a list of fixed size
		template<typename L>
		void add_to_fixed_l_list(size_t size, L& l, const typename L::value_type& e) {
			if (l.size() >= size) {
				l.pop_back();
			}
			l.push_front(e);
		}

		// Utility function to analyze convergence of vectors
		bool conv(const std::list<std::vector<float> >& l, int nb_convergence,
				float convergence_threshold) {
			if (l.size() < nb_convergence) {
				return false;
			}
			std::list<std::vector<float> >::const_iterator it1 = l.begin();
			std::list<std::vector<float> >::const_iterator it2 = ++l.begin();
			while (it2 != l.end()) {
				for (size_t i = 0; i < it1->size(); ++i) {
					if (fabs((*it1)[i] - (*it2)[i]) > convergence_threshold) {
						return false;
					}
				}
				++it1;
				++it2;
			}
			return true;
		}

		// Forward propagation of the training pattern's input through
		// the neural network in order to generate the propagation's
		// output activations.
		template<typename Indiv>
		std::vector<float> propagate_inputs(Indiv& ind,
				std::vector<float> input_variables, bool &convergence) {

			convergence = false;
			// We store the last outputs to analyze convergence
			std::list<std::vector<float> > last_out;

			for (size_t step_number = 0; step_number < Params::fit::nb_steps; step_number++) {
				// Propagation
				ind.nn_mrf().step(input_variables);
				std::vector<float> out = ind.nn_mrf().get_outf();
				// Queue to analyze convergence
				add_to_fixed_l_list(Params::dnn::nb_convergence, last_out, out);
				// Check convergence
				if (conv(last_out, Params::dnn::nb_convergence,
								Params::dnn::convergence_threshold)) {
					convergence = true;
					return out;
				}
			}

			return ind.nn_mrf().get_outf();
		}

		// Compute chips' outputs vectors average, i.e. the global output of the mRF
		std::vector<float> compute_average(size_t nb_chips, std::vector<float> vect) {
			std::vector<float> output_vector_average(Params::dnn::nb_outputs);
			size_t count = 0;
			BOOST_FOREACH(float value, vect)
			{
				output_vector_average[count % Params::dnn::nb_outputs] += value;
				count++;
			}
			for (size_t z = 0; z < Params::dnn::nb_outputs; ++z) {
				output_vector_average[z] /= (float) nb_chips;
			}
			return output_vector_average;
		}

		// Winter-takes-all: only the neuron with the highest activation
		// stays active while all other neurons shut down
		int execute_wta(std::vector<float> vect) {
			std::vector<float>::iterator max_value_it = std::max_element(
					vect.begin(), vect.end());
			int max_position = std::distance(vect.begin(), max_value_it);
			return max_position;
		}

		// Apply one of the output transfers functions (see Humphries 2006 p7)
		float apply_output_transfer_function(float output, float output_contrast) {
			assert(output_contrast >= 0 && output_contrast <= 1);
			//0 is contrast isn't taken into account, actions are always fully activated
			if (Params::fit::output_transfer_function == 0) {
				return output;

				// 1 is actions are activated proportionally to the contrast
			} else if (Params::fit::output_transfer_function == 1) {
				return output * output_contrast;
				// 2 is actions are activated proportionally to the contrast square root
			} else if (Params::fit::output_transfer_function == 2) {
				return output * sqrt(output_contrast);
			}
			std::cout << "Params::fit::output_transfer_function must be 0, 1 or 2";
			assert(false);
			return 0.0f;
		}

		// Add conditions to make sure the mRF clearly selected an action.
		bool is_selection_clear(std::vector<float> vect) {

			if (Params::fit::action_selection_type == 0) {
				return true;
			} else if (Params::fit::action_selection_type == 1) {
				std::vector<float>::iterator max_value_it = std::max_element(
						vect.begin(), vect.end());
				int max_position = std::distance(vect.begin(), max_value_it);
				float lower_threshold = vect[max_position] - 0.05;
				float upper_threshold = 0;

				if (vect[max_position] < upper_threshold) {
					return false;
				}
				int count = 0;
				for (std::vector<float>::iterator it = vect.begin(); it < vect.end(); it++) {
					if (*it > lower_threshold && count != max_position) {
						return false;
					}
					count++;
				}
				return true;
			}

			std::cout << "Params::fit::action_selection_type must be 0 or 1";
			assert(false);
			return true;
		}

		// Survival task evaluation
		template<typename Indiv>
		void eval_survival(Indiv& ind) {

			// Initialization
#ifndef HUMPHRIES_MODEL_2006
		ind.nn_mrf().init();
		ind.nn_mrf().mark_useless_neurons();
		ind.log_survival_tasks().resize(Params::fit::nb_lives);
#else
		// mRF Humphries controller
		//std::vector<float> data(120);
		std::vector<float> data;
		for(size_t i = 0; i < ind.size(); i++)
		{
			//printf("phen: %f\n",ind.data(i));
			data.push_back(ind.data(i));
		}
		mrf* mrf_robot = new mrf(&data);

		// Initialization of sub-action vectors (there are 6 sub-actions):
		std::vector<float> move_forward(4,0);
		move_forward[0] = 5.0;
		move_forward[1] = 5.0;

		std::vector<float> move_backward(4,0);
		move_backward[0] = -5.0;
		move_backward[1] = -5.0;

		std::vector<float> turn_left(4,0);
		turn_left[0] = 5.5;
		turn_left[1] = -5.5;

		std::vector<float> turn_right(4,0);
		turn_right[0] = -5.5;
		turn_right[1] = 5.5;

		std::vector<float> reload_pe(4,0);
		reload_pe[3] = 1;

		std::vector<float> reload_e(4,0);
		reload_e[3] = 1;
		reload_e[2] = 1;

		// Put together all possible actions possibles in a vector
		std::vector<std::vector<float> > actions;
		actions.push_back(move_forward);
		actions.push_back(move_backward);
		actions.push_back(turn_left);
		actions.push_back(turn_right);
		actions.push_back(reload_pe);
		actions.push_back(reload_e);

#endif

		// _objs will store the final evaluation scores.
		this->_objs.resize(Params::fit::nb_objectives);

		// Map initialization
		// WARNING: the file must be in .pbm format and it must be begin with
		// P4.304 304 (GIMP adds a comment,
		// "P4# CREATOR: GIMP PNM Filter Version 1.1.304 304" you must delete
		// the comment string "# CREATOR: GIMP PNM Filter Version 1.1.".
		//std::cout << std::endl;
		std::cout << "go";
		float map_size = 400.0f;
		boost::shared_ptr<fastsim::Map> map_survival = boost::shared_ptr<
		fastsim::Map>(new fastsim::Map(Params::simu::map_name(), map_size));

		// Grid size
		int nb_tiles_h = 5;
		int nb_tiles_w = 5;

		// Size in pixels
		float map_size_pixel_h = map_survival->get_pixel_h();
		float map_size_pixel_w = map_survival->get_pixel_w();
		float tile_size_pixel_h = map_size_pixel_h / nb_tiles_h;
		float tile_size_pixel_w = map_size_pixel_w / nb_tiles_w;

		// Real size
		float map_size_real_h = map_survival->get_real_h();
		float map_size_real_w = map_survival->get_real_w();
		float tile_size_real_h = map_size_real_h / nb_tiles_h;
		float tile_size_real_w = map_size_real_w / nb_tiles_w;

		std::vector<std::vector<int> > tiles_pos;

		for (int h = 0; h < nb_tiles_h; ++h) {
			std::vector<int> temp_vect;
			for (int w = 0; w < nb_tiles_w; ++w) {
				temp_vect.push_back(grey);
			}
			//display_io_vector(temp_vect);
			tiles_pos.push_back(temp_vect);
		}

		tiles_pos[1][2] = white;
		tiles_pos[3][3] = white;
		tiles_pos[3][1] = black;
		tiles_pos[2][3] = black;

		// Draw tiles
		//int radius = sqrt(pow(tile_size_w, 2) + pow(tile_size_h, 2) / 2) / 2 ;
		int radius = tile_size_pixel_h / 2;

		for (int h = 0; h < nb_tiles_h; ++h) {
			for (int w = 0; w < nb_tiles_w; ++w) {
				if (tiles_pos[h][w] == white) {
					//display.circle((int)tile_size_w * (x + 0.5), (int)100, (int)tile_size_w/2);
					//display.disc((int)(tile_size_w * (y + 0.5)),
					//		(int)(tile_size_h * (x + 0.5)), (int)radius);

					map_survival->draw_rect((int) (tile_size_pixel_w * w),
							(int) (tile_size_pixel_h * h), tile_size_pixel_h,
							tile_size_pixel_w, fastsim::Map::white);
				}
				if (tiles_pos[h][w] == black) {
					map_survival->draw_rect((int) (tile_size_pixel_w * w),
							(int) (tile_size_pixel_h * h), tile_size_pixel_h,
							tile_size_pixel_w, fastsim::Map::black);
				}
			}
		}

		map_survival->terrain_switch_only_obstacles(Params::simu::map_name());

		// Fastsim initialization
		//map_survival->add_goal(fastsim::Goal(100, 100, 10, 0));
		fastsim::Robot robot(20.0f, fastsim::Posture(200, 200, 0));

		//robot.add_laser(fastsim::Laser(M_PI / 4.0, 100.0f));
		//robot.add_laser(fastsim::Laser(-M_PI / 4.0, 100.0f));
		//robot.add_laser(fastsim::Laser(0.0f, 100.0f));
		//robot.add_radar(fastsim::Radar(0, 4));
		//robot.move(1.0, 1.1, map_survival);
#ifdef USE_SDL
		fastsim::Display display(map_survival, robot);
#endif
		map_survival->update(robot.get_pos());
#ifdef USE_SDL
		display.update();
#endif

#ifdef USE_SDL
		while (false) {
			robot.move(0.2, 0.3, map_survival);
			display.update();
		}
#endif

		/********* Main loop: we simulate all robot's lives ***********/
		int total_num_time_step_alive = 0;
		float total_constrast = 0.0f;
		int nb_iterations = 0;
		// (one time-step is one second)
		int nb_time_steps = 0;
		int total_nb_livres = Params::fit::nb_lives;
		for (int num_life = 0; num_life < total_nb_livres; ++num_life) {
			// (Humphries 2005a p6-7)
			// The robot controller has six state variables available
			// to it, 4 external and 2 internal
			// External state variables
			float bl = 0.0f; // State of the left bumper (binary)
			float br = 0.0f; // State of the right bumper (binary)
			float ld = 0.0f; // Darkness value (binary)
			float lb = 0.0f; // Brightness value (binary)
			// lb = 1 on white, ld = 1 on black; lb = ld = 0 on neutral (i.e. grey)

			// Internal state variables
			float pe = 0.5f; // Robot's potential energy. Range (0, 1). Humphries initial value: 0.5f
			float e = 1.0f; // Robot's energy. Range (0, 1). Humphries initial value: 1.0f

			bool robot_alive = true;
			int rotation_direction = 0;
			int num_time_step_alive = 0;
			int selected_action = rest;

			// Teleport the robot to a random location in the map
			//fastsim::Robot robot(20.0f, fastsim::Posture(200, 200, 0));
			robot.set_pos(
					fastsim::Posture(
							sferes::misc::rand<int>(50, map_size_real_w - 50),
							sferes::misc::rand<int>(50, map_size_real_h - 50), 0));
			map_survival->update(robot.get_pos());

			/********* Main loop: we simulate a robot's life ***********/

#ifndef HUMPHRIES_MODEL_2006
		ind.log_survival_tasks()[num_life].resize(
				Params::fit::max_nb_time_steps);
#endif

		int num_time_step = 0;
		for (num_time_step = 0; num_time_step
				< Params::fit::max_nb_time_steps; ++num_time_step) {

			nb_iterations++;

			//std::cout << std::endl;
			// Reset _collision, _left_bumper and _right_bumper to false.
			//robot.check_collision(map_survival);
			//robot.reinit();

#ifdef USE_SDL
		display.update();
#endif

		// Get robot's position
		float robot_x = robot.get_pos().get_x();
		float robot_y = robot.get_pos().get_y();

		int robot_on_color = get_tile_color(tiles_pos, tile_size_real_w,
				tile_size_real_h, robot_x, robot_y);

		// Set variables available depending on the tile color the robot is on
		switch (robot_on_color) {
			case grey:
			ld = lb = 0.0f;
			break;
			case white:
			ld = 0.0f;
			lb = 1.0f;
			break;
			case black:
			ld = 1.0f;
			lb = 0.0f;
			break;
		}

		// Set variables available depending on the bumpers the robot may touch
		bl = br = 0.0f;
		if (robot.get_left_bumper()) {
			bl = 1.0f;
		}
		if (robot.get_right_bumper()) {
			br = 1.0f;
		}

		// Display robot's internal and external variables
		//std::cout << "bl=" << bl << ";";
		//std::cout << "br=" << br << ";";
		//std::cout << "ld=" << ld << ";";
		//std::cout << "lb=" << lb << ";";
		//std::cout << "pe=" << pe << ";";
		//std::cout << "e=" << e << ";";

		// Saliences computation - (Humphries 2005a p7)
		// These salience values are calculated at each behavioral update
		std::vector<float> saliences_input(4);

		// Wander a random walk in the environment, formed by forward movement
		// at a fixed speed	followed by a turn of a randomly selected angle
		// (2 time-steps).
		saliences_input[wander] = (-1) * bl - br + 0.8 * (1 - pe) + 0.9
		* (1 - e);

		// Avoid Obstacles: a maneuver to re-enter open space; the robot moves
		// backwards followed by either, if both bumpers activated, a turn of
		// 180° or, if one bumper activated, a turn of 45° in the opposite
		// direction to the activated	bumper (2 time-steps).
		saliences_input[avoid] = 3 * bl + 3 * br;

		// Reload On Dark: stop on a black tile and charge potential energy
		// (1 time-step).
		saliences_input[reload_on_dark] = (-2) * lb - bl - br + 3 * ld * (1
				- pe);

		// Reload On Light: stop on a white tile and charge energy by
		// consuming potential energy (1 time-step).
		saliences_input[reload_on_light] = (-2) * ld - bl - br + 3 * lb
		* (1 - e) * pow((1 - pow((1 - pe), 2)), 0.5);

		// Normalize salience, i.e. each value must be between 0 and 1.
		std::vector<float> saliences_input_normalized = normalize_vector(
				saliences_input, 0.0f, 1.0f);

		// Select action
		bool convergence = false;
		std::vector<float> output;
		std::vector<float> output_vector_average;
		int selected_action = rest;
		float output_contrast = 1.0f;
		int activation_degree = 1;
		// There is 4 available types of controller (see the parameter
		// Params::fit::controller_type in params_def.hpp)
#ifndef HUMPHRIES_MODEL_2006
		if (Params::fit::controller_type == 0) { // mrf controller
			// Duplicate vectors so that each chip receive the same information
			std::vector<float> saliences_input_dup = duplicate_inputs(
					ind.nb_chips(), saliences_input_normalized);

			convergence = false;
			// Propagate salience input in mRF & output are actions
			if (Params::fit::input_format == 0) {
				output
				= propagate_inputs(ind, saliences_input_dup, convergence);
			} else if (Params::fit::input_format == 1) {
				std::vector<float> input_direct;

				for (int chip_number = 0; chip_number < ind.nb_chips(); ++chip_number) {
					input_direct.push_back(bl);
					input_direct.push_back(br);
					input_direct.push_back(ld / 2);
					input_direct.push_back(lb / 2);
					input_direct.push_back(pe);
					input_direct.push_back(e);
					input_direct.push_back(1 - pe);
					input_direct.push_back(1 - e);
				}

				output = propagate_inputs(ind, input_direct, convergence);

				convergence = true;
			}
			//std::cout << "D";

			// Compute average output vector "output_vector_average"
			output_vector_average = compute_average(ind.nb_chips(), output);

			// Compute output contrast if the output transfer function is not == 0
			//if (Params::fit::output_transfer_function != 0) {
			//			output_vector_average[0] = 1;
			//			output_vector_average[1] = 0;
			//			output_vector_average[2] = 0;
			//			output_vector_average[3] = 0;
			output_contrast = compute_vector_contrast(output_vector_average);
			total_constrast += output_contrast;
			//std::cout << output_contrast;
			//}

			// Choose the action which has the biggest value (winner-takes-all)
			// If convergence is not reached, then the Rest behavior
			// is selected.
			selected_action = execute_wta(output_vector_average);
			//% ind.nb_chips();
			assert(selected_action >= 0 && selected_action <= rest);
			//std::cout << "C";
		} else if (Params::fit::controller_type == 1) { // WTA controller
			selected_action = execute_wta(saliences_input_normalized);
			//std::cout << "selected_action = " << selected_action << std::endl;
			convergence = true;
		} else if (Params::fit::controller_type == 2) { // Random controller
			selected_action = sferes::misc::rand<int>(0, 5);
			convergence = true;
		}
#else
		// if (Params::fit::controller_type == 3) { // mRF Humphries controller

		// Define 8 inputs
		std::vector<float> entree;
		entree.push_back(bl);
		entree.push_back(br);
		entree.push_back(ld);
		entree.push_back(lb);
		entree.push_back(e);
		entree.push_back(pe);
		entree.push_back(1 - e);
		entree.push_back(1 - pe);

		//mrf_robot->update(&entree, 0.001);
		for(unsigned int n = 0; n < 1; n++)
		{
			mrf_robot->update(&entree, 0.001);
		}

		// Get output
		std::vector<float> sk = mrf_robot->get_sk_triangle();
		//display_io_vector(sk);
		selected_action = rest;
		convergence = true;
		//}
#endif

		if (!convergence) {
			selected_action = rest; // Comment this line if you don't care about convergence
		}
		if (!is_selection_clear(output_vector_average)) {
			selected_action = rest;
		}
		//display_io_vector(output_vector_average);
		//std::cout << "selected_action = " << selected_action;
		//std::cout << selected_action;
		//std::cout << std::endl;
		//std::cout << "B";
		// Perform action

		// This information is present neither in Humphries 2005a
		// nor in Girard 2003/2008...
		// Yet it DOES have a great impact on the result.
		int step_forward_movement = 5;
		float random_angle = 0.0f;
		bool kill_individual = false;

		// If we allows only action to be selected
		if (Params::fit::selection_type == 0) {
			switch (selected_action) {

				case wander:
				nb_time_steps = 2;
				if (robot.get_collision()) {
					break;
				}
				// randomly selected angle for rotation
				random_angle = sferes::misc::rand<float>(0.0f, 1.0f) * M_PI * 3;

				if (sferes::misc::rand<float>(0.0f, 1.0f) < 0.1f) {
					//if ( num_time_step % 10 == 0) {
					//rotation_direction = sferes::misc::rand<int>(0, 2);
					rotation_direction = 1 - rotation_direction;
				}
				//std::cout << "random_angle=" << random_angle;
				//std::cout << "rotation_direction=" << rotation_direction;

				//random_angle = M_PI/4;
				//robot.move(cos(random_angle), sin(random_angle), map_survival);
				robot.move(random_angle * rotation_direction,
						random_angle * (1 - rotation_direction), map_survival);
				if (robot.get_collision()) {
					break;
				}
				// forward movement at a fixed speed
				robot.move(
						apply_output_transfer_function(step_forward_movement,
								output_contrast),
						apply_output_transfer_function(step_forward_movement,
								output_contrast), map_survival);
				//robot.get_pos().set_theta(sferes::misc::rand(0.0f, 2.0f) * M_PI);
				break;

				case avoid:
				nb_time_steps = 2;
				// Rotate depending on bumpers
				// TODO: check if we put right angle...
				if (bl == 0.0f && br == 0.0f) {
					if (Params::fit::dumm_actions_forbidden == 1) {
						kill_individual = true;
					}
					break;
				}

				robot.move(apply_output_transfer_function(-60, output_contrast),
						apply_output_transfer_function(-60, output_contrast),
						map_survival);
				robot.reinit();
				//std::cout << "M_PI=" << M_PI;
				//rotation_direction = 1 - rotation_direction;

				if (bl == 1.0f && br == 1.0f) {
					//if (bl == 1.0f || br == 1.0f) {
					robot.move(
							apply_output_transfer_function(robot.get_radius() * M_PI,
									output_contrast), 0, map_survival);
					//robot.move(-5, -5, map_survival);
				} else if (bl == 1.0f) {
					robot.move(
							apply_output_transfer_function(robot.get_radius() * M_PI,
									output_contrast), 0, map_survival);
					//robot.move(0, M_PI/2, map_survival);

				} else if (br == 1.0f) {
					//robot.move(M_PI/2, 0, map_survival);
					robot.move(
							0,
							apply_output_transfer_function(robot.get_radius() * M_PI,
									output_contrast), map_survival);
				}

				// forward movement at a fixed speed
				robot.move(
						apply_output_transfer_function(step_forward_movement,
								output_contrast),
						apply_output_transfer_function(step_forward_movement,
								output_contrast), map_survival);
				//robot.set_pos(fastsim::Posture(200, 200, 0));
				break;

				case reload_on_dark:
				nb_time_steps = 1;
				if (Params::fit::dumm_actions_forbidden == 1 && (ld == 0 || pe
								>= 1)) {
					kill_individual = true;
				}
				pe += apply_output_transfer_function(0.027 * ld, output_contrast);
				if (pe > 1) {
					pe = 1;
				}

				break;

				case reload_on_light:
				nb_time_steps = 1;
				if (Params::fit::dumm_actions_forbidden == 1 && (lb == 0 || e
								>= 1 || pe <= 0.27)) {
					kill_individual = true;
				}
				if (pe > 0.027) {
					e
					+= apply_output_transfer_function(0.027 * lb,
							output_contrast);
					pe -= apply_output_transfer_function(0.027 * lb,
							output_contrast);
				}
				if (e > 1) {
					e = 1;
				}

				break;

				case rest:
				nb_time_steps = 1;
				break;

			}
		}
#ifdef HUMPHRIES_MODEL_2006
		else if (Params::fit::selection_type == 1) {
			// If several actions can be selected at the same time
			nb_time_steps = 1;
			// Actions 0 and 1 make the robot move
			robot.move(sk[0]*actions[0][0]+ sk[1]*actions[1][0],
					sk[0]*actions[0][1]+ sk[1]*actions[1][1],
					map_survival);
			// Actions 2 and 3 make the robot rotate
			fastsim::Posture pos = robot.get_pos();
			pos.set_theta(pos.theta() + (sk[2]-sk[3])*M_PI);
			robot.set_pos(pos);
			//s.robot_rotate((sk[2]-sk[3])*3.14);
			// Action 4 : reload on dark
			pe += sk[4] * 0.027 * ld;
			if(pe > 1) {
				pe = 1;
			}
			// Action 5 : reload on bright
			if((pe - sk[5]*0.027 * lb) >= 0) {
				pe -= sk[5] * 0.027 * lb;
				e += sk[5] * 0.027 * lb;
			}
			if(e > 1) {
				e = 1;
			}
		}
#endif
		else {
			std::cout << "Params::fit::selection_type must be 0 or 1";
		}

		if (kill_individual) {
			nb_time_steps++;
			num_time_step++;
			num_time_step_alive++;
			break;
		}
		// Regardless of the action selected, energy E is consumed
		// at a constant rate of 0.002 unit/s.
		// Note that some actions take 2 time steps, other only 1.
		e -= 0.002 * nb_time_steps;
		num_time_step_alive += nb_time_steps;
		if (e <= 0) {
			robot_alive = false;
			break;
		}

		// Log the robot's position, decision and internal values
		//std::cout << "Z";
#ifndef HUMPHRIES_MODEL_2006
		ind.log_survival_tasks()[num_life][num_time_step].push_back(
				num_time_step_alive);
		ind.log_survival_tasks()[num_life][num_time_step].push_back(robot_x);
		ind.log_survival_tasks()[num_life][num_time_step].push_back(robot_y);
		ind.log_survival_tasks()[num_life][num_time_step].push_back(
				selected_action);
		ind.log_survival_tasks()[num_life][num_time_step].push_back(e);
		ind.log_survival_tasks()[num_life][num_time_step].push_back(pe);
		ind.log_survival_tasks()[num_life][num_time_step].push_back(bl);
		ind.log_survival_tasks()[num_life][num_time_step].push_back(br);
		ind.log_survival_tasks()[num_life][num_time_step].push_back(ld);
		ind.log_survival_tasks()[num_life][num_time_step].push_back(lb);
		for (std::vector<float>::iterator it = saliences_input.begin(); it
				< saliences_input.end(); it++) {
			ind.log_survival_tasks()[num_life][num_time_step].push_back(*it);
		}
		ind.log_survival_tasks()[num_life][num_time_step].push_back(
				output_contrast);
		for (std::vector<float>::iterator it =
				output_vector_average.begin(); it < output_vector_average.end(); it++) {
			ind.log_survival_tasks()[num_life][num_time_step].push_back(*it);
		}

#endif

		num_time_step += (nb_time_steps - 1);
	} // End of main loop - robot's life
	//std::cout << "R";
	total_num_time_step_alive += num_time_step_alive;
#ifndef HUMPHRIES_MODEL_2006
		ind.survival_times().push_back(num_time_step_alive);
#endif

		// Generalization objective :
		// If this was the last life of the usual Params::fit::nb_lives life number,
		// check out average survival time. If the latter is high, then we'll compute
		// more survival tasks.
		if (num_life == Params::fit::nb_lives - 1) {
			// Compute the first 2 objectives
			this->_objs[0] = total_num_time_step_alive / float(
					Params::fit::nb_lives);
			if (this->_objs.size() >= 2) {
				assert(total_num_time_step_alive > 0);
				this->_objs[1] = total_constrast / nb_iterations;
			}

			if (this->_objs[0] < Params::fit::min_score_for_generalization) {
				break;
			}
			total_nb_livres = Params::fit::nb_lives_generalization;
#ifndef HUMPHRIES_MODEL_2006
		ind.log_survival_tasks().resize(
				Params::fit::nb_lives_generalization);
#else
		break;
#endif

	}
}// End of all robot's lives

//std::cout << "S";
// The WTA controller thus simply
// selects the action with the highest salience value
// as the winner, and the robot executes that action.
// The random controller simply randomly selects one
// of the five possible actions with equal probability at
// each behavioral update.

		if (this->_objs[0] >= Params::fit::min_score_for_generalization) {
this->_objs[0] = Params::fit::max_nb_time_steps + total_num_time_step_alive / float(
		Params::fit::nb_lives_generalization);
		}

// Benoît Girard saliences:
//saliences_input[0] = 300 * 3 * onEBlob * (2/(1+exp(-5 * Ep * (1-E))) - 1); // ReloadE
//saliences_input[1] = 300 * 2 * onEpBlob * (2/(1+exp(-5 * (1-Ep))) - 1); // ReloadEp
//saliences_input[2] = 300 * 0.39; // Wander
//saliences_input[3] = 300 * E * Ep /1.8; // Sleep

// Compute the remaining objectives
// this->_objs[0] = total_num_time_step_alive / float(Params::fit::nb_lives);
#ifndef HUMPHRIES_MODEL_2006
		float anat_plausibility = anatomical_plausibility(ind);
		float anat_plausibility_real = anatomical_plausibility(ind, true);

		//std::cout << "num_time_step_alive=" << num_time_step_alive;
		//if (this->_objs.size() >= 2) {
		//	assert(total_num_time_step_alive > 0);
		//	this->_objs[1] = total_constrast / nb_iterations;
		//}
		if (this->_objs.size() >= 3) {
			this->_objs[2] = anat_plausibility;
		}
#else
		this->_value = total_num_time_step_alive / float(
				Params::fit::nb_lives);
		//std::cout << "value=" << this->_value;
#endif
	}
#endif

		template<typename Indiv>
		void eval(Indiv& ind) {
			//std::cout << "counter_individuals = " << counter_individuals << " ; ";
			counter_individuals++;
			long int analyze_from = 9999999;
			// init everything. init() is usually called at the beginning of the fitness function
			// to allow a few optimizations.
#ifndef HUMPHRIES_MODEL_2006
		ind.nn_mrf().init();
		ind.nn_mrf().mark_useless_neurons();
#endif
		if (counter_individuals > analyze_from) {
			std::cout << std::endl << "in fitness function:";
			ind.show(std::cout);
		}

		// _objs will store the final evaluation scores.
		this->_objs.resize(Params::fit::nb_objectives);

		// init objectives
		std::fill(this->_objs.begin(), this->_objs.end(), 0.0f);

#ifdef EXP_SURVIVAL
		this->eval_survival(ind);
		return;
#endif
#ifndef HUMPHRIES_MODEL_2006
		// Generate inputs/outputs pairs
		//std::vector< std::vector<io_t> > all_inputs_variables;
		//std::vector< std::vector<io_t> > all_outputs_variables;


		// Main loop: propagate input vectors in the mRF and compute objective function
		// Objective 1: minimize Euclidean distance between
		//              vector goal and nn output vector
		//              -> float distance
		// Objective 2: maximize right number of good selection (i.e. biggest value
		//              in the input vector is the biggest value in the output vector)
		//              -> int good_selections
		// Objective 3: maximize coherence with anatomical data
		//              -> float anatomical_plausibility(Indiv& ind)
		// Objective 4: maximize contrast between the maximum value and the others.
		//              -> constrast_sum
		float distance = 0.0f;
		int good_selections = 0;
		float constrast_sum = 0.0f;
		for (size_t i = 0; i < all_inputs_variables.size(); ++i) {
			// ind.nn_mrf().init();
			// Set inputs and wanted outputs for the nn
			std::vector<io_t> input_variables = all_inputs_variables[i];
			//std::cout << "all_inputs_variables.size() = " << all_inputs_variables.size() << " ";
			//std::cout << "input_variables.size() = " << input_variables.size() << " ";
			std::vector<io_t> output_variables = all_outputs_variables[i];

			int count = 0;
			//std::cout << std::endl;

			if (counter_individuals > analyze_from)
			display_io_vector(input_variables, 0, 1);
			if (counter_individuals > analyze_from)
			display_io_vector(output_variables, 0, 1);

			// Propagate inputs in nn
			std::vector<io_t> output_vector;
			bool stop = (counter_individuals > analyze_from);
			for (size_t step_number = 0; step_number < Params::fit::nb_steps; step_number++) {
				ind.nn_mrf().step(input_variables);
				// print all elements of the output
				std::vector<io_t> output_vector = ind.nn_mrf().get_outf();
				if (counter_individuals > analyze_from)
				display_io_vector(output_vector, 0, 1);
			}
			output_vector = ind.nn_mrf().get_outf();

			// Compute average output vector "output_vector_average"
			std::vector<io_t> output_vector_average(Params::dnn::nb_outputs);
			count = 0;
			BOOST_FOREACH(io_t value, output_vector)
			{
				output_vector_average[count % Params::dnn::nb_outputs]
				+= value; //output_variables[] //
				count++;
			}
			for (size_t z = 0; z < Params::dnn::nb_outputs; ++z) {
				output_vector_average[z] /= (float) ind.nb_chips();
			}

			// Objective 1: Compute Euclidean distance between vector goal and nn output vector
			count = 0;
			float distance_temp = distance;
			distance = 0;
			BOOST_FOREACH(io_t value, output_variables)
			{
				if (counter_individuals > analyze_from
						&& output_vector[count] < 0) {
					std::cout << output_variables[count] << " - "
					<< output_vector[count] << " ; ";
					std::cout << pow(
							output_variables[count] - output_vector[count], 2)
					<< " ; ";
					std::cin.get();
				}
				distance += pow(
						output_variables[count] - output_vector[count], 2);
				count++;
				//std::cout <<";"<<distance;
			}
			//std::cout << std::endl;
			distance = sqrt(distance) + distance_temp;

			// Objective 2: maximize the number of good selections (i.e. biggest value
			// 	            in the input vector is the biggest value in the output vector)
			//int nb_of_max = 0;
			int max_position = -1;
			int max_value = -999;
			int loop_count = 0;
			int nb_max = 0;
			float max_vector_value = *(std::max_element(
							output_vector_average.begin(), output_vector_average.end()));
			for (std::vector<io_t>::iterator it = output_vector_average.begin(); it
					!= output_vector_average.end(); ++it) {
				if (*it > max_value) {
					max_value = *it;
					max_position = loop_count;
				}
				if (*it == max_vector_value) {
					nb_max++;
				}
				loop_count++;
			}
			assert(nb_max >= 1);
			//std::cout << "nb_max" << nb_max << ";";

			// Or, if max isn't 1, instead of == 1
			// do  == *(std::max_element(output_variables.begin(), output_variables.end()));
			// Better to precompute the max value in init_random_io*() for running time optimization,
			// as it's the case when doing == 1.
			// Note: we could use nb_of_max to make sure there is only 1 max in output_vector,
			//       but it's very unlikely to happen. Better to ignore it for running time
			//       optimization again.
			if (output_variables[max_position] == 1) {
				good_selections++;
			}

			// Objective 4: maximize contrast between the selected input and the others.

			// We compute the ratio between the output's contrast and the input's contrast
			if (all_inputs_variables_contrast[i] > 0) {
				constrast_sum += (compute_vector_contrast(output_vector_average)
						/ ((float) all_inputs_variables_contrast[i]));
			}
			//std::cout << "constrast_sumTEMP=" << constrast_sum << ";";

		}
		// Normalize the distance so as to make it comparable with other experiments
		distance /= all_inputs_variables.size() * ind.nb_chips();
		constrast_sum /= all_inputs_variables.size();
		//std::cout << "constrast_sumf=" << constrast_sum << ";";
		//if ( distance > 0 ) distance = 1.0 / distance;
		//else distance = std::numeric_limits<int>::max();

		distance = (-1) * distance;
		//constrast_sum = (-1) * constrast_sum;

		//if (distance > 1.2) std::cin.get();
		//this->_objs[0] = distance;
		//std::cout << "good_selections=" << good_selections;
		//std::cout << "all_inputs_variables.size()=" << all_inputs_variables.size();
		//std::cout << "constrast_sum=" << constrast_sum << ";";
		//std::cout << "Params::dnn::nb_outputs=" << Params::dnn::nb_outputs;
		//std::cout << std::endl;
		this->_objs[0] = (constrast_sum / contrast_best_possible_average) + (4
				* (((float) good_selections) / all_inputs_variables.size()));// + (good_selections - all_inputs_variables.size());
		//this->_objs[0] = good_selections;
		//std::cout <<"this->_objs[0]="<<this->_objs[0]<<";" << std::endl;
		//if (this->_objs[0] > 1) {
		//	std::cin.get();
		//}
		if (counter_individuals > analyze_from) {
			std::cout << std::endl << " this->_objs[0]=" << this->_objs[0] << ";";
		}

		//std::cout << "nb_inputsEVAL=" << ind.nn_mrf().get_nb_inputs() << ";";
		float anat_plausibility = anatomical_plausibility(ind);
		float anat_plausibility_real = anatomical_plausibility(ind, true);
		if (this->_objs.size() >= 2) {
			//this->_objs[1] = anat_plausibility;
			this->_objs[0] = constrast_sum / contrast_best_possible_average;
			this->_objs[1] = good_selections;
			//std::cout << std::endl << " this->_objs[1]=" << this->_objs[1] << ";";
			if (counter_individuals > analyze_from) {
				std::cout << std::endl << " this->_objs[1]=" << this->_objs[1]
				<< ";";
			}
		}
		if (this->_objs.size() >= 3) {
			this->_objs[0] = constrast_sum / contrast_best_possible_average;
			this->_objs[1] = good_selections;
			this->_objs[2] = anat_plausibility;
		}

		// Penalties
		//std::cout << " ind.nn_mrf().get_nb_connections()=" << ind.nn_mrf().get_nb_connections() << " ;";
		/*if (ind.nn_mrf().get_nb_connections() != 4 ) {
		 this->_objs[0] = -10;
		 this->_objs[1] = -10;
		 }*/

		if (this->_objs[0] > 0.4) {
			//ind.show(std::cout);
			//std::cin.get();
		}
		if (counter_individuals > analyze_from) {
			std::cin.get();
		}

		// Save all objective scores of the mRF
		ind.set_objective_scores("anat_plausibility", anat_plausibility);
		ind.set_objective_scores("anat_plausibility_real",
				anat_plausibility_real);
		ind.set_objective_scores("constrast_sum", constrast_sum);
		ind.set_objective_scores("distance", distance);
		ind.set_objective_scores("good_selections", good_selections);
#endif
	}
}
;

// This function checks if all parameters are correctly set
void check_params() {
	assert(Params::dnn::nb_inputs > 0);
	assert(Params::dnn::nb_outputs > 0);
	assert(Params::fit::nb_objectives > 0);
#ifdef EXP_SURVIVAL
	assert(Params::fit::nb_lives > 0);
	if (Params::fit::input_format == 0) {
		assert(Params::dnn::nb_inputs == 4);
	} else if (Params::fit::input_format == 1) {
		assert(Params::dnn::nb_inputs == 8);
	}
#endif
#ifdef HUMPHRIES_MODEL_2006
	assert(Params::fit::controller_type == 3);
#else
	assert(Params::fit::controller_type != 3);
#endif
}

using namespace sferes;
using namespace sferes::gen::evo_float;

#include <iostream>
#include <sferes/phen/parameters.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/ea/rank_simple.hpp>
#include <sferes/ea/nsga2.hpp>
#include <sferes/stat/pareto_front.hpp>
#include <sferes/eval/eval.hpp>
#include <sferes/stat/best_fit.hpp>
#include <sferes/stat/mean_fit.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/run.hpp>
#include <modules/fastsim/simu_fastsim.hpp>

#ifdef HUMPHRIES_MODEL_2006
// Humphries controller
int main_humphries(int argc, char **argv) {

	std::cout << "main_humphries";
	typedef MrfFit<Params> fit_t;
	typedef gen::EvoFloat<120, Params> gen_t;
	typedef phen::Parameters<gen_t, fit_t, Params> phen_t;
	typedef eval::Eval<Params> eval_t;

	typedef modif::Dummy<> modifier_t;
	typedef boost::fusion::vector<sferes::stat::BestFit<phen_t, Params>,
	sferes::stat::MeanFit<Params> > stat_t;
	typedef ea::RankSimple<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;

	//typedef boost::fusion::vector<stat::ParetoFront<phen_t, Params> >  stat_t;
	//typedef ea::Nsga2<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;

	ea_t ea;
	run_ea(argc, argv, ea);
	return EXIT_SUCCESS;
}
#endif

// Main function for our evolving our model
int main(int argc, char **argv) {
	std::cout << "running " << argv[0] << " ... try --help for options (verbose)"
			<< std::endl;
	check_params();

#ifdef HUMPHRIES_MODEL_2006
	if (Params::fit::controller_type == 3) {
		main_humphries(argc, argv);
	}
#else

	// We define the genotype. Here we choose to define our own genotype, mRFgen,
	// which is basically a vector of NN.
	typedef sferes::gen::MrfGen<Params> gen_t;
	// typedef gen::EvoFloat<30, Params> gen_t2;
	// The genotype will be transformed into one big NN (i.e. the phenotype), which will be
	// evaluated through the fitness funtion mRFfit<Params>.
	typedef phen::MrfPhen<gen_t, MrfFit<Params> , Params> phen_t;
	// Evaluator: it will It should call phen_t::develop() then
	// phen_t::fit::eval() for each individual.
	typedef eval::Eval<Params> eval_t;
	// Record stats: sferes::stat::ParetoFront<phen_t, Params> will output pareto.dat
	typedef boost::fusion::vector<sferes::stat::ParetoFront<phen_t, Params>,
			sferes::stat::AllIndiv<phen_t, Params>,
			sferes::stat::AllIndivObjectiveScores<phen_t, Params>,
			sferes::stat::AllIndivRealDataStats<phen_t, Params>,
#ifdef EXP_SURVIVAL
			sferes::stat::AllIndivLogMovesSurvival<phen_t, Params>,
			sferes::stat::AllIndivLogSurvivalTaskScores<phen_t, Params>,
#endif
			sferes::stat::AllIndivDataStats<phen_t, Params> > stat_t;
	// Modifiers are run once all the individuals have been evaluated but before
	// any sorting. They are designed to allow to modify the fitness values to
	// implement niching strategies, diversity preservation mechanisms, etc
	// The only predefined modifier available in sferes2 is modif::Dummy,
	// which does nothing.
	typedef modif::Dummy<Params> modifier_t;
	// Choose the evolutionary algorithm we want to use:
	// NSGA-II is a very famous multi-objective optimization algorithm.
	typedef ea::Nsga2<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;
	ea_t ea;

#ifdef EXP_6_VECT
	// Initialize 6 input-output pairs
	init_random_io();
#endif
#ifdef EXP_1155_VECT
	// Initialize 1155 input-output pairs
	init_random_io2();
#endif

	// Run the evolutionary algorithm (ea). Note we can add an additional parameter to
	// run_ea() to give the seed of the random.
	run_ea(argc, argv, ea);

#endif
	return EXIT_SUCCESS;
}

#endif

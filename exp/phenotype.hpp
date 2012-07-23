#ifndef PHENOTYPE_HPP
#define PHENOTYPE_HPP

#include <sferes/phen/indiv.hpp>
#include <sferes/stc.hpp>
#include <modules/nn2/nn.hpp>
#include <iostream>

#include "mrf.cpp"

namespace sferes {
	namespace phen {
		SFERES_INDIV(MrfPhen, Indiv){
		public:
		typedef typename Gen::nn_t nn_t;
		typedef typename Gen::nn_t::vertex_desc_t vertex_desc_t;
		typedef typename Gen::nn_t::vertex_list_t vertex_list_t;
		typedef typename Gen::nn_t::edge_desc_t edge_desc_t;
		typedef typename Gen::genotype_t genotype_t;

		// If one day we want to decrease the RAM, use the follow struct
				// instead of copying chips in std::map<int, genotype_t> _chips_gen
				// if it's still only used for statistical purpose
				struct chip_stats_t {
					size_t nb_neurons;
					size_t nb_connections;
				};

				// constructor
				MrfPhen() {
					//std::cout << "newMRF";
				}

				// develops the genotype, called by the evaluator before evaluating
				// the individual.
				void develop() {

					// Initialize the mRF nn (which is the phenotype)
					_nn_mrf = nn_t();

					// Map the vertex from our chips to the mRF to keep track of them. (rmap stands for Relation map)
					std::map<vertex_desc_t, vertex_desc_t> rmap;
					std::map<vertex_desc_t, vertex_desc_t> reverse_rmap;

					// Copy all chips in mRF
					for (size_t chip_number = 0; chip_number < this->gen().chips().size(); ++chip_number) {
						add_subnn(rmap, reverse_rmap, this->gen().chips(chip_number).get_graph(), chip_number);
						_chips_gen[chip_number] = this->gen().chips(chip_number); // Note: this takes a lot of RAM (> 50 MiB)
					}

					// Run a few checks to make sure the phenotype is correct.
					unsigned int total_connections = 0;
					for (size_t chip_number = 0; chip_number < this->gen().chips().size(); ++chip_number) {
						total_connections += this->gen().chips(chip_number).get_nb_connections();
					}
					//std::cout << "r";
					if (_nn_mrf.get_nb_connections() != total_connections) {
						this->gen().chips(0).write(std::cout);
						std::cout << std::endl;
						_nn_mrf.write(std::cout);
						std::cin.get();
					}
					assert(_nn_mrf.get_nb_connections() == total_connections);
					//std::cout << "t";

					unsigned int total_neurons = 0;
					for (size_t chip_number = 0; chip_number < this->gen().chips().size(); ++chip_number) {
						total_neurons += this->gen().chips(chip_number).get_nb_neurons();
					}
					assert(_nn_mrf.get_nb_neurons() == total_neurons);

					// Copy all interconnections
					for (size_t interconnection_number = 0; interconnection_number
							< this->gen().interconnections().size(); ++interconnection_number) {
						typename Gen::interconnection_t interconnection = this->gen().interconnections(interconnection_number);
						vertex_desc_t vertex_src = rmap[this->gen().chips(interconnection.src_chip_id).get_neuron(interconnection.src_neuron_id)];
						vertex_desc_t vertex_dest = rmap[this->gen().chips(interconnection.dest_chip_id).get_neuron(interconnection.dest_neuron_id)];
						interconnection.connection.get_weight().develop(); // So that connection's weight isn't 0 in the mRF graph (WHY?)
						//std::cout << "a";
						_nn_mrf.add_connection(vertex_src, vertex_dest, interconnection.connection.get_weight(), false);
						//std::cout << "c";
					}
					//std::cout << "end";

					//std::cout << "_nn_mrf summary: ";
					//show(std::cout);
					//std::cin.get();

					// Define handicrafted phenotype
					/*
					 _nn_mrf = nn_t();
					 _nn_mrf.set_nb_inputs(4);
					 _nn_mrf.set_nb_outputs(4);
					 std::vector<vertex_desc_t> inputs_neurons = _nn_mrf.get_inputs();
					 std::vector<vertex_desc_t> outputs_neurons = _nn_mrf.get_outputs();
					 typename Gen::weight_t weight;
					 weight.show(std::cout);
					 std::cin.get();
					 //weight.set_weight(1); // not working
					 _nn_mrf.add_connection(inputs_neurons[0], outputs_neurons[0], weight);
					 */

#ifdef DEL_ISOLATED_NEURONS
				// Delete all isolated neurons (except input/output neurons)
				vertex_list_t isolated_neurons = _nn_mrf.get_isolated_neurons_without_io();
				for (typename vertex_list_t::iterator it = isolated_neurons.begin(); it
						!= isolated_neurons.end(); ++it) {
					//std::cout << "del!";
					//std::cout << this->gen().chips(_nn_mrf.get_graph()[*it].numchip()).get_nb_neurons() << ";";
					this->gen().chips(_nn_mrf.get_graph()[*it].numchip()).del_neuron(reverse_rmap[*it]);
					_nn_mrf.del_neuron(*it);
					//std::cout << this->gen().chips(_nn_mrf.get_graph()[*it].numchip()).get_nb_neurons() << ";";
				}
				//for (size_t chip_number = 0; chip_number < this->gen().chips().size(); ++chip_number) {
				//	this->gen().chips(chip_number).del_isolated_neurons();
				//}
#endif

			}

			// Ad a neural network in another neural network.
			// Note: there will be no connection at all between the new neurons
			// (i.e. of the added neural network) and the old ones (i.e. the
			// neural network in which we add a new neural network)
			void add_subnn(std::map<vertex_desc_t, vertex_desc_t>& rmap,
					std::map<vertex_desc_t, vertex_desc_t>& reverse_rmap,
					typename Gen::nn_t::graph_t& graph_src, size_t chip_number) {
				// Copy neurons (1 neuron = 1 vertex)
				int count = 0;
				BGL_FORALL_VERTICES_T(vertex, graph_src, typename Gen::nn_t::graph_t) // Problem with is_input(vertex)
				//BOOST_FOREACH(vertex_desc_t vertex, all_neurons)
				{
					// Add new vertex in the mRF NN
					// Note : add_neuron() will call rams() and set_pfparams() in nn.hpp
					//        If we don't call rams() and set_pfparams(), we'll fail
					//        assert(_tau != 0); in  float operator()(float p) in <modules/evoneuro2/lpds.hpp>
					graph_src[vertex].get_afparams().develop(); // Maybe useless
					graph_src[vertex].get_pfparams().develop(); // Maybe useless
					typename Gen::potential_t::params_t pfparams = graph_src[vertex].get_pfparams();
					typename Gen::activation_t::params_t afparams = graph_src[vertex].get_afparams();

					//afparams.show(std::cout);
					//std::cin.get();
					std::string vertex_label = std::string("chipi") + boost::lexical_cast<std::string>(count);
					vertex_desc_t vertex_copy = _nn_mrf.add_neuron(vertex_label, pfparams, afparams);
					_nn_mrf.get_graph()[vertex_copy].get_afparams().develop(); // Maybe useless
					_nn_mrf.get_graph()[vertex_copy].get_pfparams().develop(); // Maybe useless

					// Need this fix, don't know why: sometimes inhib is false but data(2) > 0.5,
					// which causes a bug.
					/*
					 if (graph_src[vertex].get_afparams().data(2) > 0.5) {
					 graph_src[vertex].set_inhib(true);
					 } else {
					 graph_src[vertex].set_inhib(false);
					 }
					 */
#ifdef CONSTRAINT_EDGE_ALWAYS_INHIB_AND_EXCIT
				//assert(graph_src[vertex].inhib() != _nn_mrf.get_graph()[vertex_copy].inhib());
#endif

				// Display neuron's ID
				//std::cout << std::endl << "ID : " << _nn_mrf.get_graph()[vertex_copy].get_id()<< ";";

				// Keep track of it
				rmap[vertex] = vertex_copy;
				reverse_rmap[vertex_copy] = vertex;
				_vertex_numchip_map[vertex_copy] = chip_number;
				_nn_mrf.get_graph()[vertex_copy].set_numchip(chip_number);

				// Check if the neuron input or output
				// Could have use (this->gen().chips(chip_number).is_input(vertex)){, but caution
				// check if not always true.
				// _in is -1 if not an input of the nn, id of input otherwise (see neuron.hpp)
				if (!(graph_src[vertex].get_in() == -1)) {
					_nn_mrf.add_input(vertex_copy);
				}
				// _in is -1 if not an input of the nn, id of input otherwise (see neuron.hpp)
				else if (!(graph_src[vertex].get_out() == -1)) {
					_nn_mrf.add_output(vertex_copy);
				}

				count++;
			}

			// Copy connections (1 connection = 1 edge)
			BGL_FORALL_EDGES_T(edge, graph_src, typename Gen::nn_t::graph_t)
			{
				// Set weights
				graph_src[edge].get_weight().develop();

				//int in  = graph_src[source(edge, graph_src)].get_in();
				//int out = graph_src[target(edge, graph_src)].get_out();
				//std::cout << "a";
				_nn_mrf.add_connection(rmap[source(edge, graph_src)], rmap[target(edge, graph_src)], graph_src[edge].get_weight());
				//std::cout << "b";
			}

		}

		// Describe this individual
		void show(std::ostream& os) {_nn_mrf.write(os);}

		// Describe this individual's stats
		void show_data_stats(std::ostream& os) {
			// We cannot iterate because we want a precise order
			os << this->anatomical_stats("proba_gabaergic_projection") << ",";
			os << this->anatomical_stats("nb_interneuron_not_within_chip") << ",";
			os << this->anatomical_stats("nb_projection_within_chip") << ",";
			os << this->anatomical_stats("nb_clusters") << ",";
			os << this->anatomical_stats("nb_neurons") << ",";
			os << this->anatomical_stats("nb_useless_neurons") << ",";
			os << this->anatomical_stats("nb_connections") << ",";
			os << this->anatomical_stats("nb_useless_connections") << ",";
			os << this->anatomical_stats("ratio_inhib") << ",";
			os << this->anatomical_stats("proba_c") << ",";
			os << this->anatomical_stats("proba_p") << ",";
			os << this->anatomical_stats("proba_l");
		}

		// Describe this individual's "real" stats
		// i.e. by ignoring the useless neurons
		void show_real_data_stats(std::ostream& os) {
			// We cannot iterate because we want a precise order
			os << this->anatomical_real_stats("proba_gabaergic_projection") << ",";
			os << this->anatomical_real_stats("nb_interneuron_not_within_chip") << ",";
			os << this->anatomical_real_stats("nb_projection_within_chip") << ",";
			os << this->anatomical_real_stats("nb_clusters") << ",";
			os << this->anatomical_real_stats("nb_neurons") << ",";
			os << this->anatomical_real_stats("nb_useless_neurons") << ",";
			os << this->anatomical_real_stats("nb_connections") << ",";
			os << this->anatomical_real_stats("nb_useless_connections") << ",";
			os << this->anatomical_real_stats("ratio_inhib") << ",";
			os << this->anatomical_real_stats("proba_c") << ",";
			os << this->anatomical_real_stats("proba_p") << ",";
			os << this->anatomical_real_stats("proba_l");
		}

		// Describe this individual's objective scores
		void show_objective_scores(std::ostream& os) {
			// We cannot iterate because we want a precise order
			os << this->objective_scores("anat_plausibility") << ",";
			os << this->objective_scores("anat_plausibility_real") << ",";
			os << this->objective_scores("constrast_sum") << ",";
			os << this->objective_scores("distance") << ",";
			os << this->objective_scores("good_selections");
		}

		// Display a vector
		template<typename t>
		void display_vector(std::ostream& os, std::vector<t> vector_to_display) {
			BOOST_FOREACH(t value, vector_to_display)
			{
				os << value << ", ";
			}
		}

		// Display the robot log during the survival tasks: moves
		void show_log_survival_tasks(std::ostream& os) {
			for (int num_life = 0;
					//num_life < Params::fit::nb_lives;
					num_life < 1;
					++num_life) {

				for (int num_time_step = 0;
						//num_time_step < Params::fit::max_nb_time_steps;
						num_time_step < this->log_survival_tasks()[num_life].size();
						++num_time_step) {
					if (this->log_survival_tasks()[num_life][num_time_step].size() < 2) continue;
					display_vector(os, this->log_survival_tasks()[num_life][num_time_step]);
					os << std::endl;
				}
			}
		}

		// Display the robot log during the survival tasks: survival times for each task
		void show_log_survival_times(std::ostream& os) {
			for ( std::vector<float>::iterator it=this->survival_times().begin() ; it < this->survival_times().end(); it++ ) {

				std::string sep = ", ";
				os << *it << sep;
				}
		}

		// Getters
		nn_t& nn_mrf() {return this->_nn_mrf;}
		int nb_chips() {return this->gen().nb_chips();}
		std::vector<typename Gen::genotype_t> chips() {return this->gen().chips();}
		std::map<vertex_desc_t, int>& vertex_numchip_map() {return _vertex_numchip_map;}
		int vertex_numchip_map(vertex_desc_t vertex) {return _vertex_numchip_map[vertex];}

		std::map<std::string, float>& anatomical_plausibility_scores() {return _anatomical_plausibility_scores;}
		float anatomical_plausibility_scores(std::string constraint) {return _anatomical_plausibility_scores[constraint];}
		std::map<std::string, float>& anatomical_stats() {return _anatomical_stats;}
		float anatomical_stats(std::string data) {return _anatomical_stats[data];}

		std::map<std::string, float>& anatomical_plausibility_real_scores() {return _anatomical_plausibility_real_scores;}
		float anatomical_plausibility_real_scores(std::string constraint) {return _anatomical_plausibility_real_scores[constraint];}
		std::map<std::string, float>& anatomical_real_stats() {return _anatomical_real_stats;}
		float anatomical_real_stats(std::string data) {return _anatomical_real_stats[data];}

		std::map<std::string, float>& objective_scores() {return _objective_scores;}
		float objective_scores(std::string data) {return _objective_scores[data];}
		std::map<int, genotype_t>& chips_gen() {return _chips_gen;}
		genotype_t chips_gen(int data) {return _chips_gen[data];}
		std::vector<float>& survival_times() {return _survival_times;}
		float survival_times(int index) {return _survival_times[index];}


#ifdef EXP_SURVIVAL
				std::vector<std::vector<std::vector<float> > >& log_survival_tasks() {
					return this->_log_survival_tasks;
				}
#endif

				// Setters
				void set_anatomical_plausibility_scores(std::string constraint, float score) {
					_anatomical_plausibility_scores[constraint] = score;
				}
				void set_anatomical_stats(std::string data, float score) {
					_anatomical_stats[data] = score;
				}

				void set_anatomical_plausibility_real_scores(std::string constraint, float score) {
					_anatomical_plausibility_real_scores[constraint] = score;
				}
				void set_anatomical_real_stats(std::string data, float score) {
					_anatomical_real_stats[data] = score;
				}

				void set_objective_scores(std::string data, float score) {
					_objective_scores[data] = score;
				}

				protected:
				// This NN will contain the whole mRF
				nn_t _nn_mrf;
				// This is the number of chips of the mRF
				int _nb_chips;
				// This map contains the chip's number a neuron in the mRF belongs to.
				std::map<vertex_desc_t, int> _vertex_numchip_map;
				// This map contains the plausibility score for each anatomical constraint.
				std::map<std::string, float> _anatomical_plausibility_scores;
				// This map contains the "real" plausibility score for each anatomical
				// constraint, i.e. by ignoring the useless neurons
				std::map<std::string, float> _anatomical_plausibility_real_scores;
				// This map contains the anatomical statistics of the mRF.
				std::map<std::string, float> _anatomical_stats;
				// This map contains the "real" anatomical statistics of the mRF
				// i.e. by ignoring the useless neurons
				std::map<std::string, float> _anatomical_real_stats;
				// This map contains the objective scores of the mRF.
				std::map<std::string, float> _objective_scores;
				// This map contains the objective scores of the mRF.
				std::map<int, genotype_t> _chips_gen;
				// This map contains the moves of the robot during the survival task
				// for each, as well as the robot's internal values (energy, potential
				// energy, ...) and decisions (wander, avoid, reload, and so on).
				std::vector<std::vector<std::vector<float> > > _log_survival_tasks;
				std::vector<float> _survival_times;

			};
		}
	}

#endif

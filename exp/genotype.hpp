#ifndef GENOTYPE_HPP
#define GENOTYPE_HPP

#include <sferes/stc.hpp>
#include <modules/nn2/gen_dnn.hpp> // For DNN (dynamic Neural Network), i.e. NN with evolutionary computing
// operators (mutation, crossover, etc)
#include <modules/nn2/nn.hpp> // Neural Network module.
#include <modules/evoneuro2/lpds.hpp> // For LPDS.
#include <sferes/misc.hpp> // For misc::rand()
namespace sferes {
	namespace gen {
		SFERES_CLASS(MrfGen){
		public:
		typedef phen::Parameters<gen::EvoFloat<1, Params>, fit::FitDummy<>,
		Params> weight_t;
		typedef phen::Parameters<gen::EvoFloat<4, Params>, fit::FitDummy<>,
		Params> node_label_t;
		typedef nn::PfWSum<weight_t> potential_t;
		typedef nn::AfLpds<node_label_t> activation_t;
		//typedef nn::AfSigmoidNoBias<> activation_t;
				typedef nn::Neuron<potential_t, activation_t> neuron_t;
				typedef nn::Connection<weight_t> connection_t;
				typedef nn::NN<neuron_t, connection_t> nn_t;
				typedef sferes::gen::Dnn<neuron_t, connection_t, Params> genotype_t;

				struct interconnection_t {
					int src_chip_id;
					int src_neuron_id;
					int dest_chip_id;
					int dest_neuron_id;
					connection_t connection;
				};

				MrfGen() {
					// Set the initial number of chips
					_nb_chips = misc::rand<int>(Params::dnn_mrf::initial_min_nb_chips,
							Params::dnn_mrf::initial_max_nb_chips + 1);
					_chips.clear();

					//
					for (size_t i = 0; i < _nb_chips; ++i) {
						genotype_t gen;
						_chips.push_back(gen);
					}
				}

				std::vector<float> _data;
				template<class Archive>
				void serialize(Archive & ar, const unsigned int version) {
					ar & BOOST_SERIALIZATION_NVP(_data);
				}

				// Generate a random genotype
				void random() {

					// Randomize each chip
					for (size_t i = 0; i < _chips.size(); ++i) {
						_chips[i].random();
					}

					// Randomize interconnections (we need at least 2 chips)
					if (_chips.size() >= 2) {

#ifdef CONSTRAINT_PROJECTION_INTERCONNECTIONS
				for (size_t chip_number = 0; chip_number < _chips.size(); ++chip_number) {

					typename nn_t::graph_t graph_src = _chips[chip_number].get_graph();
					int neuron_id = -1;
					BGL_FORALL_VERTICES_T(vertex, graph_src, typename nn_t::graph_t)
					{
						neuron_id++;
						// If this is an input neuron, output neuron or inhib neuron, don't create interconnection from it
						if ((!(graph_src[vertex].get_in() == -1))
								|| (!(graph_src[vertex].get_out() == -1))
								|| (graph_src[vertex].inhib())) {
							continue;
						}
						for (size_t chip_number2 = 0; chip_number2 < _chips.size(); ++chip_number2) {
							// Interconnection is between 2 different chips
							if (chip_number == chip_number2) continue;
							// Random interconnection with another chip
							if (misc::rand<float>() < Params::dnn_mrf::interconn_other_chip_proba) {
								interconnection_t interconnection;
								interconnection.src_chip_id = chip_number;
								interconnection.src_neuron_id = neuron_id;//graph_src[vertex].get_id();
								interconnection.dest_chip_id = chip_number2;
								interconnection.dest_neuron_id = misc::rand<int>(chips(interconnection.dest_chip_id).get_nb_inputs(),
										chips(interconnection.dest_chip_id).get_nb_neurons());;
								weight_t weight;
								weight.random();
								interconnection.connection.set_weight(weight);
								_interconnections.push_back(interconnection);
							}
						}

					}
				}

#else
				// Randomize interconnections
				size_t initial_nb_interconnections = misc::rand<int>(Params::dnn_mrf::initial_min_nb_interconn, Params::dnn_mrf::initial_max_nb_interconn + 1);
				for (size_t interconnections_number = 0; interconnections_number < initial_nb_interconnections; ++interconnections_number) {
					_interconnections.push_back(random_interconnection_nodup());
				}
#endif
				//std::cout << "random->erase" << std::endl;
				//_interconnections.erase(_interconnections.begin());
				//std::cout << "oker" << std::endl;

				//_chips[0].write(std::cout);
				//std::cin.get();
			}

		}

		// Generate a random interconnection and make sure the new
		// interconnection is not a duplicate of a vector already
		// present in _interconnections
		interconnection_t random_interconnection_nodup() {
			interconnection_t interconnection = random_interconnection();
			bool duplicate = false;
			for (size_t interconnections_number = 0; interconnections_number < _interconnections.size(); ++interconnections_number) {
				if (interconnection.src_chip_id == _interconnections[interconnections_number].src_chip_id
						&& interconnection.src_neuron_id == _interconnections[interconnections_number].src_neuron_id
						&& interconnection.dest_chip_id == _interconnections[interconnections_number].dest_chip_id
						&& interconnection.dest_neuron_id == _interconnections[interconnections_number].dest_neuron_id) {
					duplicate = true;
					break;
				}
			}
			if (duplicate) { // If duplicate, generate another random interconnection
				return random_interconnection_nodup();
			}
			else {
				return interconnection;
			}

		}

		// Generate a random interconnection
		interconnection_t random_interconnection() {
			// We must have at least 2 chips to generate an interconnection
			assert(chips().size() >= 2);

			interconnection_t interconnection;
			int src_chip_id = misc::rand<int>(0, _chips.size());
			int src_chip_id_initial = src_chip_id;

#ifdef CONSTRAINT_PROJECTION_INTERCONNECTIONS
				/* TODO
				 while(true) {
				 chips(src_chip_id).get_nb_neurons() -
				 // If we didn't find a right chip, then go to the next chip!
				 src_chip_id = (src_chip_id + 1) % nb_chips();
				 }
				 */
#endif
				interconnection.src_chip_id = src_chip_id;
				interconnection.dest_chip_id = misc::rand<int>(0, _chips.size());
				// Make sure the destination chip ID is different from the source chip ID
				while (interconnection.src_chip_id == interconnection.dest_chip_id) {
					interconnection.dest_chip_id = misc::rand<int>(0, _chips.size());
				}
				assert(interconnection.src_chip_id != interconnection.dest_chip_id);

				interconnection.src_neuron_id = misc::rand<int>(0,
						chips(interconnection.src_chip_id).get_nb_neurons());
				interconnection.dest_neuron_id = misc::rand<int>(0,
						chips(interconnection.dest_chip_id).get_nb_neurons());
				neuron_t neuron_src;
				neuron_t neuron_tgt;
				do {
					interconnection.src_neuron_id = misc::rand<int>(0,
							chips(interconnection.src_chip_id).get_nb_neurons());
					interconnection.dest_neuron_id = misc::rand<int>(0,
							chips(interconnection.dest_chip_id).get_nb_neurons());
					/*
					 neuron_src = chips(interconnection.src_chip_id).get_neuron_directly(interconnection.src_neuron_id);
					 neuron_tgt = chips(interconnection.dest_chip_id).get_neuron_directly(interconnection.dest_neuron_id);
					 neuron_src.get_afparams().develop();
					 neuron_tgt.get_afparams().develop();
					 neuron_src.set_afparams(neuron_src.get_afparams());
					 neuron_tgt.set_afparams(neuron_tgt.get_afparams());
					 std::cout << neuron_src.get_afparams().data(2)<< ";";
					 std::cout << neuron_tgt.get_afparams().data(2)<< ";";
					 std::cout << neuron_src.inhib() << ";";
					 std::cout << neuron_tgt.inhib() << ";";
					 */
				}
				while (false);
				/*
				 //Implement CONSTRAINT_EDGE_ALWAYS_INHIB_AND_EXCIT for interconnections
				 //Note: there is a bug somewhere
				 while (chips(interconnection.src_chip_id).get_neuron_directly(interconnection.src_neuron_id).inhib() == chips(interconnection.dest_chip_id).get_neuron_directly(interconnection.dest_neuron_id).inhib());
				 if ((neuron_src.get_afparams().data(2) > 0.5 != neuron_src.inhib())
				 || (neuron_tgt.get_afparams().data(2) > 0.5 != neuron_tgt.inhib())) {
				 std::cout << "PB6z";
				 }
				 */
				weight_t weight;
				weight.random();
				interconnection.connection.set_weight(weight);
				return interconnection;
			}

			// Mutate an individual
			void mutate() {

				//std::cout << "MUTATE";
				// Mutate all chips
				for (size_t chip_number = 0; chip_number < _chips.size(); ++chip_number) {
					_chips[chip_number].mutate();
				}

				// Mutate interconnections' weights
				for (size_t interconnection_number = 0; interconnection_number < _interconnections.size(); ++interconnection_number) {

					if (misc::rand<float>()
							< Params::dnn_mrf::m_rate_change_interconn) {
						_interconnections[interconnection_number].connection.get_weight().mutate();
					}
				}

				// Add interconnection
				if ((_interconnections.size() < Params::dnn_mrf::max_nb_interconn)
						&& (misc::rand<float>() < Params::dnn_mrf::m_rate_add_interconn)) {
					_interconnections.push_back(random_interconnection_nodup());
				}

				// Delete interconnections
				if ((_interconnections.size() > Params::dnn_mrf::min_nb_interconn)
						&& ((misc::rand<float>() < Params::dnn_mrf::m_rate_del_interconn) && !(_interconnections.empty()))) {
					_interconnections.erase(_interconnections.begin() + misc::rand<int>(0, _interconnections.size()) );
				}

				//std::cout << "u";

				// TODO :
				// * Add constraints
				// •	ca. 35 to 75 chips
				// •  p % of projection neurons (therefore 1 - p is the % of interneurons)
				//         (p=0.8 known in literature (p25 Humphries 2007)) ;
				// •  P(c) -> probability of each projection neuron contacting a given
				//         cluster (P(c) = 0.25 , p25 Humphries 2007 known in literature) ;
			}

			size_t size() const {
				return 5;
			}

			// WARNING: the arguments of the method cross() must be of the same type
			// than the genotype.
			void cross(const MrfGen& o, MrfGen& c1, MrfGen& c2) {
				// Usual crossover
				if (misc::flip_coin()) {
					c1 = *this;
					c2 = o;
				} else {
					c2 = *this;
					c1 = o;
				}
			}

			// This is a debug function. Display a nn properties.
			static void display_nn_properties(nn_t nn, std::string nn_name = "") {

				if (!nn_name.empty()) {
					std::cout << nn_name << " : ";
				}
				std::cout << " inputs=" << nn.get_inputs().size();
				std::cout << " outputs=" << nn.get_outputs().size();
				std::cout << " nb_inputs=" << nn.get_nb_inputs();
				std::cout << " nb_outputs=" << nn.get_nb_outputs();
				std::cout << " connections=" << nn.get_nb_connections();
				std::cout << " neurons=" << nn.get_nb_neurons();
				std::cout << std::endl;
			}

			// Getters
			int nb_chips() {
				return this->_nb_chips;
			}
			std::vector<genotype_t> chips() {
				return this->_chips;
			}
			genotype_t& chips(unsigned int chip_number) {
				return this->_chips[chip_number];
			}
			std::vector<interconnection_t> interconnections() {
				return this->_interconnections;
			}
			interconnection_t& interconnections(
					unsigned int interconnection_number) {
				return this->_interconnections[interconnection_number];
			}

			protected:

			// This is the number of chips of the mRF
			unsigned int _nb_chips;
			// This contain mRF's chips. 1 chip = 1 neural network.
			std::vector<genotype_t> _chips;
			// This contain mRF's chips. 1 chip = 1 neural network.
			std::vector<interconnection_t> _interconnections;
		}
		;
	}
}

#endif

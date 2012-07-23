#ifndef GEN_EVO_NEURO_HPP_
#define GEN_EVO_NEURO_HPP_

#include <modules/nn/gen_dnn.hpp>

namespace sferes
{
  namespace gen
  {
    template<typename Params>
    class EvoNeuro : public DnnT<nn::Neuron<nn::PfT<Params::evo_neuro::neuron_t::pf_t::nb_params>,
					    nn::AfDirectT<std::vector<float>, Params::evo_neuro::neuron_t::af_t::nb_params>, 
					    std::vector<float> >, 
				 nn::ConnectionT<std::vector<float> >, 
				 Params>
    {
    };
  }
}

#endif

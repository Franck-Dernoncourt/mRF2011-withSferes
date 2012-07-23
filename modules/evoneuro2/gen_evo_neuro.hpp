#ifndef GEN_EVO_NEURO_HPP_
#define GEN_EVO_NEURO_HPP_

#include <modules/nn2/gen_dnn.hpp>

namespace sferes
{
  namespace gen
  {
    template<typename LabelNeuron, typename LabelConnection, typename Params>
    class EvoNeuro : public Dnn<nn::Neuron<nn::Pf<>, nn::AfDirect<LabelNeuron> >, 
				nn::Connection<LabelConnection>, 
				Params>
    {
    public:
      void init() { /* nothing */ }
    };
  }
}

#endif

#ifndef PHEN_HYPER_NN_HPP
#define PHEN_HYPER_NN_HPP

#include <map>
#include <sferes/phen/indiv.hpp>
#include <modules/nn/nn.hpp>
#include <modules/nn/io_trait.hpp>

#include "gen_evo_neuro_2.hpp"

namespace sferes
{
  namespace phen
  {
    namespace evo_neuro
    {
      // decorate neurons with their coordinates in the map
      template<typename N>
        class Neuron : public N
      {
        public:
          // generic version
          void set_coords(const std::vector<float>& c) { _coords = c; }
          const std::vector<float>& get_coords() const { return _coords; }
          // 1-d version
          void set_coords(float x) { _coords.clear(); _coords.push_back(x); }
        protected:
          std::vector<float> _coords;
      };
    }

    SFERES_INDIV(EvoNeuro, Indiv)
    {
      public:
        typedef typename evo_neuro::Neuron<typename Params::evo_neuro::neuron_t> neuron_t;
        typedef typename Params::evo_neuro::connection_t connection_t;
        typedef typename nn::NN<neuron_t, connection_t> nn_t;
        typedef typename Gen::nn_t::graph_t g_graph_t;
        typedef typename Gen::nn_t::vertex_desc_t g_v_d_t;
        typedef typename nn_t::vertex_desc_t v_d_t;
        typedef typename std::map<g_v_d_t, std::vector<v_d_t> > map_t;
        static const size_t map_size = Params::evo_neuro::map_size;

        void develop()
        {
          assert(_gen_nb_inputs() == Params::evo_neuro::nb_inputs);
          assert(_gen_nb_outputs() == Params::evo_neuro::nb_outputs);
          _create_neurons();
          _create_connections();
          assert(_nn.get_nb_inputs() == Params::evo_neuro::nb_inputs);
          assert(_nn.get_nb_outputs() == Params::evo_neuro::nb_outputs);
        }
        void show(std::ostream& os) 
        {
          //	  this->_nn.write(os);
          // custom NN
          this->gen().init();
          std::ofstream ofs("/tmp/nn.dot");
          ofs << "digraph G {" << std::endl;
          g_graph_t& g = this->gen().get_graph();
          BGL_FORALL_VERTICES_T(v, g, g_graph_t)
          {
            ofs<<g[v]._id<<"[label=\""<<g[v]._id<<" a:";
            for (size_t i = 0; i < g[v].get_afparams().size(); ++i)
              ofs<<g[v].get_afparams()[i]<<";";
            if (g[v].get_pfparams().size())
            {
              ofs<<" p:";
              for (size_t i = 0; i < g[v].get_pfparams().size(); ++i)
                ofs<<g[v].get_pfparams()[i];
            }
            ofs<<"\"]"<<std::endl;
          }	      	      	      
          BGL_FORALL_EDGES_T(e, g, g_graph_t)
          {
            const std::vector< float >& gen = g[e].get_weight();
            bool type = gen[0]< 0;
            bool f_type = gen[1]> 0;
            float weight_mul = Params::evo_neuro::max_weight;
            float param = (gen[2]+ 1) / 2.0f;
            ofs<<g[source(e, g)]._id<<" -> "<<g[target(e, g)]._id
              <<" [label=\"" << (type ? "1-1 " : "1-all ")
              <<" w="<<weight_mul * param
              <<(!type ? (f_type ? " cst" : " gauss.") :"")
              <<"\""<<(!type ? ",style=bold":"")<<"]"
              <<std::endl;
          }	      	      	      

          ofs << "}" << std::endl;

        }
        nn_t& nn() { return _nn; }
        const nn_t& nn() const { return _nn; }
      protected:		
        nn_t _nn;
        map_t _map;

        void _nb_inputs()
          {
            size_t total=0;
            for (size_t i = 0 ; i < Params::evo_neuro::input_maps_size_size() )
              total+=Params::evo_neuro::input_maps_size(i);
            return total;
          }
        void _nb_outputs()
          {
            size_t total=0;
            for (size_t i = 0 ; i < Params::evo_neuro::output_maps_size_size() )
              total+=Params::evo_neuro::output_maps_size(i);
            return total;
          }

        void _create_neurons()
        {
          g_graph_t &g = this->gen().get_graph();
          size_t k = 0;
          this->_nn.set_nb_inputs(_nb_inputs());
          this->_nn.set_nb_outputs(_nb_outputs());
          BGL_FORALL_VERTICES_T(v, g, g_graph_t)
          {
            if (this->gen().is_input(v))
            {
              std::vector<v_d_t> inputs,all_inputs;
              /* we first need to identify the correct input */
              all_inputs = this->gen().get_inputs();
              size_t input_index = (size_t) distance( all_inputs.begin(), std::find( all_inputs.begin(), all_inputs.end(), v));
              assert(input_index>=0);
              assert(input_index<Params::dnn::nb_inputs);
              /* add all relevant inputs to a table */
              for (size_t i = _gen_nb_inputs(input_index) ;i<  _gen_nb_inputs(input_index+1);++i)
                inputs.push_back(_nn.get_input(i));
              _map[v]=inputs;
              BOOST_FOREACH(g_v_d_t x, inputs)
              {
                _nn.get_graph()[x].set_pfparams(g[v].get_pfparams());
                _nn.get_graph()[x].set_afparams(g[v].get_afparams());
              }
            }
            else if (this->gen().is_output(v))
            {
              std::vector<v_d_t> outputs,all_outputs;
              /* we first need to identify the correct output */
              all_outputs = this->gen().get_outputs();
              size_t output_index = (size_t) distance( all_outputs.begin(), std::find( all_outputs.begin(), all_outputs.end(), v));
              assert(output_index>=0);
              /* add all relevant outputs to a table */
              for (size_t i = _gen_nb_outputs(output_index) ;i<  _gen_nb_outputs(output_index+1);++i)
                outputs.push_back(_nn.get_output(i));
              _map[v]=outputs;
              BOOST_FOREACH(g_v_d_t x, outputs)
              {
                _nn.get_graph()[x].set_pfparams(g[v].get_pfparams());
                _nn.get_graph()[x].set_afparams(g[v].get_afparams());
              }
            }
            else
            {
              std::vector<v_d_t> vect;
              assert(g[v].get_af().get_params().size()==1);
              size_t s = g[v].get_af().get_params()[0] > 0.5*Params::dnn::max_weight ? 1 : map_size;
              for (size_t i = 0; i < s; ++i)
              {
                v_d_t n = _nn.add_neuron(boost::lexical_cast<std::string>(k) + "." +
                    boost::lexical_cast<std::string>(i),
                    g[v].get_pfparams(), 
                    g[v].get_afparams());
                _nn.get_graph()[n].set_coords((float) i / s);
                vect.push_back(n);		  
              }
              _map[v] = vect;
            }
            ++k;
          }
        }
        // connections
        // 1 -> 1
        // 1 -> all, same weight
        // 1 -> sub-set, weight distributed with a function
        // all -> 1, weight with a function
        void _create_connections()
        {
          g_graph_t &g = this->gen().get_graph();
          BGL_FORALL_EDGES_T(e, g, g_graph_t)
          {
            const std::vector<v_d_t>& src_map = _map[boost::source(e, g)];
            const std::vector<v_d_t>& tgt_map = _map[boost::target(e, g)]; 
            assert(src_map.size() == tgt_map.size() || src_map.size() == 1 || tgt_map.size() == 1);
            const std::vector<float>& gen = g[e].get_weight();
            assert(gen.size() == 3);
            bool type = gen[0] < 0;
            float weight_mul = Params::evo_neuro::max_weight;
            float param = (gen[2]+Params::dnn::max_weight)/2.0f;//sigma or constant or
            assert(param>=0);
            if (type && src_map.size() != 1 && tgt_map.size() != 1) // 1-1 connection
            {
              for (size_t i = 0; i < src_map.size(); ++i)
                _nn.add_connection(src_map[i], tgt_map[i],std::make_pair((float)(param * weight_mul),0.0f));
            }
            else // 1 - all
            {
              bool f_type = gen[1] > 0;
              for (size_t i = 0; i < src_map.size(); ++i)
                for (size_t j = 0; j < tgt_map.size(); ++j)
                {
                  if (f_type)
                    _nn.add_connection(src_map[i], tgt_map[j],std::make_pair((float)(param * weight_mul),0.0f));
                  else
                  {
                    float d = ((int)i - (int)j) / (float) src_map.size();
                    _nn.add_connection(src_map[i], 
                        tgt_map[j], 
                        std::make_pair((float)(weight_mul * exp(-d * d / (param * param))),0.0f));
                  }

                }
            }
          }
        }
        /* returns the sum of input map sizes inferior or equalt to last input */
        size_t _gen_nb_inputs(size_t last_input = Params::dnn::nb_inputs)
        {
          size_t sum_nb_inputs=0;
          assert(last_input<=Params::dnn::nb_inputs);
          assert(Params::evo_neuro::input_maps_size_size()==Params::dnn::nb_inputs);
          assert(this->gen().get_nb_inputs() == Params::dnn::nb_inputs);
          for (size_t i = 0 ; i< last_input ;++i)
          {
            sum_nb_inputs+=Params::evo_neuro::input_maps_size(i);
          }
          return sum_nb_inputs;
        }
        /* returns the sum of input map sizes inferior or equalt to last output */
        size_t _gen_nb_outputs(size_t last_output = Params::dnn::nb_outputs)
        {
          size_t sum_nb_outputs(0);
          assert(last_output<=Params::dnn::nb_outputs);
          assert(Params::evo_neuro::output_maps_size_size()==Params::dnn::nb_outputs);
          assert(this->gen().get_nb_outputs() == Params::dnn::nb_outputs);
          for (size_t i = 0 ; i< last_output ;++i)
          {
            sum_nb_outputs+=Params::evo_neuro::output_maps_size(i);
          }
          return sum_nb_outputs;
        }

    };
  }
}


#endif

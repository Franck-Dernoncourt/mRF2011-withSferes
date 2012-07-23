#ifndef PHEN_EVONEURO_HPP
#define PHEN_EVONEURO_HPP

#include <map>
#include <sferes/phen/indiv.hpp>
#include "gen_evo_neuro.hpp"

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
	  assert(this->_gen.get_nb_inputs() == 1);
	  assert(this->_gen.get_nb_outputs() == 1);
	  _create_neurons();
	  _create_connections();
	  assert(_nn.get_nb_inputs() == map_size);
	  assert(_nn.get_nb_outputs() == map_size);
	}
	void show(std::ostream& ofs) 
	{
	  this->gen().init();
	  ofs << "digraph G {" << std::endl;
	  g_graph_t& g = this->gen().get_graph();
	  BGL_FORALL_VERTICES_T(v, g, g_graph_t)
	    {
	      ofs<<g[v]._id<<"[label=\""<<g[v]._id<<" a:";
	      for (size_t i = 0; i < g[v].get_afparams().size(); ++i)
	  	ofs<<g[v].get_afparams().data(i)<<";";
	      if (g[v].get_pfparams().size())
	  	{
	  	  ofs<<" p:";
	  	  for (size_t i = 0; i < g[v].get_pfparams().size(); ++i)
	  	    ofs<<g[v].get_pfparams().data(i);
	  	}
	      ofs<<"\"]"<<std::endl;
	    }	      	      	      
	  BGL_FORALL_EDGES_T(e, g, g_graph_t)
	    {
	      const typename Gen::weight_t& gen = g[e].get_weight();
	      bool type = gen.data(0) < 0;
	      bool f_type = gen.data(1) > 0;
	      float weight_mul = Params::evo_neuro::max_weight;
	      float param = (gen.data(2) + 1) / 2.0f;
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

	void _create_neurons()
	{
	  g_graph_t &g = this->gen().get_graph();
	  size_t k = 0;
	  assert(this->gen().get_nb_inputs() == 1);
	  assert(this->gen().get_nb_outputs() == 1);
	  BGL_FORALL_VERTICES_T(v, g, g_graph_t)
	    {
        /*added to avoid empy params (transfer from gen to phen) */
        g[v].get_pfparams().develop();
        g[v].get_afparams().develop();
	       if (this->gen().is_input(v))
		{
		  this->_nn.set_nb_inputs(map_size);
		  _map[v] = this->_nn.get_inputs();
		  BOOST_FOREACH(g_v_d_t x, this->_nn.get_inputs())
		    {
		      //@Magic (yes, a magic cast is happening here !)
		      _nn.get_graph()[x].set_pfparams(g[v].get_pfparams());
		      _nn.get_graph()[x].set_afparams(g[v].get_afparams());
		    }
		}
	      else if (this->gen().is_output(v))
		{
		  this->_nn.set_nb_outputs(map_size);
		  _map[v] = this->_nn.get_outputs();		  
		  BOOST_FOREACH(g_v_d_t x, this->_nn.get_outputs())
		    {
		      //@Magic
		      _nn.get_graph()[x].set_pfparams(g[v].get_pfparams());
		      _nn.get_graph()[x].set_afparams(g[v].get_afparams());
		    }
		}
	      else
		{
		  std::vector<v_d_t> vect;
		  size_t s = g[v].get_af().get_params().data(3) > 0.75 ? 1 : map_size;
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
        /*added to avoid empy params (transfer from gen to phen) */
	      typename Gen::weight_t& gen = g[e].get_weight();
        gen.develop();
	      assert(gen.size() == 3);
	      bool type = gen.data(0) < 0;
	      float weight_mul = Params::evo_neuro::max_weight;
	      float param = (gen.data(2) + 1) / 2.0f;//sigma or constant or
				 		//weight -> positive
	      assert(param >= 0);
	      if (type && src_map.size() != 1 && tgt_map.size() != 1) // 1-1 connection
		{
		  for (size_t i = 0; i < src_map.size(); ++i)
		    _nn.add_connection(src_map[i], tgt_map[i], param * weight_mul);
		}
	      else // 1 - all
		{
		  bool f_type = gen.data(1) > 0;
		  for (size_t i = 0; i < src_map.size(); ++i)
		    for (size_t j = 0; j < tgt_map.size(); ++j)
		      {
			if (f_type)
			    _nn.add_connection(src_map[i], tgt_map[j], param * weight_mul);
			else
			  {
			    float d = ((int)i - (int)j) / (float) src_map.size();
			    _nn.add_connection(src_map[i], 
					       tgt_map[j], 
					       weight_mul * exp(-d * d / (param * param)));
			  }
			
		      }
		}
	    }
	}
	
      };
  }
}


#endif

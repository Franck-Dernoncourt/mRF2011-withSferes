#ifndef PF_LPDS_HPP_
#define PF_LPDS_HPP_

#include <modules/nn2/params.hpp>

namespace nn {
	//see Girard et al., Contracting model of the basal ganglia (2008)
	template<typename Params = nn::params::Vectorf<4> >
	class AfLpds: public Af<Params> {
		public:
		// WARNING : not standard value
			SFERES_ARRAY(float, tau, 5e-3, 10e-3, 20e-3, 40e-3);
			SFERES_ARRAY(float, threshold, -5, -2.5, 0, 2.5, 5);
			static const float dt = 1e-3;
			typedef Params params_t;

			AfLpds() :
				_a(0) {
			}

			// WARNING : modified function for the mRF experiment
			void set_params(const params_t& v) {
				assert(v.size() == 4);
				this->_params = v;
#ifdef LPDS_FIXED_TAU
				_tau = 5e-3;
#else
				float t = this->_params.data(0) == 1 ? 1 - 1e-5 : this->_params.data(0);
				_tau = tau((int) (t * tau_size()));
#endif
#ifdef LPDS_FIXED_THRESHOLD
				_threshold = 0;
#else
				float t2 = this->_params.data(1) == 1 ? 1 - 1e-5 : this->_params.data(1);
				_threshold = threshold((int) (t2 * threshold_size()));
#endif

				_inhib = this->_params.data(2) > 0.5;
				_alone = this->_params.data(3) > 0.5;
				assert(_tau != 0);
			}

			float operator()(float p) {
				assert(_tau != 0);
				//_threshold = 0; // DEBUG
				float p_temp = p;
				_a = std::max(0.0f,
						std::min(1.0f, _a + (p - _a + _threshold) * dt / _tau));
				float b = (_inhib ? -_a : _a);
				//if (_inhib) std::cout << "I" << b;
				//else std::cout << "E" << b;
				return b;
			}

			// Setters
			// WARNING : not a standard function
			void set_inhib(bool inhib){
				//std::cout <<"OKH";
				_inhib = inhib;
				if (inhib) {
					this->_params.set_data(2, 0.75);
				} else {
					this->_params.set_data(2, 0.25);
				}
			}

			// Getters
			bool alone() const {
				return _alone;
			}
			// WARNING : not a standard function
			bool inhib() const {
				return _inhib;
			}
			// WARNING : not a standard function
			float tau() const {
				//std::cout <<"OKH";
				return _tau;
			}
			// WARNING : not a standard function
			float threshold() const {
				//std::cout <<"OKH";
				return _threshold;
			}

		protected:
			float _a;
			float _tau;
			float _threshold;
			bool _inhib;
			bool _alone;
	};

}

#endif

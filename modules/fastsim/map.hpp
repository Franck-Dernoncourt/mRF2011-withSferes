/*
 ** map.hh
 ** Login : <mouret@asuncion.lip6.fr>
 ** Started on  Mon Jan 14 16:38:00 2008 Jean-Baptiste MOURET
 ** $Id$
 **
 ** Copyright (C) 2008 Jean-Baptiste MOURET
 ** This program is free software; you can redistribute it and/or modify
 ** it under the terms of the GNU General Public License as published by
 ** the Free Software Foundation; either version 2 of the License, or
 ** (at your option) any later version.
 **
 ** This program is distributed in the hope that it will be useful,
 ** but WITHOUT ANY WARRANTY; without even the implied warranty of
 ** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 ** GNU General Public License for more details.
 **
 ** You should have received a copy of the GNU General Public License
 ** along with this program; if not, write to the Free Software
 ** Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#ifndef   	MAP_HH_
# define   	MAP_HH_

#include <vector>
#include <fstream>
#include <cmath>
#include <cassert>
#include <boost/shared_ptr.hpp>
#include "misc.hpp"
#include "goal.hpp"
#include "illuminated_switch.hpp"

namespace fastsim {

	// 0, 0 is top left
	// map must be a square
	class Map {
		public:
			enum status_t {
				// Not standard
				obstacle = 0, black = 1, grey = 127, white = 255, free = 128
			};
			Map(const char* fname, float real_w) :
				_real_w(real_w), _real_h(real_w) {
				try {
					//std::cout << "go";
					//_w = 0
					//_h = 0;
					_read_file(fname, false);
				} catch (Exception e) {
					std::cerr << "error : " << e.get_msg() << std::endl;
				}
				//std::cout << "_w=" << _w;
				//std::cout << "_h=" << _h;
				assert(_w == _h);
				_fx = _w / _real_w;
			}
			// copy ONLY the picture (no goal, illuminated switches, etc)
			// REASON:
			// we want to avoid reading the data data but we don't want to
			// require good copy constructors for all objects
			Map(const Map& o) :
				_data(o._data), _w(o._w), _h(o._h), _real_w(o._real_w),
						_real_h(o._real_h), _fx(o._fx) {
				// check that we are copying an empty map
				assert(_goals.size() == 0);
				assert(_illuminated_switches.size() == 0);
			}
			status_t get_pixel(unsigned x, unsigned y) const {
				return (y * _h + x >= 0 && y * _h + x < _data.size()) ? _data[y * _h
						+ x] : _data[_data.size() - 1];
			}
			void set_pixel(unsigned x, unsigned y, status_t v) {
				if (y * _h + x >= 0 && y * _h + x < _data.size())
					_data[y * _h + x] = v;

			}
			status_t get_real(float x, float y) const {
				return get_pixel(real_to_pixel(x), real_to_pixel(y));
			}
			int real_to_pixel(float x) const {
				return (unsigned) roundf(x * _fx);
			}
			float pixel_to_real(unsigned i) const {
				return i / _fx;
			}
			float get_real_w() const {
				return _real_w;
			}
			float get_real_h() const {
				return _real_h;
			}
			unsigned get_pixel_w() const {
				return _w;
			}
			unsigned get_pixel_h() const {
				return _h;
			}
			// see
			// http://lifc.univ-fcomte.fr/~dedu/projects/bresenham/index.html
			// in PIXEL coordinates
			bool check_inter_pixel(int x1, int y1, int x2, int y2, int& x_res,
					int& y_res) const;
			// in world coordinates
			bool check_inter_real(float x1, float y1, float x2, float y2,
					float& x_res, float &y_res) const {
				int px = 0, py = 0;
				bool res = check_inter_pixel(real_to_pixel(x1), real_to_pixel(y1),
						real_to_pixel(x2), real_to_pixel(y2), px, py);
				x_res = pixel_to_real(px);
				y_res = pixel_to_real(py);
				return res;
			}

			const std::vector<Goal>& get_goals() const {
				return _goals;
			}
			void add_goal(const Goal& g) {
				_goals.push_back(g);
			}

			typedef boost::shared_ptr<IlluminatedSwitch> ill_sw_t;
			void add_illuminated_switch(ill_sw_t is) {
				_illuminated_switches.push_back(is);
			}
			const std::vector<ill_sw_t>& get_illuminated_switches() const {
				return _illuminated_switches;
			}
			ill_sw_t get_illuminated_switch_by_color(int color) {
				for (size_t i = 0; i < _illuminated_switches.size(); ++i)
					if (_illuminated_switches[i]->get_color() == color)
						return _illuminated_switches[i];
				assert(0);
				return ill_sw_t();
			}
			void update(const Posture& robot_pos) {
				for (size_t i = 0; i < _illuminated_switches.size(); ++i)
					_illuminated_switches[i]->try_to_activate(robot_pos);

			}
			void terrain_switch(const std::string &fname) {
				_read_file(fname, false);
			}
			void terrain_switch_only_obstacles(const std::string &fname) {
				_read_file(fname, true);
						}


			// Draws a rectangle with (x,y) the upper left point and (lx,ly) the size
			void draw_rect(int x, int y, int lx, int ly);
			void draw_rect(int x, int y, int lx, int ly, status_t color);

		protected:


			//void _read_file(const std::string &fname);
			void _read_file(const std::string &fname, bool only_obstacles);
			bool _get_bit(char c, int i) const {
				return (c & (1 << (7 - i)));
			}
			//
			bool _try_pixel(int x, int y) const;
			//
			std::vector<status_t> _data;
			std::vector<Goal> _goals;
			std::vector<ill_sw_t> _illuminated_switches;
			//
			int _w, _h;
			float _real_w, _real_h;
			float _fx;
	};
}

#endif	    /* !MAP_HH_ */

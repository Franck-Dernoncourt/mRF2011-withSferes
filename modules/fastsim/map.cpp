/*
** map.cc
** Login : <mouret@asuncion.lip6.fr>
** Started on  Mon Jan 14 16:39:08 2008 Jean-Baptiste MOURET
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

#include <iostream>
#include "map.hpp"

namespace fastsim
{
  void Map::_read_file(const std::string& fname, bool only_obstacles = false)
  {
  	//std::cout << "go1";
    std::string str;
    std::ifstream ifs(fname.c_str());
    if (!ifs.good())
      throw Exception(std::string("cannot open map :") + fname);
    ifs >> str >> _w >> _h;
    // std::cout << "go2";
    //std::cout << "str=" << str;
    if (str != "P4")
      throw Exception("wrong file type for map");
    if (!only_obstacles) {
    	_data.resize(_w * _h);
    }
    //std::cout << "go3";
    //std::cout << "_w=" << _w;
    //std::cout << "_h=" << _h;
    if (_w % 8 != 0) {
    	std::cout << "boom";
    	 throw Exception("wrong map size");
    }
    // std::cout << "go4";
    int k = _w * _h / 8;
    //std::cout << "go5";
    //std::cout << "k="<<k;
    char buffer[k];
    //std::cout << "go6";
    ifs.read((char*)buffer, k);
    for (int i = 0; i < k - 1; ++i)
      for (int j = 0; j < 8; ++j) {
      	if (only_obstacles) {
      		if (_get_bit(buffer[i + 1], j)) {
      			_data[i * 8 + j] = obstacle;
      		}
      	} else {
      		_data[i * 8 + j] = _get_bit(buffer[i + 1], j) ? obstacle : free;
      	}
      }

    //std::cout << "_w=" << _w;
    //std::cout << "_h=" << _h;
  }



  
  bool Map :: _try_pixel(int x, int y) const
  {
    if (x >= 0 && y >= 0 
	&& x < get_pixel_w() 
	&& y < get_pixel_h() 
	// && get_pixel(x, y) == free)
	&& get_pixel(x, y) != obstacle)
      return false;
    else
      return true;
  }
  
  // see
  // http://lifc.univ-fcomte.fr/~dedu/projects/bresenham/index.html
  // In PIXEL coordinates
  bool Map :: check_inter_pixel(int y1, int x1,
				int y2, int x2,
				int& y_res, int& x_res) const
  {
    int i;               // loop counter
    int ystep, xstep;    // the step on y and x axis
    int error;           // the error accumulated during the increment
    int errorprev;       // *vision the previous value of the error variable
    int y = y1, x = x1;  // the line points
    int ddy, ddx;        // compulsory variables: the double values of dy and dx
    int dx = x2 - x1;
    int dy = y2 - y1;
    bool inter = _try_pixel(y1, x1);
    if (dy < 0) { ystep = -1; dy = -dy; } else ystep = 1;
    if (dx < 0) { xstep = -1; dx = -dx; } else xstep = 1;
    ddy = dy * 2;
    ddx = dx * 2;
    if (ddx >= ddy) // first octant (0 <= slope <= 1)
      {  
	errorprev = error = dx;
	for (i = 0 ; i < dx ; i++)
	  {  // do not use the first point (already done)
	    x += xstep;
	    error += ddy;
	    if (error > ddx)
	      {
		y += ystep;
		error -= ddx;
		if (error + errorprev < ddx)  // bottom square also
		  inter = inter || _try_pixel(y - ystep, x);
		else if (error + errorprev > ddx)  // left square also
		  inter = inter || _try_pixel(y, x - xstep);
		else // corner: bottom and left squares also
		  {
		    inter = inter || _try_pixel(y - ystep, x);
		    inter = inter || _try_pixel(y, x - xstep);
		  }
	      }
	    inter = inter || _try_pixel(y, x);
	    errorprev = error;
	    if (inter)
	      {
		x_res = x;
		y_res = y;
		return true;
	      }
	  }
      }
    else
      {  // the same as above
	errorprev = error = dy;
	for (i = 0 ; i < dy ; i++)
	  {
	    y += ystep;
	    error += ddx;
	    if (error > ddy){
	      x += xstep;
	      error -= ddy;
	      if (error + errorprev < ddy)
		inter = inter || _try_pixel(y, x - xstep);
	      else if (error + errorprev > ddy)
		inter = inter || _try_pixel(y - ystep, x);
	      else
		{
		  inter = inter || _try_pixel(y, x - xstep);
		  inter = inter || _try_pixel(y -  ystep, x);
		}
	    }
	    inter = inter || _try_pixel(y, x);
	    errorprev = error;
	    if (inter)
	      {
		x_res = x;
		y_res = y;
		return true;
	      }
	  }
      }
    return false;
  }

    // Draws a rectangle with (x,y) the upper left point and (lx,ly) the size
  void Map:: draw_rect(int x, int y, int lx, int ly) {
    int i,j;

    for (i=0;i<lx;i++)
      for (j=0;j<ly;j++) {
	if ((x+i) >= 0 && (y+j) >= 0 
	    && (x+i) < get_pixel_w() 
	    && (y+j) < get_pixel_h())
	  set_pixel(x+i,y+j,obstacle);
      }
  }
  void Map:: draw_rect(int x, int y, int lx, int ly, status_t color) {
      int i,j;

      for (i=0;i<lx;i++)
        for (j=0;j<ly;j++) {
  	if ((x+i) >= 0 && (y+j) >= 0
  	    && (x+i) < get_pixel_w()
  	    && (y+j) < get_pixel_h())
  	  set_pixel(x+i,y+j,color);
        }
    }

}

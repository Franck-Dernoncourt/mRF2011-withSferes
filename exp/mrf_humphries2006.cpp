// Original authors: Félix Grèzes & Jean Bellot (March 2011)

#ifndef MRF_HUMPHRIES2006_CPP
#define MRF_HUMPHRIES2006_CPP

#include <vector>
#include <iostream>
#include "mrf_humphries2006.hpp"

mrf::mrf(std::vector<float>* dat) :
	i(6, 0.0), c(6, 0.0) {

	//Creation des neurones c et i :
	/*
	 std::vector<float> c;

	 for(unsigned int n = 0; n < 6; n++)
	 {
	 c.push_back(0.0);
	 }

	 printf("Dans l'initialisation i est de taille : %d \n", c.size());

	 std::vector<float> i;

	 for(unsigned int n = 0; n < 6; n++)
	 {
	 i.push_back(0.0);
	 }
	 */

	std::vector<float> wtemp;
	std::vector<float> atemp;
	std::vector<float> ctemp;

	//Creation de W a partir du genotype
	// 8 is the number of inputs
	for (unsigned n = 0; n < 48; n++) {
		if (n % 8 == 0 and n != 0) {
			W.push_back(wtemp);
			wtemp.clear();
		}
		wtemp.push_back(dat->at(n) * 1.5 - 0.5);
	}
	W.push_back(wtemp);

	//Creation de A a partir du genotype
	//On a en moyenne 30.625 connections entre neurones de projection de deux clusters
	unsigned int lig = 0;
	unsigned int col = 0;
	for (unsigned int n = 0; n < 30; n++) {
		if (col % 5 == 0 and n != 0) {
			A.push_back(atemp);
			atemp.clear();
			lig++;
			col = 0;
		}
		if (lig == col) {
			atemp.push_back(0.0);
		}
		atemp.push_back(dat->at(n + 48) * 3 * 30.625);
		col++;
	}

	A.push_back(atemp);

	//Creation de C a partir du genotype
	//On a en moyenne 13.125 connections entre neurones de projection de deux clusters
	lig = 0;
	col = 0;
	for (unsigned int n = 0; n < 30; n++) {
		if (col % 5 == 0 and n != 0) {
			C.push_back(ctemp);
			ctemp.clear();
			lig++;
			col = 0;
		}
		if (lig == col) {
			ctemp.push_back(0.0);
		}
		ctemp.push_back(dat->at(n + 78) * 3 * 13.125);
		col++;
	}

	C.push_back(ctemp);

	//Creation de b a partir du genotype
	//52.5:  Nombre moyen de connections entre neurones de projection d'un cluster et les neurones inhibiteurs de ce mÃªme cluster
	for (unsigned int n = 0; n < 6; n++) {
		b.push_back(dat->at(n + 108) * 3 * 52.5);
	}

	//Creation de d a partir du genotype
	//22.5: Nombre moyen de connections entre neurones inhibiteurs d'un cluster et les neurones inhibiteurs de ce mÃªme cluster
	for (unsigned int n = 0; n < 6; n++) {
		d.push_back(dat->at(n + 114) * 3 * 22.5);
	}

	//Ne = nombre de connections exitatrice
	Ne = 0;
	for (unsigned int it = 0; it < 6; it++) {
		for (unsigned int j = 0; j < 6; j++) {
			Ne += A[it][j] + C[it][j];
		}
	}

	//Ni = nombre de connections inhibitrices
	Ni = 0;
	for (unsigned int it = 0; it < 6; it++) {
		Ni += b[it] + d[it];
	}
}

//Calcul les uk a partir des entrees (uk = somme(Wki * entreei)) 
std::vector<float> mrf::calcul_u(std::vector<float>* entree) {
	std::vector<float> u;
	float uk = 0;
	// 6 clusters
	for (unsigned int k = 0; k < 6; k++) {

		uk = 0;
		//std::cout << "entree->size()=" << entree->size();
		//std::cout << "W.size()=" << W.size();
		for (unsigned int n = 0; n < entree->size(); n++) {
			//uk = uk + W.at(n).at(k) * entree->at(n);
			uk = uk + W.at(k).at(n) * entree->at(n);
		}
		if (uk < -0.5) {
			u.push_back(-0.5);
			continue;
		}
		if (uk > 1) {
			u.push_back(1);
			continue;
		}
		u.push_back(uk);
	}
	//std::cout << "ok";
	return u;
}

///////////////////////////////////////////////////
void mrf::update(std::vector<float>* entree, float dt) {
	std::vector<float> u = calcul_u(entree);

	float we = 0.2;
	float wi = -we * (Ne / Ni);

	std::vector<float> dc(6, 0.0);
	for (unsigned int k = 0; k < 6; k++) {
		for (unsigned int j = 0; j < 6; j++) {
			dc[k] += A[j][k] * c[j];
		}
		dc[k] = we * dc[k] + wi * b[k] * i[k] + u[k];
		if (dc[k] < 0) {
			dc[k] = 0;
		}
		if (dc[k] > 1) {
			dc[k] = 1;
		}

		dc[k] = (-c[k] + dc[k]) * (dt / 0.005);
	}

	//Faire pareil pour ik abruti !!
	std::vector<float> di(6, 0.0);
	for (unsigned int k = 0; k < 6; k++) {
		for (unsigned int j = 0; j < 6; j++) {
			di[k] += C[j][k] * c[j];
		}
		//Ici 15 represente le nombre de neurones inhibiteurs dans un cluster :
		//lamdas = 0
		di[k] = we * di[k] + wi * d[k] * (i[k] - i[k] / 15);
		if (di[k] < 0) {
			di[k] = 0;
		}
		if (di[k] > 1) {
			di[k] = 1;
		}

		di[k] = (-i[k] + di[k]) * (dt / 0.005);
	}

	for (unsigned int k = 0; k < 6; k++) {
		i[k] = i[k] + di[k];
	}

	for (unsigned int k = 0; k < 6; k++) {
		c[k] = c[k] + dc[k];
	}
}

//voir article : thetaL = 0.1 et thetaU = 0.8
float fonction_m_triangle(float x) {
	if (x < 0.1) {
		return 0;
	}
	//f(x) = 10/7 x - 1/7
	if (x < 0.8) {
		return (1.42857143 * x - 0.142857143);
	}
	return 1;
}

float fonction_m_duale(float x) {
	if (x < 0.1) {
		return 0;
	}

	//f(x) = 20/7 x - 2/7
	if (x < ((0.1 + 0.8) / 2)) {
		return (2.857142857 * x - 0.285714286);
	}

	//f(x) = -20/7 x + 16/7
	if (x < 0.8) {
		return (2.857142857 * x + 2.285714286);
	}

	return 1;

}

std::vector<float> mrf::get_sk_duale() {
	std::vector<float> sk;
	for (unsigned int it = 0; it < 6; it++) {
		sk.push_back(fonction_m_duale(c[it]));
	}
	return sk;
}

std::vector<float> mrf::get_sk_triangle() {
	std::vector<float> sk;
	for (unsigned int it = 0; it < 6; it++) {
		sk.push_back(fonction_m_triangle(c[it]));
	}
	return sk;
}

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
// DEFINITION DES GETTEURS
std::vector<std::vector<float> > mrf::getW() {
	return W;
}

std::vector<std::vector<float> > mrf::getC() {
	return C;
}

std::vector<std::vector<float> > mrf::getA() {
	return A;
}

std::vector<float> mrf::get_b() {
	return b;
}

std::vector<float> mrf::get_d() {
	return d;
}

std::vector<float> mrf::get_i() {
	return i;
}

std::vector<float> mrf::get_c() {
	return c;
}

float mrf::get_Ni() {
	return Ni;
}
float mrf::get_Ne() {
	return Ne;
}

#endif

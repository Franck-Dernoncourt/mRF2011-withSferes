// Original authors: Félix Grèzes & Jean Bellot (March 2011)

#ifndef MRF_HUMPHRIES2006_HPP
#define MRF_HUMPHRIES2006_HPP

#include <vector>

class mrf{
	
  public :
	//Constructeur
	mrf(std::vector<float>* dat);
	
	//Met a jour les neurones de projections et d'inhibitions
	//a partir des entrees de la mRF
	void update(std::vector<float>* entree, float dt);
	
	std::vector<float> calcul_u(std::vector<float>* entree);
	
	//fonction renvoyant un vecteur definissant les actions a effectuer
	//m est ici la fonction triangle
	std::vector<float> get_sk_triangle();
	
	//fonction renvoyant un vecteur definissant les actions a effectuer
	//m est ici la fonction duale
	std::vector<float> get_sk_duale();
	
	//Renvoi les poids sur les variables d'entrees
	std::vector<std::vector<float> > getW();
	
	//Renvoi les connections entre les neurones de projections
	std::vector<std::vector<float> > getC();
	
	//Renvoi les connections entre les neurones de projections
	//et les neurones d'inhibitions
	std::vector<std::vector<float> > getA();
	
	//Renvoi les connections entre les neurones inhibiteur et les neurones de projections
	//d'un meme cluster
	std::vector<float> get_b();
	
	//Renvoi les connections entre neurones inhibiteur d'un meme cluster
	std::vector<float> get_d();
	
	//Renvoi l'activite courante des neurones inhibiteur de chaque cluster
	std::vector<float> get_i();
	
	//Renvoi l'activite courante des neurones de projection de chaque cluster
	std::vector<float> get_c();
	
	float get_Ni();
	float get_Ne();

	
	

  protected :
  
	std::vector<std::vector<float> > W;
	std::vector<std::vector<float> > A;
	std::vector<std::vector<float> > C;
	std::vector<float> b;
	std::vector<float> d;
	std::vector<float> i;
	std::vector<float> c;
	float Ne;
	float Ni;
	
};

#endif


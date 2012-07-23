// The file logs a few statistics of the evolution

#ifndef ALL_INDIV_HPP
#define ALL_INDIV_HPP

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <iostream>

namespace sferes {
namespace stat {
SFERES_STAT(AllIndiv, Stat)
{
public:

	typedef std::vector<boost::shared_ptr<Phen> > pop_t;

	// Examines the population (via the EA) to update the statistics
	template<typename E>
	void refresh(const E& ea)
	{
		assert(!ea.pop().empty());
		// takes all the population
		_pop = ea.pop();
		// Sorts by objective scores (descending order)
		parallel::sort(_pop.begin(), _pop.end(), fit::compare_objs_lex());
		this->_create_log_file(ea, "all_phenotypes.dat");
		if (ea.dump_enabled()) {
			show_all(*(this->_log_file), ea.gen());
		}
	}

	// writes the content of the statistics in the stream; k is the number
	// of the individual in the statistics (e.g. in a Pareto front, the
	// statistics contains many individuals)
	void show_all(std::ostream& os, size_t gen = 0) const
	{
		for (unsigned i = 0; i < _pop.size(); ++i) {
			// gen is the generation number,
			// i the rank of the individual in the generation
			os << gen << "," << i << ",";
			// save all objective scores
			for (unsigned j = 0; j < _pop[i]->fit().objs().size(); ++j) {
				os << _pop[i]->fit().obj(j) << ",";
			}
			// call the method show() in the phenotype object, which write
			// the phenotype data in the stream os (here the log file)
			_pop[i]->show(os);
			os << std::endl;;
		}
	}

protected:
	pop_t _pop;
};

SFERES_STAT(AllIndivDataStats, Stat)
{
public:

	typedef std::vector<boost::shared_ptr<Phen> > pop_t;

	// Examines the population (via the EA) to update the statistics
	template<typename E>
	void refresh(const E& ea)
	{
		assert(!ea.pop().empty());
		// takes all the population
		_pop = ea.pop();
		// Sorts by objective scores (descending order)
		parallel::sort(_pop.begin(), _pop.end(), fit::compare_objs_lex());
		this->_create_log_file(ea, "all_phenotypes_data_stats.dat");
		if (ea.dump_enabled()) {
			show_all(*(this->_log_file), ea.gen());
		}
	}

	// writes the content of the statistics in the stream; k is the number
	// of the individual in the statistics (e.g. in a Pareto front, the
	// statistics contains many individuals)
	void show_all(std::ostream& os, size_t gen = 0) const
	{
		for (unsigned i = 0; i < _pop.size(); ++i) {
			// gen is the generation number,
			// i the rank of the individual in the generation
			os << gen << "," << i << ",";
			// save all objective scores
			for (unsigned j = 0; j < _pop[i]->fit().objs().size(); ++j) {
				os << _pop[i]->fit().obj(j) << ",";
			}
			// call the method show() in the phenotype object, which write
			// the phenotype data in the stream os (here the log file)
			_pop[i]->show_data_stats(os);
			os << std::endl;;
		}
	}

protected:
	pop_t _pop;
};

SFERES_STAT(AllIndivRealDataStats, Stat)
{
public:

	typedef std::vector<boost::shared_ptr<Phen> > pop_t;

	// Examines the population (via the EA) to update the statistics
	template<typename E>
	void refresh(const E& ea)
	{
		assert(!ea.pop().empty());
		// takes all the population
		_pop = ea.pop();
		// Sorts by objective scores (descending order)
		parallel::sort(_pop.begin(), _pop.end(), fit::compare_objs_lex());
		this->_create_log_file(ea, "all_phenotypes_real_data_stats.dat");
		if (ea.dump_enabled()) {
			show_all(*(this->_log_file), ea.gen());
		}
	}

	// writes the content of the statistics in the stream; k is the number
	// of the individual in the statistics (e.g. in a Pareto front, the
	// statistics contains many individuals)
	void show_all(std::ostream& os, size_t gen = 0) const
	{
		for (unsigned i = 0; i < _pop.size(); ++i) {
			// gen is the generation number,
			// i the rank of the individual in the generation
			os << gen << "," << i << ",";
			// save all objective scores
			for (unsigned j = 0; j < _pop[i]->fit().objs().size(); ++j) {
				os << _pop[i]->fit().obj(j) << ",";
			}
			// call the method show() in the phenotype object, which writes
			// the phenotype data in the stream os (here the log file)
			_pop[i]->show_real_data_stats(os);
			os << std::endl;;
		}
	}

protected:
	pop_t _pop;
};

SFERES_STAT(AllIndivObjectiveScores, Stat)
{
public:

	typedef std::vector<boost::shared_ptr<Phen> > pop_t;

	// Examines the population (via the EA) to update the statistics
	template<typename E>
	void refresh(const E& ea)
	{
		assert(!ea.pop().empty());
		// takes all the population
		_pop = ea.pop();
		// Sorts by objective scores (descending order)
		parallel::sort(_pop.begin(), _pop.end(), fit::compare_objs_lex());
		this->_create_log_file(ea, "all_phenotypes_objective_scores.dat");
		if (ea.dump_enabled()) {
			show_all(*(this->_log_file), ea.gen());
		}
	}

	// writes the content of the statistics in the stream; k is the number
	// of the individual in the statistics (e.g. in a Pareto front, the
	// statistics contains many individuals)
	void show_all(std::ostream& os, size_t gen = 0) const
	{
		for (unsigned i = 0; i < _pop.size(); ++i) {
			// gen is the generation number,
			// i the rank of the individual in the generation
			os << gen << "," << i << ",";
			// save all objective scores
			for (unsigned j = 0; j < _pop[i]->fit().objs().size(); ++j) {
				os << _pop[i]->fit().obj(j) << ",";
			}
			// call the method show() in the phenotype object, which writes
			// the phenotype data in the stream os (here the log file)
			_pop[i]->show_objective_scores(os);
			os << std::endl;;
		}
	}

protected:
	pop_t _pop;
};


SFERES_STAT(AllIndivLogMovesSurvival, Stat)
{
public:

	typedef std::vector<boost::shared_ptr<Phen> > pop_t;

	// Examines the population (via the EA) to update the statistics
	template<typename E>
	void refresh(const E& ea)
	{
		assert(!ea.pop().empty());
		// takes all the population
		_pop = ea.pop();
		// Sorts by objective scores (descending order)
		parallel::sort(_pop.begin(), _pop.end(), fit::compare_objs_lex());
		this->_create_log_file(ea, "all_phenotypes_moves.dat");
		if (ea.dump_enabled()) {
			show_all(*(this->_log_file), ea.gen());
		}
	}

	// writes the content of the statistics in the stream; k is the number
	// of the individual in the statistics (e.g. in a Pareto front, the
	// statistics contains many individuals)
	void show_all(std::ostream& os, size_t gen = 0) const
	{
		for (unsigned i = 0; i < 1; ++i) {
			// gen is the generation number,
			// i the rank of the individual in the generation
			os << gen << "," << i << ",";
			// save all objective scores
			for (unsigned j = 0; j < _pop[i]->fit().objs().size(); ++j) {
				os << _pop[i]->fit().obj(j) << ",";
			}
			// call the method show_log_survival_tasks() in the phenotype object,
			// which writes the robot log in the stream os (here the log file)
			os << "NewIndividual";
			os << std::endl;
			_pop[i]->show_log_survival_tasks(os);
			os << std::endl;;
		}
	}
protected:
	pop_t _pop;
};


SFERES_STAT(AllIndivLogSurvivalTaskScores, Stat)
{
public:

	typedef std::vector<boost::shared_ptr<Phen> > pop_t;

	// Examines the population (via the EA) to update the statistics
	template<typename E>
	void refresh(const E& ea)
	{
		assert(!ea.pop().empty());
		// takes all the population
		_pop = ea.pop();
		// Sorts by objective scores (descending order)
		parallel::sort(_pop.begin(), _pop.end(), fit::compare_objs_lex());
		this->_create_log_file(ea, "all_survival_tasks_scores.dat");
		if (ea.dump_enabled()) {
			show_all(*(this->_log_file), ea.gen());
		}
	}

	// writes the content of the statistics in the stream; k is the number
	// of the individual in the statistics (e.g. in a Pareto front, the
	// statistics contains many individuals)
	void show_all(std::ostream& os, size_t gen = 0) const
	{
		for (unsigned i = 0; i < _pop.size(); ++i) {
			// gen is the generation number,
			// i the rank of the individual in the generation
			os << gen << "," << i << ",";
			// save all objective scores
			for (unsigned j = 0; j < _pop[i]->fit().objs().size(); ++j) {
				os << _pop[i]->fit().obj(j) << ",";
			}
			// call the method show_log_survival_tasks() in the phenotype object,
			// which writes the robot log in the stream os (here the log file)
			_pop[i]->show_log_survival_times(os);
			os << std::endl;;
		}
	}
protected:
	pop_t _pop;
};


}
}

#endif

/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Warlock.hpp                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: tlucanti <tlucanti@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2022/02/02 12:40:23 by tlucanti          #+#    #+#             */
/*   Updated: 2022/02/02 20:20:53 by tlucanti         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#pragma once

#include <string>
#include <iostream>
#include <list>

#include "ASpell"

using namespace::std;

class Warlock
{
public:
	Warlock(const string &_n, const string &_t) : name(_n), title(_t)
	{ cout << _n << ": This looks like another boring day.\n"; }
	~Warlock() { cout << name << ": My job here is done!\n"; }

	const string &getName() const { return name; }
	const string &getTitle() const { return title; }
	void setTitle(const string &_t) { title = _t; }
	void introduce() const { cout << name+": I am "+name+", "+title+"!\n"; }

	learnSpell(ASpell *_s)
	{

	}

private:
	string name;
	string title;
	list<ASpell> learned;

	Warlock(const Warlock &_w);
	Warlock();
	Warlock &operator =(const Warlock &_w);
};

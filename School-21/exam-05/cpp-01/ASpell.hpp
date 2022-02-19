/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ASpell.hpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: tlucanti <tlucanti@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2022/02/02 13:16:06 by tlucanti          #+#    #+#             */
/*   Updated: 2022/02/02 17:00:47 by tlucanti         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#pragma once

#include <string>
#include <iostream>

#include "ATarget.hpp"

using namespace::std;

class ATarget;

class ASpell
{
public:
	ASpell(const string &_n, const string &_e) : name(_n), effects(_e) {}
	ASpell(const ASpell &_s) : name(_s.name), effects(_s.effects) {}
	virtual ~ASpell() = 0;
	ASpell &operator =(const ASpell &_s)
	{ name = _s.name; effects = _s.effects; return *this; }

	string getName() const { return name; }
	string getEffects() const { return effects; }

	virtual ASpell *clone() const = 0;
	void launchSpell(const ATarget &t) { t.getHitBySpell(*this); }

protected:
	ASpell() {}

	string name;
	string effects;
};

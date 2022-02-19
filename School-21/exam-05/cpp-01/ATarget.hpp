/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ATarget.hpp                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: tlucanti <tlucanti@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2022/02/02 13:30:40 by tlucanti          #+#    #+#             */
/*   Updated: 2022/02/02 17:02:35 by tlucanti         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <string>
#include <iostream>

#include "ASpell.hpp"

class ASpell;

class ATarget
{
public:
	ATarget(const string &_t) : type(_t){}
	ATarget(const ATarget &_t) : type(_t.type) {}
	virtual ~ATarget() = 0;
	ATarget &operator =(const ATarget &_t)
	{ target = _t.target; return *this; }

	const string &getType() const { return type; }
	ATarget *clone() const = 0;

	void getHitBySpell(const ASpell &_s) const
	{ cout << type + "has been " + _s.getEffects() + "!\n"; }
private:
	ATarget() {}

	string type;
};

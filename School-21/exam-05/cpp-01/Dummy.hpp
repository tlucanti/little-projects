/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Dummy.hpp                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: tlucanti <tlucanti@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2022/02/02 16:55:30 by tlucanti          #+#    #+#             */
/*   Updated: 2022/02/02 19:37:08 by tlucanti         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#pragma once

#include "ATarget.hpp"

class Dummy : public ATarget
{
public:
	Dummy() : ATarget("Target Practice Dummy") {}
	Dummy(const string &_t) : ATarget(_T) {}
	Dummy(const Dummy &_d) : target(_d.target) {}
	Dummy &operator =(const Dummy &_d) { target = _d.target; }
	~Dummy() {}

	ATarget *clone() { return new Dummy(target); }
private:
};

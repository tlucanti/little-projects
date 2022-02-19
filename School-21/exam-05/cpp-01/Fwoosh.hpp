/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Fwoosh.hpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: tlucanti <tlucanti@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2022/02/02 15:52:03 by tlucanti          #+#    #+#             */
/*   Updated: 2022/02/02 19:35:04 by tlucanti         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#pragma once

#include "ASpell.hpp"

class Fwoosh : public ASpell
{
public:
	Fwoosh() : name("Fwoosh"), effects("fwooshed") {}
	Fwoosh(const string &_n, const string &_e) : ASpell(_n, _e) {}
	~Fwoosh() {}

	ASpell *clone() { return new Fwoosh(name, effects); }
private:

};

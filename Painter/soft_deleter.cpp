#include "soft_deleter.h"
#include <iostream>

Soft_deleter::Soft_deleter(Object *_obj)
		: alpha(150)
		{
			obj = _obj;
		}

int
	Soft_deleter::soft_delete(QPainter *deleter)
	{
		deleter->setPen(QPen(QColor(255, 255, 255, std::min(255, alpha + 105))));
		deleter->setBrush(QBrush(QColor(255, 255, 255, alpha)));

		obj->__internal_paint(deleter);
		alpha -= 4;
		if (alpha <= 0)
			return 1;
		return 0;

	}
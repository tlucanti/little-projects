#ifndef SOFT_DELETER_H
#define SOFT_DELETER_H

#include "circle.h"
#include "square.h"
#include "triangle.h"

class Soft_deleter
{
public:
	Soft_deleter(Object *_obj);

	int soft_delete(QPainter *deleter);

private:
	Object *obj;
	int alpha;
};

#endif // SOFT_DELETER_H

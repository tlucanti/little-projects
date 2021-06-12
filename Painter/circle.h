#ifndef CIRCLE_H
#define CIRCLE_H

#include "object.h"

class Circle : public Object
{
public:
	Circle(QWidget *parent, const Point &_center, unsigned radius, float _line_width,
		const QColor &_edge_color, const QColor &_face_color);
	Circle(const Circle &obj);

	Object *copy() const override;
	bool	inside(const QPoint &point) const override;
	void	move(const QPoint &vec) override;
	void	__internal_paint(QPainter *painter) const override;
	void	__internal_moover(QPainter *moover) const override;
	std::string str() const override;

protected:
	Point			center;
	unsigned	radius;
	unsigned	radius_sq;
};

#endif // CIRCLE_H

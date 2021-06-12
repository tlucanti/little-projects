#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "object.h"

template<size_t N>
class Triangle : public Object
{
public:
	Triangle(QWidget *_parent, const std:array<Point, N> pPoint &_p1, const Point &_p2, const Point &_p3, float _line_width, const QColor &_edge_color, const QColor &_face_color);
	Triangle(const Triangle &obj);

	Object *copy() const override;
	bool	inside(const QPoint &point) const override;
	void	move(const QPoint &vec) override;
	void	__internal_paint(QPainter *painter) const override;
	void	__internal_moover(QPainter *painter) const override;
	std::string		str() const override;

protected:
	std::array<Point, 3> p;
	Point		p1, p2, p3;
	QPoint	Qp1, Qp2, Qp3;
	QPoint	v1, v2, v3;
};

#endif // TRIANGLE_H

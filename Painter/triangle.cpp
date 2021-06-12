#include "triangle.h"

Triangle::Triangle(QWidget *_parent, const Point &_p1, const Point &_p2, const Point &_p3, float _line_width, const QColor &_edge_color, const QColor &_face_color)
	: Object(_parent, _line_width, _edge_color, _face_color), p1(_p1), p2(_p2), p3(_p3), Qp1(_p1.x, _p1.y), Qp2(_p2.x, _p2.y), Qp3(_p3.x, _p3.y), v1(Qp1 - Qp2), v2(Qp2 - Qp3), v3(Qp3 - Qp1)
	{
		Qcenter = QPoint((_p1.x + p2.x + p3.x) / 3, (_p1.y + p2.y + p3.y) / 3);
		WIDGET_HEIGHT += 110;
		WIDGET_WIDTH += 150;
		info_widget->resize(WIDGET_WIDTH, WIDGET_HEIGHT);
	}

Triangle::Triangle(const Triangle &obj)
	: Triangle(obj.parent, obj.p1, obj.p2, obj.p3, obj.line_width, obj.edge_color, obj.face_color) {}

void
	Triangle::__internal_paint(QPainter *painter) const
	{
		painter->drawPolygon(QPolygon({{(int)p1.x, (int)p1.y}, {(int)p2.x, (int)p2.y}, {(int)p3.x, (int)p3.y}}));
	}

void
	Triangle::__internal_moover(QPainter *painter) const
	{
		int dx = drag_vector->x();
		int dy = drag_vector->y();
		painter->drawPolygon(QPolygon({{(int)p1.x + dx, (int)p1.y + dy}, {(int)p2.x + dx, (int)p2.y + dy}, {(int)p3.x + dx, (int)p3.y + dy}}));
	}

bool
	Triangle::inside(const QPoint &p) const
	{
		int d1 = (p1.x - p.x()) * (p2.y - p1.y) - (p2.x - p1.x) * (p1.y - p.y());
		int d2 = (p2.x - p.x()) * (p3.y - p2.y) - (p3.x - p2.x) * (p2.y - p.y());
		int d3 = (p3.x - p.x()) * (p1.y - p3.y) - (p1.x - p3.x) * (p3.y - p.y());
		bool b1, b2;

		b1 = (d1 < 0) || (d2 < 0) || (d3 < 0);
		b2 = (d1 > 0) || (d2 > 0) || (d3 > 0);

		return  !(b1 && b2);
	}

std::string
	Triangle::str() const
	{
		return "Triangle object with properties\n"
		"point 1: " + p1.str() + "\n"
		"point 2: " + p2.str() + "\n"
		"point 3: " + p3.str();
	}

void Triangle::move(const QPoint &vec) {
	p1 += vec;
	p2 += vec;
	p3 += vec;
	Qcenter += vec;
}

Object *Triangle::copy() const {
	return new Triangle(*this);
}


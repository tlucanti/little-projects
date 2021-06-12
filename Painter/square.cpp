#include "square.h"

Square::Square(QWidget *_parent, const Point &_center, unsigned _side, float _line_width, const QColor &_edge_color, const QColor &_face_color)
	: Object(_parent, _line_width, _edge_color, _face_color), center(_center), side_half(_side / 2), side(_side)
	{
		Qcenter = QPoint(_center.x, _center.y);
		WIDGET_HEIGHT += 80;
		WIDGET_WIDTH += 210;
		info_widget->resize(WIDGET_WIDTH, WIDGET_HEIGHT);
	}

Square::Square(const Square &obj)
	: Square(obj.parent, obj.center, obj.side, obj.line_width, obj.edge_color, obj.face_color) {}

void
	Square::__internal_paint(QPainter *painter) const
	{
		painter->drawRect(center.x - (int)side_half, center.y - (int)side_half, (int)side, (int)side);
	}

void
	Square::__internal_moover(QPainter *painter) const
	{
		painter->drawRect(center.x - (int)side_half + drag_vector->x(), center.y - (int)side_half + drag_vector->y(), (int)side, (int)side);
	}

bool
	Square::inside(const QPoint &point) const
	{
		return abs(point.x() - center.x) < side_half and abs(point.y() - center.y) < side_half;
	}

std::string
	Square::str() const
	{
		return "Square object with properties\n"
			"center: " + center.str() + "\n"
			"side size: " + std::to_string(side);
	}

void Square::move(const QPoint &vec) {
	Qcenter += vec;
	center = Qcenter;
}

Object *Square::copy() const {
	return new Square(*this);
}

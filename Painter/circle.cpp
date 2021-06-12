#include "circle.h"
#include <iostream>

Circle::Circle(QWidget *_parent, const Point &_center, unsigned _radius, float _line_width,
		const QColor &_edge_color, const QColor &_face_color)
	: Object(_parent, _line_width, _edge_color, _face_color), center(_center), radius(_radius), radius_sq(_radius * _radius)
	{
		Qcenter = QPoint(_center.x, _center.y);
		WIDGET_HEIGHT += 80;
		WIDGET_WIDTH += 200;
		info_widget->resize(WIDGET_WIDTH, WIDGET_HEIGHT);
	}

Circle::Circle(const Circle &obj)
	: Circle(obj.parent, obj.center, obj.radius, obj.line_width, obj.edge_color, obj.face_color) {}

void
	Circle::__internal_paint(QPainter *painter) const
	{
		painter->drawEllipse(center.x - (int)radius, center.y - (int)radius, 2 * (int)radius, 2 * (int)radius);
	}

void
	Circle::__internal_moover(QPainter *moover) const
	{
		moover->drawEllipse(center.x - (int)radius + drag_vector->x(), center.y - (int)radius + drag_vector->y(), 2 * (int)radius, 2 * (int)radius);
	}

bool
	Circle::inside(const QPoint &point) const
	{
		return QPoint::dotProduct(point - Qcenter, point - Qcenter) < radius_sq;
	}

std::string
	Circle::str() const
	{
		return "Circle object with properties\n"
			"center: " + center.str() + "\n"
			"radius: " + std::to_string(radius);
	}

void Circle::move(const QPoint &vec)
{
	Qcenter += vec;
	center = Qcenter;
}

Object *Circle::copy() const {
	return new Circle(*this);
};


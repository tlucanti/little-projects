#ifndef SQUARE_H
#define SQUARE_H

#include "object.h"

class Square : public Object
{
public:
	Square(QWidget *_parent, const Point &_center, unsigned _side, float _line_width, const QColor &_edge_color, const QColor &_face_color);
	Square(const Square &obj);

	Object *copy() const override;
	bool	inside(const QPoint &point) const override;
	void	move(const QPoint &vec) override;
	void	__internal_paint(QPainter *painter) const override;
	void	__internal_moover(QPainter *painter) const override;
	std::string str() const override;

protected:
	Point center;
	unsigned side_half;
	unsigned side;
};

#endif // SQUARE_H

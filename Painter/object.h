#ifndef OBJECT_H
#define OBJECT_H

#include <QColor>
#include <QPainter>
#include <QWidget>
#include <QGridLayout>
#include <QTextEdit>
#include <string>
#include "qdynamicbutton.h"

#define WINDOW_HEIGHT 800
#define WINDOW_WIDTH 1150

class Object
{
public:
	Object(QWidget *_parent, float _line_width, const QColor &_edge_color, const QColor &_face_color);
	Object(const Object &cpy);
	~Object();
	
	void	show_info(const QPoint &pos);
	void	hide_info();
	void	update_info(const QPoint &pos);
	void	paint(QPainter *painter);
	void	light(QPainter *painter);
	void	unlight(QPainter *lighter);
	void	move_preview(QPainter *moover);

	virtual bool	inside(const QPoint &point) const = 0;
	virtual	void	move(const QPoint &vec) = 0;
	virtual std::string	str() const = 0;
	virtual Object *copy() const = 0;

	QPoint			*drag_point;
	QPoint			*drag_vector;
	char				inside_flag;

//protected:
	void	__init_painter(QPainter *painter);
	void	__init_lighter(QPainter *painter);
	void	__unlight_init(QPainter *lighter);
	void	__init_moover(QPainter *moover);

	virtual void	__internal_paint(QPainter *painter) const = 0;
	virtual void	__internal_moover(QPainter *painter) const = 0;

protected:
	QPoint					Qcenter;
	int							selection_alpha;
	float						line_width;
	QColor					edge_color;
	QColor					face_color;
	QWidget					*info_widget;
	QTextEdit				*text_widget;
	QGridLayout			*layout;
	QDynamicButton	*delete_button;
	QWidget					*parent;

	int WIDGET_HEIGHT;
	int WIDGET_WIDTH;
	class Point
	{
	public:
			Point(int __x, int __y);
			Point operator +(const Point &p2);
			Point operator -(const Point &p2);
			Point &operator =(const QPoint &p);
			Point operator +(const QPoint &p2);
			Point operator -(const QPoint &p2);
			Point &operator +=(const Point &p2);
			Point &operator +=(const QPoint &p2);

			int x;
			int y;

			std::string str() const;
	};
};


#endif // OBJECT_H

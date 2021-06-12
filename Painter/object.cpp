#include "object.h"
#include <iostream>

Object::Object(QWidget *_parent, float _line_width, const QColor &_edge_color, const QColor &_face_color)
	: line_width(_line_width), edge_color(_edge_color), face_color(_face_color), inside_flag(false), drag_point(nullptr),
		drag_vector(nullptr), selection_alpha(0), parent(_parent), WIDGET_WIDTH(0), WIDGET_HEIGHT(30)
	{
		info_widget = new QWidget(_parent);
		layout = new QGridLayout(info_widget);
		text_widget = new QTextEdit( info_widget);
		delete_button = new QDynamicButton(QString("delete"), info_widget, this);
		QPalette palette;
		palette.setColor(QPalette::Base, QColor(53, 53, 53, 150));
		text_widget->setPalette(palette);
		text_widget->setReadOnly(true);
		layout->addWidget(text_widget);
		layout->addWidget(delete_button);
		QObject::connect(delete_button, SIGNAL(clicked()), _parent, SLOT(delete_button_clicked()));
//		delete_button->connect()
//		delete_button->setFixedSize(70, 25);
		info_widget->hide();
	}

Object::Object(const Object &cpy)
	: Object(cpy.parent, cpy.line_width, cpy.edge_color, cpy.face_color) {
	std::cout << "copy constructor called\n";
}

Object::~Object()
{
	delete delete_button;
	delete text_widget;
	delete layout;
	delete info_widget;
}

Object::Point::Point(int __x, int __y)
	: x(__x), y(__y) {}

std::string
	Object::Point::str() const
	{
		return "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
	}

Object::Point
	Object::Point::operator+(const Object::Point &p2)
	{
		return {x + p2.x, y + p2.y};
	}

Object::Point
	Object::Point::operator-(const QPoint &p2)
	{
		return {x - p2.x(), y - p2.y()};
	}

Object::Point
	Object::Point::operator+(const QPoint &p2)
	{
		return {x + p2.x(), y + p2.y()};
	}

Object::Point
	Object::Point::operator-(const Object::Point &p2)
	{
		return {x - p2.x, y - p2.y};
	}

Object::Point &
	Object::Point::operator=(const QPoint &p)
	{
		x = p.x();
		y = p.y();
		return *this;
	}

Object::Point &
	Object::Point::operator+=(const Object::Point &p2)
	{
		x += p2.x;
		y += p2.y;
		return *this;
	}

Object::Point &
	Object::Point::operator+=(const QPoint &p2)
	{
		x += p2.x();
		y += p2.y();
		return *this;
	}

void
	Object::__init_painter(QPainter *painter)
	{
		painter->setPen(QPen(edge_color, line_width));
		painter->setBrush(QBrush(face_color));
	}

void
Object::__init_lighter(QPainter *lighter)
{
	selection_alpha = std::min(200, selection_alpha + 4);
	lighter->setPen(QPen(QColor(0, 0, 0, 0), 0));
	lighter->setBrush(QBrush(QColor(255, 255, 255, selection_alpha)));
}

void
	Object::__unlight_init(QPainter *lighter)
	{
		selection_alpha = std::max(0, selection_alpha - 4);
		lighter->setPen(QPen(QColor(0, 0, 0, 0), 0));
		lighter->setBrush(QBrush(QColor(255, 255, 255, selection_alpha)));
		if (selection_alpha == 0)
			inside_flag = 0;
	}

void
	Object::__init_moover(QPainter *moover)
	{
		moover->setPen(QPen(QColor(255, 255, 255)));
		moover->setBrush(QBrush(QColor(255, 255, 255, std::max(30, selection_alpha))));
	}

void
	Object::paint(QPainter *painter)
	{
		__init_painter(painter);
		__internal_paint(painter);
	}

void
	Object::light(QPainter *lighter)
	{
		__init_lighter(lighter);
		__internal_paint(lighter);
	}

void
	Object::unlight(QPainter *lighter)
	{
		__unlight_init(lighter);
		__internal_paint(lighter);
	}

void
	Object::move_preview(QPainter *moover)
	{
		__init_moover(moover);
		__internal_moover(moover);
		moover->setPen(QColor(255, 255, 255, 150));
		moover->drawLine(Qcenter, Qcenter + *drag_vector);
	}

void
	Object::show_info(const QPoint &pos)
{
	update_info(pos);
	info_widget->show();
}

void
	Object::hide_info()
{
	info_widget->hide();
}

void
	Object::update_info(const QPoint &pos)
	{
		int x = pos.x();
		int y = pos.y();
		text_widget->setText(QString::fromStdString(str()));
		if (x > WINDOW_WIDTH - WIDGET_WIDTH)
			x -= WIDGET_WIDTH;
		if (y > WINDOW_HEIGHT - WIDGET_HEIGHT)
			y -= WIDGET_HEIGHT;
		info_widget->move(x, y);
	}

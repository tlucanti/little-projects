#include "mywidget.h"
#include <iostream>

MyWidget::MyWidget(QWidget *qw)
	: QWidget(qw)
	{
	setBackgroundRole(QPalette::Base);
	setAutoFillBackground(true);
	setMouseTracking(true);
	//																										center			radius	edge_width, edge_color	face_color
	obj_map.push_back(std::unique_ptr<Object> (new Circle(this, {100, 100}, 50,			1, 					{255, 255, 255},	{255, 0, 0})));
	obj_map.push_back(std::unique_ptr<Object> (new Circle(this, {500, 500}, 100, 2, {0,255,255}, {0, 0, 255})));
	obj_map.push_back(std::unique_ptr<Object> (new Circle(this, {800, 200}, 80, 3, {255,255,0}, {0, 200, 0})));
	//																										center			side	edge_width	edge_color	face_color
	obj_map.push_back(std::unique_ptr<Object> (new Square(this, {400, 200}, 100,		3,					{112,154,113},	{212, 73, 113})));
	obj_map.push_back(std::unique_ptr<Object> (new Square(this, {100, 400}, 150, 3, {185,124,83}, {212, 143, 93})));

	obj_map.push_back(std::unique_ptr<Object> (new Triangle(this, {150, 150}, {250, 200}, {200, 250}, 3, {134,224,54}, {70,70,70})));
	obj_map.push_back(std::unique_ptr<Object> (new Triangle(this, {/*700*/ 1150, 700}, {550, 800}, {600, 630}, 4, {23,51,73}, {224,134,54})));
}

void pr(const QPoint &p) {
	std::cout << '(' << p.x() << ", " << p.y() << ")\n";
}

void
	MyWidget::paintEvent(QPaintEvent *event)
	{
		QPainter painter(this);

		Q_UNUSED(event);
		for (auto it=del_map.begin(); it != del_map.end();) {
			auto next = ++it;
			--it;
			if ((*it)->soft_delete(&painter))
				del_map.erase(it);
			it = next;
		}
		for (const auto &obj : obj_map) {
			obj->paint(&painter);
			if (obj->inside_flag == 1)
				obj->light(&painter);
			else if (obj->inside_flag == -1)
				obj->unlight(&painter);
		}
		for (const auto &obj : obj_map)
		{
			if (obj->drag_vector != nullptr)
				obj->move_preview(&painter);
		}
		update();
	}

void
	MyWidget::mouseMoveEvent(QMouseEvent *event)
	{
		QPoint point(event->pos());

		for (const auto &obj : obj_map)
		{
			if (obj->drag_point != nullptr) {
				if (obj->drag_vector == nullptr)
					obj->drag_vector = new QPoint(0, 0);
				else
					*obj->drag_vector = event->pos() - *obj->drag_point;
			}
			if (obj->inside(point))
			{
				obj->inside_flag = true;
				update();
			}
			else if (obj->inside_flag)
			{
				obj->inside_flag = -1;
				obj->hide_info();
				update();
			}
		}
	}

void
	MyWidget::mousePressEvent(QMouseEvent *event)
	{
		if (event->button() == Qt::RightButton)
		{
			for (const auto &obj : obj_map)
			{
				if (obj->inside(event->pos())) {
					obj->show_info(event->pos());
				}
			}
		}
		else if (event->button() == Qt::LeftButton)
		{
			for (const auto &obj : obj_map)
			{
				if (obj->inside(event->pos())) {
					if (obj->drag_point == nullptr)
						obj->drag_point = new QPoint(event->pos());
				}
			}
		}
	}

void
	MyWidget::mouseReleaseEvent(QMouseEvent *event)
	{
		for (const auto &obj : obj_map)
		{
			if (obj->drag_point != nullptr)
			{
				del_map.push_back(std::unique_ptr<Soft_deleter>(new Soft_deleter(obj->copy())));
				obj->move(event->pos() - *obj->drag_point);
				delete obj->drag_point;
				obj->drag_point = nullptr;
				delete obj->drag_vector;
				obj->drag_vector = nullptr;
			}
		}
		mouseMoveEvent(event);
		update();
	}

void MyWidget::delete_button_clicked() {
	auto *obj = (Object *)((QDynamicButton *)sender())->connected_object;
	del_map.push_back(std::unique_ptr<Soft_deleter>(new Soft_deleter(obj->copy())));
	for (auto it = obj_map.begin(); it != obj_map.end(); ++it) {
		if ((*it).get() == obj) {
			obj_map.erase(it);
			break;
		}
	}
//	std::cout << "clicked button: " << obj->str() << std::endl;
}


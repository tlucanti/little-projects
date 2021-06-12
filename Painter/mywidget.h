#ifndef MYWIDGET_H
#define MYWIDGET_H

#include <QWidget>
#include <QMainWindow>
#include <QMouseEvent>
#include <QPainter>
#include <QMessageBox>
#include <memory>
#include "object.h"
#include "circle.h"
#include "square.h"
#include "triangle.h"
#include "soft_deleter.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MyQWidget; }
QT_END_NAMESPACE

class MyWidget : public QWidget
{
	Q_OBJECT

public:
	explicit MyWidget(QWidget *qw=nullptr);

protected:
	void	paintEvent(QPaintEvent *event) override;
	void	mouseMoveEvent(QMouseEvent *event) override;
	void	mousePressEvent(QMouseEvent *event) override;
	void	mouseReleaseEvent(QMouseEvent *event) override;

private slots:
		void	delete_button_clicked();
private:
	std::list<std::unique_ptr<Object>>				obj_map;
	std::list<std::unique_ptr<Soft_deleter>>	del_map;

};


#endif // MYWIDGET_H

#include "qdynamicbutton.h"

QDynamicButton::QDynamicButton(const QString &text, QWidget *parent, void *_connected_object)
	: QPushButton(text, parent)
{
	connected_object = _connected_object;
}

#ifndef QDYNAMICBUTTON_H
#define QDYNAMICBUTTON_H

#include <QPushButton>
//#include "object.h"

class QDynamicButton : public QPushButton
{
		Q_OBJECT
public:
    QDynamicButton(const QString &text, QWidget *parent, void * _connected_object);

		void *connected_object;
};

#endif // QDYNAMICBUTTON_H

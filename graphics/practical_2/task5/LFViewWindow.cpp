#include <QGridLayout>
#include <QTimer>
#include <QShortcut>

#define _USE_MATH_DEFINES
#include <math.h>

#include "LFViewWindow.h"


LFViewerWindow::LFViewerWindow()
{
    setWindowTitle(tr("Light Field Renderer"));

	cameraPosLabel = new QLabel();
    cameraPosLabel->setAlignment(Qt::AlignHCenter);

	LFViewWidget * lfViewWidget = new LFViewWidget(&lfRenderer, this);

	QHBoxLayout *main_layout = new QHBoxLayout;
    main_layout->addWidget(lfViewWidget,2);
	QVBoxLayout *controls_layout = new QVBoxLayout;
	main_layout->addLayout(controls_layout);
	controls_layout->addWidget(cameraPosLabel);

	QHBoxLayout *focal_layout = new QHBoxLayout;
	QLabel *focalLabel = new QLabel(tr("Focal depth"));
	focalLabel->setAlignment(Qt::AlignRight);
	focal_layout->addWidget(focalLabel);
	focalSB = new QDoubleSpinBox(this);
	focalSB->setMinimum(0);
	focalSB->setMaximum(15000);
	focal_layout->addWidget(focalSB);
	controls_layout->addLayout(focal_layout);

	QHBoxLayout *aperture_layout = new QHBoxLayout;
	QLabel *apertureLabel = new QLabel(tr("Aperture"));
	apertureLabel->setAlignment(Qt::AlignRight);
	aperture_layout->addWidget(apertureLabel);
	apertureSB = new QDoubleSpinBox(this);
	apertureSB->setMinimum(1);
	apertureSB->setMaximum(100);
	aperture_layout->addWidget(apertureSB);
	controls_layout->addLayout(aperture_layout);

	QHBoxLayout *fovlayout = new QHBoxLayout;
	QLabel *fovLabel = new QLabel(tr("Field-of-view"));
	fovLabel->setAlignment(Qt::AlignRight);
	fovlayout->addWidget(fovLabel);
	fovSB = new QDoubleSpinBox(this);
	fovSB->setMinimum(2);
	fovSB->setMaximum(180);
	fovlayout->addWidget(fovSB);
	controls_layout->addLayout(fovlayout);
	focalLengthLabel = new QLabel();
	focalLengthLabel->setAlignment(Qt::AlignRight);
	controls_layout->addWidget(focalLengthLabel);

	controls_layout->addStretch();

	QLabel *controlsHelpLabel = new QLabel(tr("To move the camera:\n Hold LMB and move (X/Z);\n Hold Shift+LMB for (X/Y); or\n Keys A/D, S/W and Q.R\n\n"
			"To change focal depth:\n Hold RMB and move; or\n Keys Up/Down\n\n"
			"To change aperture:\n Scroll wheel; or\n Keys Left/Right\n\n"
			"To change fov:\n Keys -/="
			));
	controls_layout->addWidget( controlsHelpLabel );
	renderTimeLabel = new QLabel(this);
	controls_layout->addWidget( renderTimeLabel );

	QWidget *central_widget = new QWidget( this );
	central_widget->setLayout(main_layout);
	setCentralWidget(central_widget);

	QShortcut *quit_shortcut = new QShortcut(QKeySequence(tr("Ctrl+Q", "Quit")), this );
	connect(quit_shortcut, &QShortcut::activated, this, &LFViewerWindow::close);


    lfViewWidget->setFocus();

	connect(lfViewWidget, &LFViewWidget::KposChanged, this, &LFViewerWindow::KposChanged);
	connect(lfViewWidget, &LFViewWidget::focalChanged, focalSB, &QDoubleSpinBox::setValue);
	connect(lfViewWidget, &LFViewWidget::apertureChanged, apertureSB, &QDoubleSpinBox::setValue);
	connect(lfViewWidget, &LFViewWidget::fovChanged, fovSB, &QDoubleSpinBox::setValue);
	connect(lfViewWidget, &LFViewWidget::fovChanged, this, &LFViewerWindow::fovUpdated);

	connect(lfViewWidget, &LFViewWidget::renderTimeUpdated, this, &LFViewerWindow::renderTimeUpdated);

	connect(focalSB, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), lfViewWidget, &LFViewWidget::setFocal);
	connect(apertureSB, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), lfViewWidget, &LFViewWidget::setAperture);
	connect(fovSB, static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), lfViewWidget, &LFViewWidget::setFov);

    //QTimer *timer = new QTimer(this);
    //connect(timer, &QTimer::timeout, lfViewWidget, &LFViewWidget::animate);
    //timer->start(50);

	// Update labels
	lfViewWidget->setKpos(lfViewWidget->getKpos());
	lfViewWidget->setFocal(lfViewWidget->getFocal());
	lfViewWidget->setAperture(lfViewWidget->getAperture());
	lfViewWidget->setFov(lfViewWidget->getFov());
}

void LFViewerWindow::KposChanged(QVector3D newKpos)
{
	QString label = QString("Camera Position\n(%1, %2, %3)").arg(QString::number(newKpos.x()), QString::number(newKpos.y()), QString::number(newKpos.z()));
	cameraPosLabel->setText(label);
}

void LFViewerWindow::renderTimeUpdated(int renderTime)
{
	QString label = QString("Rendering time %1 ms").arg(QString::number(renderTime));
	renderTimeLabel->setText(label);
}

void LFViewerWindow::fovUpdated(double fov)
{
	const double sensorWidth = 36.;
	double focalLength = sensorWidth / 2. / tan(fov / 2. * M_PI / 180.);
	QString label = QString("Focal length: %1 mm").arg(QString::number(focalLength));
	focalLengthLabel->setText(label);
}

void LFViewerWindow::loadLFData(QString lf_dir)
{
	lfRenderer.readLfData(lf_dir.toLocal8Bit());
}
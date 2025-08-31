# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Qt GUI window using PySide6

import sys
from os import getcwd, cpu_count, chdir
import logging

from PySide6.QtCore import QObject, QThread, QSize, Qt, QRect, QTimer, Signal, Slot, QMetaObject
from PySide6.QtWidgets import (
    QApplication, QPushButton, QTextEdit, QVBoxLayout, QWidget, 
    QGridLayout, QLabel, QPlainTextEdit, QSlider, QComboBox, QFileDialog
)

from pypef import __version__
from pypef.main import __doc__, run_main, logger, formatter
from pypef.utils.helpers import get_device, get_vram, get_torch_version, get_nvidia_gpu_info_pynvml


button_style = """
QPushButton {
	border: 2px solid rgb(52, 59, 72);
	border-radius: 5px;	
	background-color: rgb(52, 59, 72);
	color: white; 
}
QPushButton:hover {
	background-color: rgb(57, 65, 80);
	border: 2px solid rgb(61, 70, 86);
}
QPushButton:pressed {	
	background-color: rgb(35, 40, 49);
	border: 2px solid rgb(43, 50, 61);
}
QPushButton:disabled {
    background-color: grey;
}
"""

text_style = """
QLabel {
	color: white;
}"""


class QTextEditLogger(logging.Handler, QObject):
    """
    Thread-safe logging handler for PyQt/PySide applications.
    """
    log_signal = Signal(str)

    def __init__(self, parent):
        super().__init__()
        QObject.__init__(self)
        self.widget = QPlainTextEdit(parent)
        self.widget.setReadOnly(True)
        self.widget.setStyleSheet(
            "font-family:Consolas;font-size:12px;font-weight:normal;color:white;"
            "background-color:rgb(54, 69, 79);border:2px solid rgb(52, 59, 72);"
        )
        self.log_signal.connect(self.append_log)

    @Slot(str)
    def append_log(self, msg):
        self.widget.appendPlainText(msg)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)


def trap_exc_during_debug(*args):
    # When app raises uncaught exception, print info
    print(args)


# Install exception hook: without this, uncaught 
# exception would cause application to exit
sys.excepthook = trap_exc_during_debug


class Worker(QObject):
    """
    Must derive from QObject in order to emit signals, connect 
    slots to other signals, and operate in a QThread. 
    Code/logic taken from 
    https://stackoverflow.com/a/41605909/28792835.
    """
    sig_step = Signal(int, str)
    sig_done = Signal(int)
    sig_msg = Signal(str)
    sig_abort = Signal(int)

    def __init__(self, id_: int, cmd):
        super().__init__()
        self.__id = id_
        self.cmd =  cmd

    @Slot()  
    def work(self):
        """
        This worker method does work that takes a long time: 
        During this time, the thread's event loop is blocked, 
        except if the application's processEvents() is called: 
        this gives every thread (incl. main) a chance to process 
        events, which in this sample means processing signals
        received from GUI (such as abort).
        If a long job is run, e.g. retraining a deep learning 
        model, the thread's event loop is blocked for a long 
        time, and the applications's processEvents() will for 
        that time receive no process updates, which means the
        threads job cannot be quit (but just forcefully terminated 
        using the QThread.terminate() function, which is not 
        advised/secure). 
        The only remaining option seems to be getting callbacks 
        from such long working thread job during run, e.g., 
        every trained epoch from the executed imported function. 
        """
        print(f"Executing command: {self.cmd}")
        run_main(argv=self.cmd)
        self.sig_done.emit(f"Done: {self.__id}")

    def abort(self):
        self.sig_msg.emit(f'Worker #{self.__id} notified to abort')


class InfoWorker(QObject):
    sig_tick = Signal(str)
    sig_abort = Signal()

    def __init__(self, id_: int):
        super().__init__()
        self.__id = id_
        self.abort = False
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.on_timeout)

        self.sig_abort.connect(self.stop)

    def start(self):
        self.timer.start()

    @Slot()
    def stop(self):
        self.abort=True
        self.timer.stop()

    @Slot()
    def on_timeout(self):
        if not self.abort:
            self.sig_tick.emit(get_vram(verbose=False)[1])
        else:
            self.timer.stop()


class SecondWindow(QWidget):
   def __init__(self):
      super().__init__()
      layout = QVBoxLayout()
      self.setLayout(layout)


class MainWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.sig_start = Signal()  # needed only due to PyCharm debugger bug
        self.llm = 'esm'
        self.regression_model = 'PLS'
        self.mklsts_cv_method = ''
        self.c = 0
        self.n_cores = 1
        self.ls_proportion = 0.8
        self.setMinimumSize(QSize(1400, 800))
        self.setWindowTitle("PyPEF GUI")
        self.setStyleSheet("background-color: rgb(40, 44, 52);")
        self.win2 = SecondWindow()

        QThread.currentThread().setObjectName('main')
        self.__workers_done = None
        self.__threads = None

        # Texts #########################################################################
        layout = QGridLayout(self)  # MAIN LAYOUT: QGridLayout
        self.version_text = QLabel(f"PyPEF v. {__version__}", alignment=Qt.AlignRight)
        self.llm_text = QLabel("LLM")
        self.regression_model_text =  QLabel("Regression model")
        self.utils_text = QLabel("Utilities")
        self.mklsts_cv_options_text = QLabel("Cross-validation split options")
        self.dca_text = QLabel("DCA & LLM (unsupervised)")
        self.hybrid_text = QLabel("Hybrid (supervised DCA)")
        self.hybrid_dca_llm_text = QLabel("Hybrid (supervised DCA+LLM)")
        self.supervised_text = QLabel("Purely supervised")
        self.slider_text = QLabel("Train set proportion: 0.8")

        for txt in [
            self.version_text, self.regression_model_text, 
            self.utils_text, self.llm_text, self.dca_text, self.hybrid_text, 
            self.supervised_text, self.hybrid_text, self.hybrid_dca_llm_text,
            self.slider_text
        ]:
            txt.setStyleSheet(text_style)

        self.device_text_out = QTextEdit(readOnly=True)
        self.device_text_out.setStyleSheet(
            "font-family:Consolas;font-size:12px;font-weight:normal;color:white;"
            "background-color:rgb(54, 69, 79);border:2px solid rgb(52, 59, 72);"
        )
        self.device_text_out.setFixedHeight(85)
        self.device_text_out_info_text = (
            f"Device (for LLM/DCA): {get_device().upper()}\n"
            f"{get_nvidia_gpu_info_pynvml()[0]}\n"
            f"PyTorch version: {get_torch_version()}\n"
            f"Driver version: {get_nvidia_gpu_info_pynvml()[1]}\n"
            f"{get_vram(verbose=False)[0]}"
        )
        self.device_text_out.setPlainText(self.device_text_out_info_text)


        self.textedit_out = QTextEdit(readOnly=True)
        self.textedit_out.setStyleSheet(
            "font-family:Consolas;font-size:12px;font-weight:normal;color:white;"
            "background-color:rgb(54, 69, 79);border:2px solid rgb(52, 59, 72);"
        )
        self.logTextBox = QTextEditLogger(self)
        self.logTextBox.setFormatter(formatter)
        logger.addHandler(self.logTextBox)

        self.logTextBox.widget.appendPlainText(
            f"Current working directory: {str(getcwd())}")

        # Horizontal slider #############################################################
        self.slider = QSlider(self)
        self.slider.setGeometry(QRect(190, 100, 200, 16))
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(80)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.move(10, 130)
        self.slider.valueChanged.connect(self.selection_ls_proportion)

        # ComboBoxes ####################################################################
        self.box_regression_model = QComboBox()
        self.regression_models = [
            'PLS', 'PLS_LOOCV', 'Ridge', 'Lasso', 'ElasticNet', 'SVR', 'RF', 'MLP'
        ]
        self.box_regression_model.addItems(self.regression_models)
        self.box_regression_model.currentIndexChanged.connect(
            self.selection_regression_model
        )
        self.box_regression_model.setStyleSheet(
            "color:white;background-color:rgb(54, 69, 79);"
        )

        self.box_llm = QComboBox()
        self.box_llm.addItems(['None', 'ESM1v', 'ProSST'])
        self.box_llm.currentIndexChanged.connect(self.selection_llm_model)
        self.box_llm.setCurrentIndex(1)
        self.box_llm.setStyleSheet("color:white;background-color:rgb(54, 69, 79);")

        self.box_mklsts_cv = QComboBox()
        self.box_mklsts_cv.addItems([
            'None', 'Random split', 'Modulo split', 
            'Continuous split', 'Plot distributions'
        ])
        self.box_mklsts_cv.currentIndexChanged.connect(self.selection_mklsts_splits)
        self.box_mklsts_cv.setStyleSheet("color:white;background-color:rgb(54, 69, 79);")
        
        # Buttons #######################################################################
        # Utilities
        self.button_work_dir = QPushButton("Set Working Directory")
        self.button_work_dir.setToolTip(
            "Set working directory for storing output files"
        )
        self.button_work_dir.clicked.connect(self.set_work_dir)
        self.button_work_dir.setStyleSheet(button_style)        

        self.button_help = QPushButton("Help")  
        self.button_help.setToolTip("Print help text")
        self.button_help.clicked.connect(self.pypef_help)
        self.button_help.setStyleSheet(button_style)

        self.button_mklsts = QPushButton("Create LS and TS (MKLSTS)")       
        self.button_mklsts.setToolTip(
            "Create \"FASL\" files for training and testing "
            "from variant-fitness CSV data"
        )
        self.button_mklsts.clicked.connect(self.pypef_mklsts)
        self.button_mklsts.setStyleSheet(button_style)

        self.button_mkps = QPushButton("Create PS (MKPS)")       
        self.button_mkps.setToolTip(
            "Create FASTA files for prediction from variant-fitness CSV data"
        )
        self.button_mkps.clicked.connect(self.pypef_mkps)
        self.button_mkps.setStyleSheet(button_style)
        # SSM (Utilities)
        self.button_gremlin_ssm = QPushButton(
            "GREMLIN SSM prediction"
        )
        self.button_gremlin_ssm.setMinimumWidth(80)
        self.button_gremlin_ssm.setToolTip(
            "Generating DCA parameters using GREMLIN (\"MSA optimization\") and save "
            "plots of visualized results; requires an MSA in FASTA or A2M format"
        )
        self.button_gremlin_ssm.clicked.connect(
            self.pypef_gremlin_ssm
        )
        self.button_gremlin_ssm.setStyleSheet(button_style)

        self.button_llm_ssm = QPushButton("LLM SSM prediction")
        self.button_llm_ssm.setMinimumWidth(80)
        self.button_llm_ssm.setToolTip(
            "Runs full site-saturation (single) mutagenesis using the selected LLM predcitor "
            "and saves resulting landscape mutation effect plot"
        )
        self.button_llm_ssm.clicked.connect(
            self.pypef_llm_ssm
        )
        self.button_llm_ssm.setStyleSheet(button_style)
        
        # DCA
        self.button_dca_inference_gremlin = QPushButton(
            "MSA optimization (GREMLIN)"
        )
        self.button_dca_inference_gremlin.setMinimumWidth(80)
        self.button_dca_inference_gremlin.setToolTip(
            "Generating DCA parameters using GREMLIN (\"MSA optimization\"); "
            "requires an MSA in FASTA or A2M format"
        )
        self.button_dca_inference_gremlin.clicked.connect(self.pypef_gremlin)
        self.button_dca_inference_gremlin.setStyleSheet(button_style)

        self.button_dca_test_dca = QPushButton("Test (DCA)")
        self.button_dca_test_dca.setMinimumWidth(80)
        self.button_dca_test_dca.setToolTip(
            "Test performance on any test dataset using "
            "the MSA-optimized GREMLIN model"
        )
        self.button_dca_test_dca.clicked.connect(self.pypef_dca_test)
        self.button_dca_test_dca.setStyleSheet(button_style)

        self.button_dca_predict_dca = QPushButton("Predict (DCA)")
        self.button_dca_predict_dca.setMinimumWidth(80)
        self.button_dca_predict_dca.setToolTip(
            "Predict any dataset using the MSA-optimized GREMLIN model"
        )
        self.button_dca_predict_dca.clicked.connect(self.pypef_dca_predict)
        self.button_dca_predict_dca.setStyleSheet(button_style)

        # Zero-shot LLM
        self.button_llm_test_zs = QPushButton("Test (LLM)")
        self.button_llm_test_zs.setMinimumWidth(80)
        self.button_llm_test_zs.setToolTip(
            "Test performance on any test dataset using "
            "the LLM model for zero-shot prediction"
        )
        self.button_llm_test_zs.clicked.connect(self.pypef_llm_test)
        self.button_llm_test_zs.setStyleSheet(button_style)

        self.button_llm_predict_zs = QPushButton("Predict (LLM)")
        self.button_llm_predict_zs.setMinimumWidth(80)
        self.button_llm_predict_zs.setToolTip(
            "Test performance on any test dataset using "
            "the LLM model for zero-shot prediction"
        )
        self.button_llm_predict_zs.clicked.connect(self.pypef_llm_predict)
        self.button_llm_predict_zs.setStyleSheet(button_style)

        # Hybrid DCA
        self.button_hybrid_train_dca = QPushButton("Train (DCA)")
        self.button_hybrid_train_dca.setMinimumWidth(80)
        self.button_hybrid_train_dca.setToolTip(
            "Optimize the GREMLIN model by supervised "
            "training on variant-fitness labels"
        )
        self.button_hybrid_train_dca.clicked.connect(self.pypef_dca_hybrid_train)
        self.button_hybrid_train_dca.setStyleSheet(button_style)

        self.button_hybrid_train_test_dca = QPushButton("Train-Test (DCA)")
        self.button_hybrid_train_test_dca.setMinimumWidth(80)
        self.button_hybrid_train_test_dca.setToolTip(
            "Optimize the GREMLIN model by supervised training "
            "on variant-fitness labels and testing the model "
            "on a test set"
        )
        self.button_hybrid_train_test_dca.clicked.connect(
            self.pypef_dca_hybrid_train_test
        )
        self.button_hybrid_train_test_dca.setStyleSheet(button_style)

        self.button_hybrid_test_dca = QPushButton("Test (DCA)")
        self.button_hybrid_test_dca.setMinimumWidth(80)
        self.button_hybrid_test_dca.setToolTip(
            "Test the trained hybrid DCA model on a test set"
        )
        self.button_hybrid_test_dca.clicked.connect(self.pypef_dca_hybrid_test)
        self.button_hybrid_test_dca.setStyleSheet(button_style)

        # Hybrid DCA prediction
        self.button_hybrid_predict_dca = QPushButton("Predict (DCA)")
        self.button_hybrid_predict_dca.setMinimumWidth(80)
        self.button_hybrid_predict_dca.setToolTip(
            "Predict FASTA dataset using the hybrid DCA model"
        )
        self.button_hybrid_predict_dca.clicked.connect(
            self.pypef_dca_hybrid_predict
        )
        self.button_hybrid_predict_dca.setStyleSheet(button_style)

        # Hybrid DCA+LLM
        self.button_hybrid_train_dca_llm = QPushButton("Train (DCA+LLM)")
        self.button_hybrid_train_dca_llm.setMinimumWidth(80)
        self.button_hybrid_train_dca_llm.setToolTip(
            "Optimize the GREMLIN model and tune the LLM by "
            "supervised training on variant-fitness labels"
        )
        self.button_hybrid_train_dca_llm.clicked.connect(
            self.pypef_dca_llm_hybrid_train
        )
        self.button_hybrid_train_dca_llm.setStyleSheet(button_style)

        self.button_hybrid_train_test_dca_llm = QPushButton("Train-Test (DCA+LLM)")
        self.button_hybrid_train_test_dca_llm.setMinimumWidth(80)
        self.button_hybrid_train_test_dca_llm.setToolTip(
            "Optimize the GREMLIN model and tune the LLM by supervised "
            "training on variant-fitness labels and testing the model "
            "on a test set"
        )
        self.button_hybrid_train_test_dca_llm.clicked.connect(
            self.pypef_dca_llm_hybrid_train_test
        )
        self.button_hybrid_train_test_dca_llm.setStyleSheet(button_style)

        self.button_hybrid_test_dca_llm = QPushButton("Test (DCA+LLM)")
        self.button_hybrid_test_dca_llm.setMinimumWidth(80)
        self.button_hybrid_test_dca_llm.setToolTip(
            "Test the trained hybrid DCA+LLM model on a test set"
        )
        self.button_hybrid_test_dca_llm.clicked.connect(
            self.pypef_dca_llm_hybrid_test
        )
        self.button_hybrid_test_dca_llm.setStyleSheet(button_style)

        self.button_hybrid_predict_dca_llm = QPushButton("Predict (DCA+LLM)")
        self.button_hybrid_predict_dca_llm.setMinimumWidth(80)
        self.button_hybrid_predict_dca_llm.setToolTip(
            "Use the trained hybrid DCA+LLM model for prediction"
        )
        self.button_hybrid_predict_dca_llm.clicked.connect(
            self.pypef_dca_llm_hybrid_predict
        )
        self.button_hybrid_predict_dca_llm.setStyleSheet(button_style)

        # Pure Supervised
        self.button_supervised_train_dca = QPushButton("Train (DCA encoding)")
        self.button_supervised_train_dca.setMinimumWidth(80)
        self.button_supervised_train_dca.setToolTip(
            "Purely supervised DCA (GREMLIN or PLMC) "
            "model training on variant-fitness labels"
        )
        self.button_supervised_train_dca.clicked.connect(
            self.pypef_dca_supervised_train
        )
        self.button_supervised_train_dca.setStyleSheet(button_style)

        self.button_supervised_train_test_dca = QPushButton(
            "Train-Test (DCA encoding)"
        )
        self.button_supervised_train_test_dca.setMinimumWidth(80)
        self.button_supervised_train_test_dca.setToolTip(
            "Purely supervised DCA (GREMLIN or PLMC) model "
            "training and testing on variant-fitness labels"
        )
        self.button_supervised_train_test_dca.clicked.connect(
            self.pypef_dca_supervised_train_test
        )
        self.button_supervised_train_test_dca.setStyleSheet(button_style)

        self.button_supervised_test_dca = QPushButton("Test (DCA encoding)")
        self.button_supervised_test_dca.setMinimumWidth(80)
        self.button_supervised_test_dca.setToolTip(
            "Purely supervised DCA (GREMLIN or PLMC) model test"
        )
        self.button_supervised_test_dca.clicked.connect(
            self.pypef_dca_supervised_test
        )
        self.button_supervised_test_dca.setStyleSheet(button_style)

        self.button_supervised_predict_dca = QPushButton(
            "Predict (DCA encoding)"
        )
        self.button_supervised_predict_dca.setMinimumWidth(80)
        self.button_supervised_predict_dca.setToolTip(
            "Purely supervised DCA (GREMLIN or PLMC) model prediction"
        )
        self.button_supervised_predict_dca.clicked.connect(
            self.pypef_dca_supervised_predict
        )
        self.button_supervised_predict_dca.setStyleSheet(button_style)


        self.button_supervised_train_onehot = QPushButton(
            "Train (One-hot encoding)"
        )
        self.button_supervised_train_onehot.setMinimumWidth(80)
        self.button_supervised_train_onehot.setToolTip(
            "Purely supervised one-hot model training "
            "on variant-fitness labels"
        )
        self.button_supervised_train_onehot.clicked.connect(
            self.pypef_onehot_supervised_train
        )
        self.button_supervised_train_onehot.setStyleSheet(button_style)

        self.button_supervised_train_test_onehot = QPushButton(
            "Train-Test (One-hot encoding)"
        )
        self.button_supervised_train_test_onehot.setMinimumWidth(80)
        self.button_supervised_train_test_onehot.setToolTip(
            "Purely supervised one-hot model training "
            "on variant-fitness labels"
        )
        self.button_supervised_train_test_onehot.clicked.connect(
            self.pypef_onehot_supervised_train_test
        )
        self.button_supervised_train_test_onehot.setStyleSheet(button_style)

        self.button_supervised_test_onehot = QPushButton(
            "Test (One-hot encoding)"
        )
        self.button_supervised_test_onehot.setMinimumWidth(80)
        self.button_supervised_test_onehot.setToolTip(
            "Purely supervised one-hot model test"
        )
        self.button_supervised_test_onehot.clicked.connect(
            self.pypef_onehot_supervised_test
        )
        self.button_supervised_test_onehot.setStyleSheet(button_style)

        self.button_supervised_predict_onehot = QPushButton(
            "Predict (One-hot encoding)"
        )
        self.button_supervised_predict_onehot.setMinimumWidth(80)
        self.button_supervised_predict_onehot.setToolTip(
            "Purely supervised one-hot model test"
        )
        self.button_supervised_predict_onehot.clicked.connect(
            self.pypef_onehot_supervised_predict
        )
        self.button_supervised_predict_onehot.setStyleSheet(button_style)

        # All buttons
        self.all_buttons = [
            self.button_work_dir,
            self.button_help,
            self.button_mklsts,
            self.button_mkps,
            self.button_dca_inference_gremlin,
            self.button_gremlin_ssm,
            self.button_dca_test_dca,
            self.button_llm_test_zs,
            self.button_dca_predict_dca,
            self.button_llm_predict_zs,
            self.button_hybrid_train_dca,
            self.button_hybrid_train_test_dca,
            self.button_hybrid_test_dca,
            self.button_hybrid_predict_dca,
            self.button_hybrid_train_dca_llm,
            self.button_hybrid_train_test_dca_llm,
            self.button_hybrid_test_dca_llm,
            self.button_hybrid_predict_dca_llm,
            self.button_llm_ssm,
            self.button_supervised_train_dca,
            self.button_supervised_train_test_dca,
            self.button_supervised_test_dca,
            self.button_supervised_predict_dca,
            self.button_supervised_train_onehot,
            self.button_supervised_train_test_onehot,
            self.button_supervised_test_onehot,
            self.button_supervised_predict_onehot
        ]

        # Layout widgets ################################################################
        # int fromRow, int fromColumn, int rowSpan, int columnSpan
        layout.addWidget(self.device_text_out, 0, 0, 1, 2)
        layout.addWidget(self.version_text, 0, 5, 1, 1)
        layout.addWidget(self.slider_text, 1, 0, 1, 1)
        layout.addWidget(self.button_work_dir, 0, 2, 1, 1)

        layout.addWidget(self.utils_text, 3, 0, 1, 1)
        layout.addWidget(self.button_help, 4, 0, 1, 1)
        layout.addWidget(self.button_mklsts, 5, 0, 1, 1)
        layout.addWidget(self.button_mkps, 6, 0, 1, 1)
        layout.addWidget(self.button_gremlin_ssm, 7, 0, 1, 1)
        layout.addWidget(self.button_llm_ssm, 8, 0, 1, 1)

        layout.addWidget(self.mklsts_cv_options_text, 1, 1, 1, 1)
        layout.addWidget(self.box_mklsts_cv, 2, 1, 1, 1)
        layout.addWidget(self.dca_text, 3, 1, 1, 1)
        layout.addWidget(self.button_dca_inference_gremlin, 4, 1, 1, 1)
        layout.addWidget(self.button_dca_test_dca, 5, 1, 1, 1)
        layout.addWidget(self.button_llm_test_zs, 6, 1, 1, 1)
        layout.addWidget(self.button_dca_predict_dca, 7, 1, 1, 1)
        layout.addWidget(self.button_llm_predict_zs, 8, 1, 1, 1)

        layout.addWidget(self.hybrid_text, 3, 2, 1, 1)
        layout.addWidget(self.button_hybrid_train_dca, 4, 2, 1, 1)
        layout.addWidget(self.button_hybrid_train_test_dca, 5, 2, 1, 1)
        layout.addWidget(self.button_hybrid_test_dca, 6, 2, 1, 1)
        layout.addWidget(self.button_hybrid_predict_dca, 7, 2, 1, 1)

        layout.addWidget(self.llm_text, 1, 3, 1, 1)
        layout.addWidget(self.box_llm, 2, 3, 1, 1)
        layout.addWidget(self.hybrid_dca_llm_text, 3, 3, 1, 1)
        layout.addWidget(self.button_hybrid_train_dca_llm, 4, 3, 1, 1)
        layout.addWidget(self.button_hybrid_train_test_dca_llm, 5, 3, 1, 1)
        layout.addWidget(self.button_hybrid_test_dca_llm, 6, 3, 1, 1)
        layout.addWidget(self.button_hybrid_predict_dca_llm, 7, 3, 1, 1)

        layout.addWidget(self.regression_model_text, 1, 4, 1, 1)
        layout.addWidget(self.box_regression_model, 2, 4, 1, 1)
        layout.addWidget(self.supervised_text, 3, 4, 1, 1)
        layout.addWidget(self.button_supervised_train_dca, 4, 4, 1, 1)
        layout.addWidget(self.button_supervised_train_test_dca, 5, 4, 1, 1)
        layout.addWidget(self.button_supervised_test_dca, 6, 4, 1, 1)
        layout.addWidget(self.button_supervised_predict_dca, 7, 4, 1, 1)
        layout.addWidget(self.button_supervised_train_onehot, 4, 5, 1, 1)
        layout.addWidget(self.button_supervised_train_test_onehot, 5, 5, 1, 1)
        layout.addWidget(self.button_supervised_test_onehot, 6, 5, 1, 1)
        layout.addWidget(self.button_supervised_predict_onehot, 7, 5, 1, 1)

        layout.addWidget(self.textedit_out, 12, 0, 1, 2)

        layout.addWidget(self.logTextBox.widget, 12, 2, 1, 4)

        # Start info thread #############################################################
        self.start_info_thread()

    def start_main_thread(self):
        self.version_text.setText("Running...")
        self.textedit_out.append(f"Executing command: {self.cmd}")
        self.__workers_done = 0
        self.__threads = []
        worker = Worker(0, cmd=self.cmd)
        thread = QThread()
        thread.setObjectName('thread_' + str(0) + '_MainThread')
        # Store refs to avoid garbage collection
        self.__threads.append((thread, worker))
        worker.moveToThread(thread)
        worker.sig_done.connect(self.on_worker_done)
        worker.sig_msg.connect(self.logTextBox.widget.appendPlainText)
        thread.started.connect(worker.work)
        thread.start()
    
    def start_info_thread(self):
        self.__info_workers_done = 0
        self.__info_threads = []
        info_worker = InfoWorker(1)
        info_thread = QThread()
        info_thread.setObjectName('thread_' + str(1) + '_InfoThread')
        info_worker.moveToThread(info_thread)
        # Store refs to avoid garbage collection
        self.__info_threads.append((info_thread, info_worker))
        info_worker.sig_tick.connect(self.handle_info_tick)
        info_thread.started.connect(info_worker.start)
        info_thread.start()

    def handle_info_tick(self, info_text: str):
        new_info = ""
        for i, s in enumerate(self.device_text_out_info_text.split("\n")):
            if i < len(self.device_text_out_info_text.split("\n")) - 1:
                new_info += s + "\n"
            else:
                new_info += info_text
        self.device_text_out.setPlainText(new_info)

    @Slot(int)
    def on_worker_done(self):
        self.end_process()
        self.__workers_done += 1
        if self.__workers_done == 1:
            for thread, _worker in self.__threads:
                thread.quit()
                thread.wait()

    @Slot()
    def abort_workers(self):
        # Currently, no aborts are happening as only single (big) tasks
        # are running in a single QThread without getting callbacks from 
        # a computing loop or so. So no qthreaded job abortions possible
        # without using QThread::terminate(), which should not be used.
        self.logTextBox.widget.appendPlainText(
            'Asking each worker to abort...'
        )
        for thread, worker in self.__threads:
            thread.quit()
            thread.wait()
        # even though threads have exited, there may still be messages 
        # on the main thread's queue (messages that threads emitted 
        # before the abort):
        self.logTextBox.widget.appendPlainText('All threads exited')

    def toggle_buttons(self, enabled: bool):
        for button in self.all_buttons:
            button.setEnabled(enabled)

    def start_process(self):
        self.target_button.setEnabled(False)
        self.logTextBox.widget.clear()
        self.c += 1
        k = f"Job: {str(self.c):<5}" + "=" * 60
        self.textedit_out.append(k)
        self.logTextBox.widget.appendPlainText(
            f"Current working directory: {str(getcwd())}"
        )
        self.logTextBox.widget.appendPlainText(
            "Job: " + str(self.c) + " " + "=" * 104
        )
        self.toggle_buttons(False)
    
    def end_process(self):
        self.target_button.setEnabled(True)
        self.toggle_buttons(True)
        self.textedit_out.append("=" * 60 + "\n")
        self.version_text.setText("Finished...")


    def closeEvent(self, event):
        """
        Overwriting self closeEvent (invoked on GUI window closing): 
        stop InfoWorker and associated threads
        """
        for thread, worker in self.__info_threads:
                worker.sig_abort.emit()  # stops timer
                QMetaObject.invokeMethod(worker, "stop", Qt.QueuedConnection)
                #worker.stop()
                thread.quit()
                thread.wait()
        event.accept()
    

    # Box selections ####################################################################
    def selection_ncores(self, i):
        if i == 0:
            self.n_cores = 1
        elif i == 1:
            self.n_cores = cpu_count()

    def selection_regression_model(self, i):
        self.regression_model = [
            r.lower() for r in self.regression_models
        ][i]

    def selection_llm_model(self, i):
        self.llm = [None, 'esm', 'prosst'][i]

    def selection_mklsts_splits(self, i):
        self.mklsts_cv_method = [
            '', '--random', '--modulo', '--cont', '--plot'
        ][i]

    def selection_ls_proportion(self, value):
        self.ls_proportion = value / 100
        self.slider_text.setText(
            f"Train set proportion: {self.ls_proportion}"
        )   

    def set_work_dir(self):
        self.working_directory = QFileDialog.getExistingDirectory(
            self.win2, 'Select Folder'
        )
        chdir(self.working_directory)
        self.logTextBox.widget.clear()
        self.logTextBox.widget.appendPlainText(
            f"Changed current working directory to: {str(getcwd())}"
        )

    # Button functions ##################################################################
    # Utils
    def pypef_help(self):
        self.target_button = self.button_help
        self.start_process()
        self.textedit_out.append(f'Executing command:\n    --help')
        self.version_text.setText("Getting help...")
        self.logTextBox.widget.appendPlainText(__doc__)
        self.end_process()

    def pypef_mklsts(self):
        self.target_button = self.button_mklsts
        self.start_process()
        wt_fasta_file = QFileDialog.getOpenFileName(
            self.win2, "Select WT FASTA File", 
            filter="FASTA file (*.fasta *.fa)"
        )[0]
        csv_variant_file = QFileDialog.getOpenFileName(
            self.win2, "Select variant CSV File", 
            filter="CSV file (*.csv)"
        )[0]
        if wt_fasta_file and csv_variant_file:
            self.version_text.setText("Running MKLSTS...")
            self.cmd = (
                f'mklsts --wt {wt_fasta_file} --input {csv_variant_file} '
                f'--ls_proportion {self.ls_proportion} {self.mklsts_cv_method}'
            )
            self.start_main_thread()
        else:
            self.end_process()

    def pypef_mkps(self):
        self.target_button = self.button_mkps
        self.start_process()
        wt_fasta_file = QFileDialog.getOpenFileName(
            self.win2, "Select WT FASTA File", 
            filter="FASTA file (*.fasta *.fa)"
        )[0]
        csv_variant_file = QFileDialog.getOpenFileName(
            self.win2, "Select variant CSV File", 
            filter="CSV file (*.csv)"
        )[0]
        if wt_fasta_file and csv_variant_file:
            self.version_text.setText("Running MKLSTS...")
            self.cmd = f'mkps --wt {wt_fasta_file} --input {csv_variant_file}'
            self.start_main_thread()
        else:
            self.end_process()

    def pypef_gremlin_ssm(self):
        self.target_button = self.button_gremlin_ssm
        self.start_process()
        wt_fasta_file = QFileDialog.getOpenFileName(
            self.win2, "Select WT FASTA File", 
            filter="FASTA file (*.fasta *.fa)"
        )[0]
        if wt_fasta_file:
            gremlin_pkl_file = QFileDialog.getOpenFileName(
                self.win2, "GREMLIN Pickle file", 
                filter="Pickle file (GREMLIN)"
            )[0]
            if gremlin_pkl_file:
                self.version_text.setText("Running GREMLIN (DCA) optimization on MSA...")
                self.cmd = f'predict_ssm --wt {wt_fasta_file} --params {gremlin_pkl_file}'
                self.start_main_thread()
            else:
                self.end_process()
        else:
            self.end_process()
    
    def pypef_llm_ssm(self):
        self.target_button = self.button_llm_ssm
        self.start_process()
        wt_fasta_file = QFileDialog.getOpenFileName(
            self.win2, "Select WT FASTA File", 
            filter="FASTA file (*.fasta *.fa)"
        )[0]
        if wt_fasta_file:
            if self.llm == 'prosst':
                pdb_file = QFileDialog.getOpenFileName(
                    self.win2, "Select PDB protein structure File",
                    filter="PDB file (*.pdb)"
                )[0]
                if pdb_file:
                    self.version_text.setText(
                        "ProSST zero shot model inference..."
                    )
                    self.cmd = (
                        f'predict_ssm --llm {self.llm} '
                        f'--wt {wt_fasta_file} --pdb {pdb_file}'
                        )
                    self.start_main_thread()
                else:
                    self.end_process()
            elif self.llm == 'esm':
                self.cmd = f'predict_ssm --llm {self.llm} --wt {wt_fasta_file}'
                self.start_main_thread()
            else:
                self.logTextBox.widget.appendPlainText(
                    "Provide a LLM option for modeling."
                )
                self.end_process()
        else:
            self.end_process()

    # Unsupervised/Zero-Shot/DCA
    def pypef_gremlin(self):
        self.target_button = self.button_dca_inference_gremlin
        self.start_process()
        wt_fasta_file = QFileDialog.getOpenFileName(
            self.win2, "Select WT FASTA File", 
            filter="FASTA file (*.fasta *.fa)"
        )[0]
        msa_file = QFileDialog.getOpenFileName(
            self.win2, 
            ("Select Multiple Sequence Alignment (MSA) "
            "file (in FASTA or A2M format)"),
            filter="MSA file (*.fasta *.a2m)"
        )[0]
        if wt_fasta_file and msa_file:
            self.version_text.setText("Running GREMLIN (DCA) optimization on MSA...")
            self.cmd = f'param_inference --wt {wt_fasta_file} --msa {msa_file}'
            self.start_main_thread()
        else:
            self.end_process()

    def pypef_dca_test(self):
        self.target_button = self.button_dca_test_dca
        self.start_process()
        test_set_file = QFileDialog.getOpenFileName(
            self.win2, "Select Test Set File in \"FASL\" format", 
            filter="FASL file (*.fasl)"
        )[0]
        params_pkl_file = QFileDialog.getOpenFileName(
            self.win2, "Parameter Pickle file", 
            filter="Pickle file (*.params GREMLIN PLMC)"
        )[0]
        if test_set_file and params_pkl_file:
            self.version_text.setText(
                "Testing DCA performance on provided test set..."
            )
            self.cmd = (f'hybrid --ts {test_set_file} -m {params_pkl_file} '
                        f'--params {params_pkl_file}')
            self.start_main_thread()
        else:
            self.end_process()

    def pypef_llm_test(self):
        self.target_button = self.button_llm_test_zs
        self.start_process()
        test_set_file = QFileDialog.getOpenFileName(
            self.win2, "Select Test Set File in \"FASL\" format", 
            filter="FASL file (*.fasl)"
        )[0]
        if test_set_file:
            if self.llm == 'prosst':
                wt_fasta_file = QFileDialog.getOpenFileName(
                    self.win2, "Select WT FASTA File",
                    filter="FASTA file (*.fasta *.fa)"
                )[0]
                pdb_file = QFileDialog.getOpenFileName(
                    self.win2, "Select PDB protein structure File",
                    filter="PDB file (*.pdb)"
                )[0]
                if wt_fasta_file and pdb_file:
                    self.version_text.setText(
                        "ProSST zero shot model inference..."
                    )
                    self.cmd = (
                        f'hybrid --ts {test_set_file} --llm {self.llm} '
                        f'--wt {wt_fasta_file} --pdb {pdb_file}'
                        )
                    self.start_main_thread()
                else:
                    self.end_process()
            elif self.llm == 'esm':
                self.cmd = f'hybrid --ts {test_set_file} --llm {self.llm}'
                self.start_main_thread()
            else:
                self.logTextBox.widget.appendPlainText(
                    "Provide a LLM option for modeling."
                )
                self.end_process()
        else:
            self.end_process()

    def pypef_dca_predict(self):
        self.target_button = self.button_dca_predict_dca
        self.start_process()
        prediction_file = QFileDialog.getOpenFileName(
            self.win2, "Select Prediction Set File in FASTA format",
            filter="FASTA file (*.fasta *.fa)"
        )[0]
        params_pkl_file = QFileDialog.getOpenFileName(
            self.win2, "Select DCA parameter Pickle file",
            filter="Pickle file (*.params GREMLIN PLMC)"
        )[0]
        if prediction_file and params_pkl_file:
            self.version_text.setText(
                "Predicting using the DCA model on provided prediction set..."
            )
            self.cmd = (
                f'hybrid --ps {prediction_file} '
                f'-m {params_pkl_file} --params {params_pkl_file}'
            )
            self.start_main_thread()
        else:
            self.end_process()

    def pypef_llm_predict(self):
        self.target_button = self.button_llm_predict_zs
        self.start_process()
        prediction_file = QFileDialog.getOpenFileName(
            self.win2, "Select Prediction Set File in FASTA format",
            filter="FASTA file (*.fasta *.fa)"
        )[0]
        if prediction_file:
            if self.llm == 'prosst':
                wt_fasta_file = QFileDialog.getOpenFileName(
                    self.win2, "Select WT FASTA File",
                    filter="FASTA file (*.fasta *.fa)"
                )[0]
                pdb_file = QFileDialog.getOpenFileName(
                    self.win2, "Select PDB protein structure File",
                    filter="PDB file (*.pdb)"
                )[0]
                if wt_fasta_file and pdb_file:
                    self.version_text.setText(
                        "ProSST zero shot model inference..."
                    )
                    self.cmd = (
                        f'hybrid --ps {prediction_file} --llm {self.llm} '
                        f'--wt {wt_fasta_file} --pdb {pdb_file}'
                        )
                    self.start_main_thread()
                else:
                    self.end_process()
            elif self.llm == 'esm':
                self.cmd = f'hybrid --ps {prediction_file} --llm {self.llm}'
                self.start_main_thread()
            else:
                self.logTextBox.widget.appendPlainText(
                    "Provide a LLM option for modeling."
                )
                self.end_process()
        else:
            self.end_process()

    # Supervised/Hybrid
    def pypef_dca_hybrid_train(self):
        self.target_button = self.button_hybrid_train_dca
        self.start_process()
        training_file = QFileDialog.getOpenFileName(
            self.win2, "Select Training Set File in \"FASL\" format",
            filter="FASL file (*.fasl)"
        )[0]
        params_pkl_file = QFileDialog.getOpenFileName(
            self.win2, "Select DCA parameter Pickle file",
            filter="Pickle file (*.params GREMLIN PLMC)"
        )[0]
        if training_file and params_pkl_file:
            self.version_text.setText("Hybrid (DCA-supervised) model training...")
            self.cmd = (
                f'hybrid --ls {training_file} --ts {training_file} '
                f'-m {params_pkl_file} --params {params_pkl_file}'
            )
            self.start_main_thread()
        else:
            self.end_process()

    def pypef_dca_hybrid_train_test(self):
        self.target_button = self.button_hybrid_train_test_dca
        self.start_process()
        training_file = QFileDialog.getOpenFileName(
            self.win2, "Select Training Set File in \"FASL\" format",
            filter="FASL file (*.fasl)"
        )[0]
        test_file = QFileDialog.getOpenFileName(
            self.win2, "Select Test Set File in \"FASL\" format",
            filter="FASL file (*.fasl)"
        )[0]
        params_pkl_file = QFileDialog.getOpenFileName(
            self.win2, "Select DCA parameter Pickle file",
            filter="Pickle file (*.params GREMLIN PLMC)"
        )[0]
        if training_file and test_file and params_pkl_file:
            self.version_text.setText(
                "Hybrid (DCA-supervised) model training and testing..."
            )
            self.cmd = (
                f'hybrid -m {params_pkl_file} --ls {training_file} '
                f'--ts {test_file} --params {params_pkl_file}'
            )
            self.start_main_thread()
        else:
            self.end_process()

    def pypef_dca_hybrid_test(self):
        self.target_button = self.button_hybrid_test_dca
        self.start_process()        
        test_file = QFileDialog.getOpenFileName(
            self.win2, "Select Test Set File in \"FASL\" format",
            filter="FASL file (*.fasl)"
        )[0]
        model_pkl_file = QFileDialog.getOpenFileName(
            self.win2, "Select Hybrid Model file in Pickle format",
            filter="Pickle file (HYBRID*)"
        )[0]
        params_pkl_file = QFileDialog.getOpenFileName(
            self.win2, "Select DCA parameter Pickle file",
            filter="Pickle file (*.params GREMLIN PLMC)"
        )[0]
        if test_file and params_pkl_file:
            self.version_text.setText("Hybrid (DCA-supervised) model testing...")
            self.cmd = (f'hybrid -m {model_pkl_file} --ts {test_file} '
                        f'--params {params_pkl_file}')
            self.start_main_thread()
        else:
            self.end_process()

    def pypef_dca_hybrid_predict(self):
        self.target_button = self.button_hybrid_predict_dca
        self.start_process()    
        prediction_file = QFileDialog.getOpenFileName(
            self.win2, "Select Prediction Set File in FASTA format",
            filter="FASTA file (*.fasta *.fa)"
        )[0]
        model_file = QFileDialog.getOpenFileName(
            self.win2, "Select Hybrid Model file in Pickle format",
            filter="Pickle file (HYBRID*)"
        )[0]
        params_pkl_file = QFileDialog.getOpenFileName(
            self.win2, "Select DCA parameter Pickle file",
            filter="Pickle file (*.params GREMLIN PLMC)"
        )[0]
        if prediction_file and params_pkl_file:
            self.version_text.setText(
                "Predicting using the hybrid (DCA-supervised) model..."
            )
            self.cmd = (
                f'hybrid -m {model_file} --ps {prediction_file} '
                f'--params {params_pkl_file}'
            )
            self.start_main_thread()
        else:
            self.end_process()

    def pypef_dca_llm_hybrid_train(self):
        self.target_button = self.button_hybrid_train_dca_llm
        self.start_process()  
        training_file = QFileDialog.getOpenFileName(
            self.win2, "Select Training Set File in \"FASL\" format",
            filter="FASL file (*.fasl)"
        )[0]
        params_pkl_file = QFileDialog.getOpenFileName(
            self.win2, "Select DCA parameter Pickle file",
            filter="Pickle file (*.params GREMLIN PLMC)"
        )[0]
        if self.llm == 'prosst':
            wt_fasta_file = QFileDialog.getOpenFileName(
                self.win2, "Select WT FASTA File",
                filter="FASTA file (*.fasta *.fa)"
            )[0]
            pdb_file = QFileDialog.getOpenFileName(
                self.win2, "Select PDB protein structure File",
                filter="PDB file (*.pdb)"
            )[0]
            if training_file and params_pkl_file and wt_fasta_file and pdb_file:
                self.version_text.setText(
                    "Hybrid (DCA+LLM-supervised) model training..."
                )
                self.cmd = (
                    f'hybrid --ls {training_file} --ts {training_file} '
                    f'--params {params_pkl_file} --llm {self.llm} '
                    f'--wt {wt_fasta_file} --pdb {pdb_file}'
                )
                self.start_main_thread()
            else:
                self.end_process()
        elif self.llm == 'esm':
            if training_file and params_pkl_file:
                self.version_text.setText(
                    "Hybrid (DCA+LLM-supervised) model training..."
                )
                self.cmd = (
                    f'hybrid --ls {training_file} --ts {training_file} '
                    f'--params {params_pkl_file} --llm {self.llm}'
                )
                self.start_main_thread()
            else:
                self.end_process()
        else:
            self.logTextBox.widget.appendPlainText("Provide a LLM option for modeling.")
            self.end_process()

    def pypef_dca_llm_hybrid_train_test(self):
        self.target_button = self.button_hybrid_train_test_dca_llm
        self.start_process()  
        training_file = QFileDialog.getOpenFileName(
            self.win2, "Select Training Set File in \"FASL\" format",
            filter="FASL file (*.fasl)"
        )[0]
        test_file = QFileDialog.getOpenFileName(
            self.win2, "Select Test Set File in \"FASL\" format",
            filter="FASL file (*.fasl)"
        )[0]
        params_pkl_file = QFileDialog.getOpenFileName(
            self.win2, "Select DCA parameter Pickle file",
            filter="Pickle file (*.params GREMLIN PLMC)"
        )[0]
        if self.llm == 'prosst':
            wt_fasta_file = QFileDialog.getOpenFileName(
                self.win2, "Select WT FASTA File",
                filter="FASTA file (*.fasta *.fa)"
            )[0]
            pdb_file = QFileDialog.getOpenFileName(
                self.win2, "Select PDB protein structure File",
                filter="PDB file (*.pdb)"
            )[0]
            if (
                training_file and test_file and params_pkl_file 
                and wt_fasta_file and pdb_file
            ):
                self.version_text.setText(
                    "Hybrid (DCA+LLM-supervised) model training..."
                )
                self.cmd = (
                    f'hybrid --ls {training_file} --ts {test_file} '
                    f'--params {params_pkl_file} --llm {self.llm} '
                    f'--wt {wt_fasta_file} --pdb {pdb_file}'
                )
                self.start_main_thread()
            else:
                self.end_process()
        elif self.llm == 'esm':
            if training_file and test_file and params_pkl_file:
                self.version_text.setText(
                    "Hybrid (DCA+LLM-supervised) model training..."
                )
                self.cmd = (
                    f'hybrid --ls {training_file} --ts {test_file} '
                    f'--params {params_pkl_file} --llm {self.llm}'
                )
                self.start_main_thread()
            else:
                self.end_process()
        else:
            self.logTextBox.widget.appendPlainText("Provide a LLM option for modeling.")
            self.end_process()

    def pypef_dca_llm_hybrid_test(self):
        self.target_button = self.button_hybrid_test_dca_llm
        self.start_process()  
        test_file = QFileDialog.getOpenFileName(
            self.win2, "Select Test Set File in \"FASL\" format",
            filter="FASL file (*.fasl)"
        )[0]
        model_file = QFileDialog.getOpenFileName(
            self.win2, "Select Hybrid Model file in Pickle format",
            filter="Pickle file (HYBRID*)"
        )[0]
        params_pkl_file = QFileDialog.getOpenFileName(
            self.win2, "Select DCA parameter Pickle file",
            filter="Pickle file (*.params GREMLIN PLMC)"
        )[0]
        if self.llm == 'prosst':
            wt_fasta_file = QFileDialog.getOpenFileName(
                self.win2, "Select WT FASTA File",
                filter="FASTA file (*.fasta *.fa)"
            )[0]
            pdb_file = QFileDialog.getOpenFileName(
                self.win2, "Select PDB protein structure File",
                filter="PDB file (*.pdb)"
            )[0]
            if (
                test_file and params_pkl_file and wt_fasta_file 
                and pdb_file and model_file
            ):
                self.version_text.setText(
                    "Hybrid (DCA+LLM-supervised) model testing..."
                )
                self.cmd = (
                    f'hybrid -m {model_file} --ts {test_file} '
                    f'--params {params_pkl_file} --llm {self.llm} '
                    f'--wt {wt_fasta_file} --pdb {pdb_file}')
                self.start_main_thread()
            else:
                self.end_process()
        elif self.llm == 'esm':
            if test_file and params_pkl_file and model_file:
                self.version_text.setText(
                    "Hybrid (DCA+LLM-supervised) model testing..."
                )
                self.cmd = (
                    f'hybrid -m {model_file} --ts {test_file} '
                    f'--params {params_pkl_file} --llm {self.llm}')
                self.start_main_thread()
            else:
                self.end_process()
        else:
            self.logTextBox.widget.appendPlainText("Provide a LLM option for modeling.")
            self.end_process()

    def pypef_dca_llm_hybrid_predict(self):
        self.target_button = self.button_hybrid_predict_dca_llm
        self.start_process()  
        prediction_file = QFileDialog.getOpenFileName(
            self.win2, "Select Prediction Set File in FASTA format",
            filter="FASTA file (*.fasta *.fa)"
        )[0]
        model_file = QFileDialog.getOpenFileName(
            self.win2, "Select Hybrid Model file in Pickle format",
            filter="Pickle file (HYBRID*)"
        )[0]
        params_pkl_file = QFileDialog.getOpenFileName(
            self.win2, "Select DCA parameter Pickle file",
            filter="Pickle file (*.params GREMLIN PLMC)"
        )[0]
        if self.llm == 'prosst':
            wt_fasta_file = QFileDialog.getOpenFileName(
                self.win2, "Select WT FASTA File",
                filter="FASTA file (*.fasta *.fa)"
            )[0]
            pdb_file = QFileDialog.getOpenFileName(
                self.win2, "Select PDB protein structure File",
                filter="PDB file (*.pdb)"
            )[0]
            if (
                prediction_file and params_pkl_file and wt_fasta_file 
                and pdb_file and model_file
            ):
                self.version_text.setText(
                    "Hybrid (DCA+LLM-supervised) model training..."
                )
                self.cmd = (
                    f'hybrid -m {model_file} --ps {prediction_file} '
                    f'--params {params_pkl_file} --llm {self.llm} '
                    f'--wt {wt_fasta_file} --pdb {pdb_file}'
                )
                self.start_main_thread()
            else:
                self.end_process()
        elif self.llm == 'esm':
            if prediction_file and params_pkl_file and model_file:
                self.version_text.setText(
                    "Hybrid (DCA+LLM-supervised) model training..."
                )
                self.cmd = (
                    f'hybrid -m {model_file} --ps {prediction_file} '
                    f'--params {params_pkl_file} --llm {self.llm}'
                )
                self.start_main_thread()
            else:
                self.end_process()
        else:
            self.logTextBox.widget.appendPlainText("Provide a LLM option for modeling.")
            self.end_process()

    def pypef_dca_supervised_train(self):
        self.target_button = self.button_supervised_train_dca
        self.start_process()  
        training_file = QFileDialog.getOpenFileName(
            self.win2, "Select Training Set File in \"FASL\" format",
            filter="FASL file (*.fasl)"
        )[0]
        params_pkl_file = QFileDialog.getOpenFileName(
            self.win2, "Select DCA parameter Pickle file",
            filter="Pickle file (*.params GREMLIN PLMC)"
        )[0]
        if training_file and params_pkl_file:
            self.version_text.setText(
                "Hybrid (DCA-supervised) model training and testing..."
            )
            self.cmd = (
                f'ml --encoding dca --ls {training_file} '
                f'--ts {training_file} --params {params_pkl_file} '
                f'--threads {self.n_cores} --regressor {self.regression_model}'
            )
            self.start_main_thread()
        else:
            self.end_process()

    def pypef_dca_supervised_train_test(self):
        self.target_button = self.button_supervised_train_test_dca
        self.start_process()  
        training_file = QFileDialog.getOpenFileName(
            self.win2, "Select Training Set File in \"FASL\" format",
            filter="FASL file (*.fasl)"
        )[0]
        test_file = QFileDialog.getOpenFileName(
            self.win2, "Select Test Set File in \"FASL\" format",
            filter="FASL file (*.fasl)"
        )[0]
        params_pkl_file = QFileDialog.getOpenFileName(
            self.win2, "Select DCA parameter Pickle file",
            filter="Pickle file (*.params GREMLIN PLMC)"
        )[0]
        if training_file and test_file and params_pkl_file:
            self.version_text.setText(
                "Hybrid (DCA-supervised) model training and testing..."
            )
            self.cmd = (
                f'ml --encoding dca --ls {training_file} '
                f'--ts {test_file} --params {params_pkl_file} '
                f'--threads {self.n_cores} --regressor {self.regression_model}'
            )
            self.start_main_thread()
        else:
            self.end_process()

    def pypef_dca_supervised_test(self):
        self.target_button = self.button_supervised_test_dca
        self.start_process()  
        test_file = QFileDialog.getOpenFileName(
            self.win2, "Select Test Set File in \"FASL\" format",
            filter="FASL file (*.fasl)")[0]
        model_file = QFileDialog.getOpenFileName(
            self.win2, "Select ML Model file in Pickle format",
            filter="Pickle file (ML*)"
        )[0]
        params_pkl_file = QFileDialog.getOpenFileName(
            self.win2, "Select DCA parameter Pickle file",
            filter="Pickle file (*.params GREMLIN PLMC)"
        )[0]
        if test_file and params_pkl_file and model_file:
            self.version_text.setText(
                "Hybrid (DCA-supervised) model testing..."
            )
            self.cmd = (
                f'ml -m {model_file} --encoding dca '
                f'--ts {test_file} --params {params_pkl_file} '
                f'--threads {self.n_cores}'
            )
            self.start_main_thread()
        else:
            self.end_process()

    def pypef_dca_supervised_predict(self):
        self.target_button = self.button_supervised_predict_dca
        self.start_process()  
        prediction_file = QFileDialog.getOpenFileName(
            self.win2, "Select Prediction Set File in FASTA format",
            filter="FASTA file (*.fasta *.fa)"
        )[0]
        model_file = QFileDialog.getOpenFileName(
            self.win2, "Select ML Model file in Pickle format",
            filter="Pickle file (ML*)"
        )[0]
        params_pkl_file = QFileDialog.getOpenFileName(
            self.win2, "Select DCA parameter Pickle file",
            filter="Pickle file (*.params GREMLIN PLMC)"
        )[0]
        if prediction_file and params_pkl_file and model_file:
            self.version_text.setText("Hybrid (DCA-supervised) model prediction...")
            self.cmd = (
                f'ml -m {model_file} --encoding dca --ps {prediction_file} '
                f'--params {params_pkl_file} --threads {self.n_cores}'
            )
            self.start_main_thread()
        else:
            self.end_process()

    def pypef_onehot_supervised_train(self):
        self.target_button = self.button_supervised_train_onehot
        self.start_process()  
        training_file = QFileDialog.getOpenFileName(
            self.win2, "Select Training Set File in \"FASL\" format",
            filter="FASL file (*.fasl)"
        )[0]
        if training_file:
            self.version_text.setText(
                "Hybrid (DCA-supervised) model training..."
            )
            self.cmd = (
                f'ml --encoding onehot --ls {training_file} --ts {training_file} '
                f'--threads {self.n_cores} --regressor {self.regression_model}'
            )
            self.start_main_thread()
        else:
            self.end_process()

    def pypef_onehot_supervised_train_test(self):
        self.target_button = self.button_supervised_train_test_onehot
        self.start_process()  
        training_file = QFileDialog.getOpenFileName(
            self.win2, "Select Training Set File in \"FASL\" format",
            filter="FASL file (*.fasl)"
        )[0]
        test_file = QFileDialog.getOpenFileName(
            self.win2, "Select Test Set File in \"FASL\" format",
            filter="FASL file (*.fasl)"
        )[0]
        if training_file and test_file:
            self.version_text.setText(
                "Hybrid (DCA-supervised) model training and testing..."
            )
            self.cmd = (
                f'ml --encoding onehot --ls {training_file} --ts {test_file} '
                f'--threads {self.n_cores} --regressor {self.regression_model}'
            )
            self.start_main_thread()
        else:
            self.end_process()

    def pypef_onehot_supervised_test(self):
        self.target_button = self.button_supervised_train_test_onehot
        self.start_process()  
        model_file = QFileDialog.getOpenFileName(
            self.win2, "Select Onehot Model file in Pickle format",
            filter="Pickle file (ONEHOT*)"
        )[0]
        test_file = QFileDialog.getOpenFileName(
            self.win2, "Select Test Set File in \"FASL\" format",
            filter="FASL file (*.fasl)"
        )[0]
        if test_file and model_file:
            self.version_text.setText("Hybrid (DCA-supervised) model testing...")
            self.cmd = (
                f'ml -m {model_file} --encoding onehot --ts {test_file} '
                f'--threads {self.n_cores}'
            )
            self.start_main_thread()
        else:
            self.end_process()

    def pypef_onehot_supervised_predict(self):
        self.target_button = self.button_supervised_predict_onehot
        self.start_process()  
        model_file = QFileDialog.getOpenFileName(
            self.win2, "Select Onehot Model file in Pickle format",
            filter="Pickle file (ONEHOT*)"
        )[0]
        prediction_file = QFileDialog.getOpenFileName(
            self.win2, "Select Prediction Set File in FASTA format",
            filter="FASTA file (*.fasta *.fa)"
        )[0]
        if prediction_file and model_file:
            self.version_text.setText("Hybrid (DCA-supervised) model prediction...")
            self.cmd = (
                f'ml -m {model_file} --encoding onehot --ps {prediction_file} '
                f'--threads {self.n_cores}'
            )
            self.start_main_thread()
        else:
            self.end_process()


def run_app():
    app = QApplication([])
    widget = MainWidget()
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()

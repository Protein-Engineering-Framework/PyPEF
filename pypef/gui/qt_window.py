# PyPEF - Pythonic Protein Engineering Framework
# https://github.com/niklases/PyPEF

# Qt GUI window using PySide6

import os
import sys
from os import getcwd, cpu_count, chdir
import logging
import time


from PySide6.QtCore import QObject, QThread, QSize, Qt, QRect, QTimer, Signal, Slot, QMetaObject
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QPushButton, QTextEdit, QVBoxLayout, QWidget,
    QGridLayout, QLabel, QPlainTextEdit, QSlider, QComboBox,
    QFileDialog, QProgressBar, QCheckBox
)

from pypef import __version__
from pypef.main import __doc__, run_main, logger, formatter
from pypef.utils.helpers import get_device, get_vram, get_torch_version, get_nvidia_gpu_info_pynvml


def get_logo_path():
    """Return the absolute path to the bundled PyPEF window-icon logo.

    Works when running from source, from a pip install (the file is shipped as
    package-data, see pyproject.toml), and from a frozen PyInstaller build
    (sys._MEIPASS). Returns None if the asset cannot be located so the caller
    can fall back to the default Qt icon without crashing.

    Note: the icon shows on X11 / Windows / macOS. Wayland compositors ignore
    Qt's setWindowIcon(); run with QT_QPA_PLATFORM=xcb to see it under Wayland.
    """
    candidates = []
    # PyInstaller onefile/onedir: assets are unpacked under sys._MEIPASS
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        candidates.append(os.path.join(meipass, 'pypef', 'gui', 'assets', 'pypef_logo.jpg'))
    # Source / pip install: next to this module
    candidates.append(os.path.join(os.path.dirname(__file__), 'assets', 'pypef_logo.jpg'))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


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


progress_style = """
QProgressBar {
    border: 1px solid #444;
    border-radius: 6px;
    background-color: #2b2b2b;
    text-align: center;
    height: 14px;
}

QProgressBar::chunk {
    background-color: #3daee9;
    border-radius: 6px;
}
"""



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
    sig_step = Signal(dict)
    sig_done = Signal(int)
    sig_msg = Signal(str)
    sig_abort = Signal(int)

    def __init__(self, id_: int, cmd):
        super().__init__()
        self.__id = id_
        self.cmd =  cmd
        self._abort = False

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

        def progress_cb(epoch, batch, epoch_total, batch_total, loss):
            progress = {'epoch': epoch, 'batch': batch, 'loss': loss,
                        'epoch_total': epoch_total, 'batch_total': batch_total }
            self.sig_step.emit(progress)
        
        def abort_cb():
            return self._abort

        run_main(argv=self.cmd, progress_cb=progress_cb, abort_cb=abort_cb)
        self.sig_done.emit(f"Done: {self.__id}")

    def abort(self):
        self._abort = True
        self.sig_msg.emit(f'Worker #{self.__id} notified to abort')


class InfoWorker(QObject):
    """
    Class for the Worker that gets GPU information.
    """
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
        self.shift = 2
        self.setMinimumSize(QSize(1400, 800))
        self.setWindowTitle("PyPEF GUI")
        logo_path = get_logo_path()
        if logo_path is not None:
            self.setWindowIcon(QIcon(logo_path))
        self.setStyleSheet("background-color: rgb(40, 44, 52);")
        self.win2 = SecondWindow()

        QThread.currentThread().setObjectName('main')
        self.__workers_done = None
        self.__threads = None

        self.train_start = None

        self._train_start_time = None
        self._last_eta_update = 0


        # Texts #########################################################################
        layout = QGridLayout(self)  # MAIN LAYOUT: QGridLayout
        self.version_text = QLabel(f"PyPEF v. {__version__}", alignment=Qt.AlignRight)
        self.working_directory_text = QLabel(f"{getcwd()}")
        self.working_directory_text.setWordWrap(True)
        self.working_directory_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.plm_text = QLabel("PLM")
        self.regression_model_text =  QLabel("Regression model")
        self.utils_text = QLabel("Utilities")
        self.mklsts_cv_options_text = QLabel("Cross-validation split options")
        self.dca_text = QLabel("DCA & PLM (unsupervised)")
        self.hybrid_text = QLabel("Hybrid (supervised DCA)")
        self.hybrid_dca_llm_text = QLabel("Hybrid (supervised DCA+PLM)")
        self.supervised_text = QLabel("Purely supervised")
        self.slider_text = QLabel("Train set proportion: 0.8")
        self.epoch_time_label = QLabel("", self)
        self.batch_time_label = QLabel("", self)

        for txt in [
            self.version_text, self.working_directory_text, self.regression_model_text, 
            self.utils_text, self.plm_text, self.dca_text, self.hybrid_text, 
            self.supervised_text, self.hybrid_text, self.hybrid_dca_llm_text,
            self.slider_text
        ]:
            txt.setStyleSheet(text_style)

        text_out_style = ("font-family:Consolas;font-size:12px;font-weight:normal;color:white;"
                          "background-color:rgb(54, 69, 79);border:2px solid rgb(52, 59, 72);")
        self.device_text_out = QTextEdit(readOnly=True)
        self.device_text_out.setStyleSheet(text_out_style)
        self.device_text_out.setFixedHeight(85)
        self.device_text_out_info_text = (
            f"Device (for PLM/DCA): {get_device().upper()}\n"
            f"{get_nvidia_gpu_info_pynvml()[0]}\n"
            f"PyTorch version: {get_torch_version()}\n"
            f"Driver version: {get_nvidia_gpu_info_pynvml()[1]}\n"
            f"{get_vram(verbose=False)[0]}"
        )
        self.device_text_out.setPlainText(self.device_text_out_info_text)

        self.textedit_out = QTextEdit(readOnly=True)
        self.textedit_out.setStyleSheet(text_out_style)
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

        self.epoch_progress_bar = QProgressBar()
        self.epoch_progress_bar.setTextVisible(False)
        self.epoch_progress_bar.setStyleSheet(progress_style)

        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setTextVisible(False)
        self.batch_progress_bar.setStyleSheet(progress_style)

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
        self.box_llm.addItems(['None', 'ESM', 'ProSST', 'ESM+ProSST'])
        self.box_llm.currentIndexChanged.connect(self.selection_llm_model)
        self.box_llm.setCurrentIndex(1)
        self.box_llm.setStyleSheet("color:white;background-color:rgb(54, 69, 79);")

        # DCA+PLM (supervised) hybrid training options (map to the --lora / --gauss_opt /
        # --gauss_comb CLI flags). Only relevant for (Train) and (Train-Test) DCA+PLM runs.
        checkbox_style = "color:white;"
        self.check_lora = QCheckBox("LoRA tuning")
        self.check_lora.setToolTip(
            "Use LoRA-based supervised fine-tuning of the PLM (training only)."
        )
        self.check_lora.setStyleSheet(checkbox_style)
        self.check_gauss_opt = QCheckBox("GP optimization")
        self.check_gauss_opt.setToolTip(
            "Use a Gaussian process (GP) to optimize the PLM embeddings and zero-shot "
            "scores as an alternative to LoRA tuning (training only; requires a WT FASTA "
            "and a PDB structure file)."
        )
        self.check_gauss_opt.setStyleSheet(checkbox_style)
        self.check_gauss_comb = QCheckBox("Combined GP")
        self.check_gauss_comb.setToolTip(
            "Additionally build a combined GP over the embeddings of both PLMs "
            "(requires GP optimization and two PLMs, i.e. ESM+ProSST)."
        )
        self.check_gauss_comb.setStyleSheet(checkbox_style)
        self.box_plm_options = QWidget()
        _plm_options_layout = QVBoxLayout(self.box_plm_options)
        _plm_options_layout.setContentsMargins(0, 0, 0, 0)
        _plm_options_layout.setSpacing(0)
        for _cb in (self.check_lora, self.check_gauss_opt, self.check_gauss_comb):
            _plm_options_layout.addWidget(_cb)

        self.box_mklsts_cv = QComboBox()
        self.box_mklsts_cv.addItems([
            'None', 'Random split', 'Modulo split', 
            'Continuous split', 'Plot distributions'
        ])
        self.box_mklsts_cv.currentIndexChanged.connect(self.selection_mklsts_splits)
        self.box_mklsts_cv.setStyleSheet("color:white;background-color:rgb(54, 69, 79);")
        
        # Buttons #######################################################################
        # Utilities
        self.button_abort = QPushButton("Stop training")
        self.button_abort.clicked.connect(self.abort_workers)
        self.button_abort.setStyleSheet(button_style)

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

        self.button_llm_ssm = QPushButton("PLM SSM prediction")
        self.button_llm_ssm.setMinimumWidth(80)
        self.button_llm_ssm.setToolTip(
            "Runs full site-saturation (single) mutagenesis using the selected PLM predcitor "
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

        # Zero-shot PLM
        self.button_llm_test_zs = QPushButton("Test (PLM)")
        self.button_llm_test_zs.setMinimumWidth(80)
        self.button_llm_test_zs.setToolTip(
            "Test performance on any test dataset using "
            "the PLM model for zero-shot prediction"
        )
        self.button_llm_test_zs.clicked.connect(self.pypef_llm_test)
        self.button_llm_test_zs.setStyleSheet(button_style)

        self.button_llm_predict_zs = QPushButton("Predict (PLM)")
        self.button_llm_predict_zs.setMinimumWidth(80)
        self.button_llm_predict_zs.setToolTip(
            "Test performance on any test dataset using "
            "the PLM model for zero-shot prediction"
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

        # Hybrid DCA+PLM
        self.button_hybrid_train_dca_llm = QPushButton("Train (DCA+PLM)")
        self.button_hybrid_train_dca_llm.setMinimumWidth(80)
        self.button_hybrid_train_dca_llm.setToolTip(
            "Optimize the GREMLIN model and tune the PLM by "
            "supervised training on variant-fitness labels"
        )
        self.button_hybrid_train_dca_llm.clicked.connect(
            self.pypef_dca_llm_hybrid_train
        )
        self.button_hybrid_train_dca_llm.setStyleSheet(button_style)

        self.button_hybrid_train_test_dca_llm = QPushButton("Train-Test (DCA+PLM)")
        self.button_hybrid_train_test_dca_llm.setMinimumWidth(80)
        self.button_hybrid_train_test_dca_llm.setToolTip(
            "Optimize the GREMLIN model and tune the PLM by supervised "
            "training on variant-fitness labels and testing the model "
            "on a test set"
        )
        self.button_hybrid_train_test_dca_llm.clicked.connect(
            self.pypef_dca_llm_hybrid_train_test
        )
        self.button_hybrid_train_test_dca_llm.setStyleSheet(button_style)

        self.button_hybrid_test_dca_llm = QPushButton("Test (DCA+PLM)")
        self.button_hybrid_test_dca_llm.setMinimumWidth(80)
        self.button_hybrid_test_dca_llm.setToolTip(
            "Test the trained hybrid DCA+PLM model on a test set"
        )
        self.button_hybrid_test_dca_llm.clicked.connect(
            self.pypef_dca_llm_hybrid_test
        )
        self.button_hybrid_test_dca_llm.setStyleSheet(button_style)

        self.button_hybrid_predict_dca_llm = QPushButton("Predict (DCA+PLM)")
        self.button_hybrid_predict_dca_llm.setMinimumWidth(80)
        self.button_hybrid_predict_dca_llm.setToolTip(
            "Use the trained hybrid DCA+PLM model for prediction"
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
        layout.addWidget(self.working_directory_text, 0, 3, 1, 1)

        layout.addWidget(self.button_abort, 3, 5, 1, 1)

        layout.addWidget(self.utils_text, self.shift + 3, 0, 1, 1)
        layout.addWidget(self.button_help, self.shift + 4, 0, 1, 1)
        layout.addWidget(self.button_mklsts, self.shift + 5, 0, 1, 1)
        layout.addWidget(self.button_mkps, self.shift + 6, 0, 1, 1)
        layout.addWidget(self.button_gremlin_ssm, self.shift + 7, 0, 1, 1)
        layout.addWidget(self.button_llm_ssm, self.shift + 8, 0, 1, 1)


        layout.addWidget(self.mklsts_cv_options_text, self.shift + 1, 1, 1, 1)
        layout.addWidget(self.box_mklsts_cv, self.shift + 2, 1, 1, 1)
        layout.addWidget(self.dca_text, self.shift + 3, 1, 1, 1)
        layout.addWidget(self.button_dca_inference_gremlin, self.shift + 4, 1, 1, 1)
        layout.addWidget(self.button_dca_test_dca, self.shift + 5, 1, 1, 1)
        layout.addWidget(self.button_llm_test_zs, self.shift + 6, 1, 1, 1)
        layout.addWidget(self.button_dca_predict_dca, self.shift + 7, 1, 1, 1)
        layout.addWidget(self.button_llm_predict_zs, self.shift + 8, 1, 1, 1)

        layout.addWidget(self.hybrid_text, self.shift + 3, 2, 1, 1)
        layout.addWidget(self.button_hybrid_train_dca, self.shift + 4, 2, 1, 1)
        layout.addWidget(self.button_hybrid_train_test_dca, self.shift + 5, 2, 1, 1)
        layout.addWidget(self.button_hybrid_test_dca, self.shift + 6, 2, 1, 1)
        layout.addWidget(self.button_hybrid_predict_dca, self.shift + 7, 2, 1, 1)

        layout.addWidget(self.plm_text, self.shift + 1, 3, 1, 1)
        layout.addWidget(self.box_llm, self.shift + 2, 3, 1, 1)
        layout.addWidget(self.hybrid_dca_llm_text, self.shift + 3, 3, 1, 1)
        layout.addWidget(self.button_hybrid_train_dca_llm, self.shift + 4, 3, 1, 1)
        layout.addWidget(self.button_hybrid_train_test_dca_llm, self.shift + 5, 3, 1, 1)
        layout.addWidget(self.button_hybrid_test_dca_llm, self.shift + 6, 3, 1, 1)
        layout.addWidget(self.button_hybrid_predict_dca_llm, self.shift + 7, 3, 1, 1)

        layout.addWidget(self.regression_model_text, self.shift + 1, 4, 1, 1)
        layout.addWidget(self.box_regression_model, self.shift + 2, 4, 1, 1)
        layout.addWidget(self.supervised_text, self.shift + 3, 4, 1, 1)
        layout.addWidget(self.button_supervised_train_dca, self.shift + 4, 4, 1, 1)
        layout.addWidget(self.button_supervised_train_test_dca, self.shift + 5, 4, 1, 1)
        layout.addWidget(self.button_supervised_test_dca, self.shift + 6, 4, 1, 1)
        layout.addWidget(self.button_supervised_predict_dca, self.shift + 7, 4, 1, 1)

        layout.addWidget(self.button_supervised_train_onehot, self.shift + 4, 5, 1, 1)
        layout.addWidget(self.button_supervised_train_test_onehot, self.shift + 5, 5, 1, 1)
        layout.addWidget(self.button_supervised_test_onehot, self.shift + 6, 5, 1, 1)
        layout.addWidget(self.button_supervised_predict_onehot, self.shift + 7, 5, 1, 1)
        layout.addWidget(self.box_plm_options, self.shift + 9, 5, 1, 1)

        layout.setRowMinimumHeight(self.shift + 9, 60)  # 60 pixels of space after row

        layout.addWidget(self.epoch_progress_bar, self.shift + 10, 0, 1, 5)
        layout.addWidget(self.epoch_time_label, self.shift + 10, 5, 1, 1)
        layout.addWidget(self.batch_progress_bar, self.shift + 11, 0, 1, 5)
        layout.addWidget(self.batch_time_label, self.shift + 11, 5, 1, 1)
        # Keep control columns compact
        for col in range(6):
            layout.setColumnStretch(col, 1)

        layout.addWidget(self.textedit_out, self.shift + 12, 0, 1, 2)

        layout.addWidget(self.logTextBox.widget, self.shift + 12, 2, 1, 4)

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

        worker.sig_step.connect(self.on_train_progress_step)

        worker.sig_done.connect(self.on_worker_done)
        worker.sig_done.connect(thread.quit)
        worker.sig_done.connect(worker.deleteLater)
        worker.sig_done.connect(thread.deleteLater)
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
    
    @Slot(dict)
    def on_train_progress_step(self, progress):
        if self._train_start_time is None:
            self._train_start_time = time.time()
            self._last_epoch = 1
            self._last_epoch_time = self._train_start_time
            self.epoch_eta = "--:--"
            self.elapsed = 0
        
        now = time.time()
        self.elapsed = now - self._train_start_time

        self.epoch_progress_bar.setValue(
            int((progress['epoch'] / progress['epoch_total']) * 100)
        )
        
        self.batch_progress_bar.setValue(
            int((progress['batch'] / progress['batch_total']) * 100)
        )

        if now - self._last_eta_update < 0.3:
            return

        # Epoch ETA
        if self._last_epoch != progress['epoch']:
            self.epoch_eta = self.estimate_eta(
                self.elapsed,
                progress['epoch'],
                progress['epoch_total']
            )
            self._last_epoch = progress['epoch']
            self._last_epoch_time = time.time()

        # Batch ETA
        elapsed_since_last_epoch = now - self._last_epoch_time
        self.batch_eta = self.estimate_eta(
            elapsed_since_last_epoch,
            progress['batch'],
            progress['batch_total']
        )

        elapsed_str = self.format_time(self.elapsed)
        # Batch update is every update
        if not progress['epoch'] == progress['epoch_total']:
            delta_elapsed_str = self.format_time(elapsed_since_last_epoch)

        # Update format text (stable width!)
        self.epoch_time_label.setText(
            f"Epoch {progress['epoch']} / {progress['epoch_total']}  "
            f"({int((progress['epoch'] / progress['epoch_total']) * 100)}%) "
            f"| Elapsed: {elapsed_str} | ETA: {self.epoch_eta}"
        )

        self.batch_time_label.setText(
            f"Batch {progress['batch']} / {progress['batch_total']}  "
            f"({int((progress['batch'] / progress['batch_total']) * 100)}%) "
            f"| Elapsed: {delta_elapsed_str} | ETA: {self.batch_eta}"
        )

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
        # TODO: Add functionality for new Signal-connected training/processing
        # for aborting (implemented for training..)
        self.logTextBox.widget.appendPlainText(
            'Asking each worker to abort...'
        )
        for thread, worker in self.__threads:
            #thread.quit()
            #thread.wait()
            worker.abort()
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
            f"Current working directory: {getcwd()}"
        )
        self.working_directory_text.setText(getcwd())
        self.logTextBox.widget.appendPlainText(
            "Job: " + str(self.c) + " " + "=" * 104
        )
        self.toggle_buttons(False)
    
    def end_process(self):
        self.target_button.setEnabled(True)
        self.toggle_buttons(True)
        self.epoch_progress_bar.setValue(0)
        self.batch_progress_bar.setValue(0)
        self._train_start_time = None
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

    def format_time(self, seconds: float) -> str:
        seconds = max(0, int(seconds))
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
    
    def estimate_eta(self, elapsed, current, total):
        if current <= 0:
            return "--:--"
        rate = elapsed / current
        remaining = rate * (total - current)
        return self.format_time(remaining)

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
        self.llm = [None, 'esm', 'prosst', 'esm+prosst'][i]

    def _llm_hybrid_flags(self):
        """
        Build the trailing CLI flag string for supervised DCA+PLM hybrid
        training from the LoRA/GP option check boxes (--lora / --gauss_opt /
        --gauss_comb).
        """
        flags = ''
        if self.check_lora.isChecked():
            flags += ' --lora'
        if self.check_gauss_opt.isChecked():
            flags += ' --gauss_opt'
        if self.check_gauss_comb.isChecked():
            flags += ' --gauss_comb'
        return flags

    def _llm_needs_structure(self, training: bool):
        """
        Whether a WT FASTA and PDB structure file are required for the current
        PLM selection: always for ProSST (single or combined), and additionally
        for Gaussian-process optimization during training.
        """
        if not self.llm:
            return False
        if 'prosst' in self.llm:
            return True
        if training and self.check_gauss_opt.isChecked():
            return True
        return False

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
        self.working_directory_text.setText(getcwd())

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
                        f'predict_ssm --plm {self.llm} '
                        f'--wt {wt_fasta_file} --pdb {pdb_file}'
                        )
                    self.start_main_thread()
                else:
                    self.end_process()
            elif self.llm == 'esm':
                self.cmd = f'predict_ssm --plm {self.llm} --wt {wt_fasta_file}'
                self.start_main_thread()
            else:
                self.logTextBox.widget.appendPlainText(
                    "Provide a PLM option for modeling."
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
            wt_fasta_file = QFileDialog.getOpenFileName(
                self.win2, "Select WT FASTA File",
                filter="FASTA file (*.fasta *.fa)"
            )[0]
            if self.llm == 'prosst':
                pdb_file = QFileDialog.getOpenFileName(
                    self.win2, "Select PDB protein structure File",
                    filter="PDB file (*.pdb)"
                )[0]
                if wt_fasta_file and pdb_file:
                    self.version_text.setText(
                        "ProSST zero shot model inference..."
                    )
                    self.cmd = (
                        f'hybrid --ts {test_set_file} --plm {self.llm} '
                        f'--wt {wt_fasta_file} --pdb {pdb_file}'
                        )
                    self.start_main_thread()
                else:
                    self.end_process()
            elif self.llm == 'esm':
                self.cmd = f'hybrid --ts {test_set_file} --plm {self.llm}  --wt {wt_fasta_file}'
                self.start_main_thread()
            else:
                self.logTextBox.widget.appendPlainText(
                    "Provide a PLM option for modeling. Combined PLM option "
                    "not implemented for zero-shot scoring."
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
            wt_fasta_file = QFileDialog.getOpenFileName(
                self.win2, "Select WT FASTA File",
                filter="FASTA file (*.fasta *.fa)"
            )[0]
            if self.llm == 'prosst':
                pdb_file = QFileDialog.getOpenFileName(
                    self.win2, "Select PDB protein structure File",
                    filter="PDB file (*.pdb)"
                )[0]
                if wt_fasta_file and pdb_file:
                    self.version_text.setText(
                        "ProSST zero shot model inference..."
                    )
                    self.cmd = (
                        f'hybrid --ps {prediction_file} --plm {self.llm} '
                        f'--wt {wt_fasta_file} --pdb {pdb_file}'
                        )
                    self.start_main_thread()
                else:
                    self.end_process()
            elif self.llm == 'esm':
                self.cmd = f'hybrid --ps {prediction_file} --plm {self.llm} --wt {wt_fasta_file}'
                self.start_main_thread()
            else:
                self.logTextBox.widget.appendPlainText(
                    "Provide a PLM option for modeling. Combined PLM option "
                    "not implemented for zero-shot scoring."
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
        if not self.llm:
            self.logTextBox.widget.appendPlainText("Provide a PLM option for modeling.")
            self.end_process()
            return
        training_file = QFileDialog.getOpenFileName(
            self.win2, "Select Training Set File in \"FASL\" format",
            filter="FASL file (*.fasl)"
        )[0]
        params_pkl_file = QFileDialog.getOpenFileName(
            self.win2, "Select DCA parameter Pickle file",
            filter="Pickle file (*.params GREMLIN PLMC)"
        )[0]
        flags = self._llm_hybrid_flags()
        if self._llm_needs_structure(training=True):
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
                    "Hybrid (DCA+PLM-supervised) model training..."
                )
                self.cmd = (
                    f'hybrid --ls {training_file} --ts {training_file} '
                    f'--params {params_pkl_file} --plm {self.llm} '
                    f'--wt {wt_fasta_file} --pdb {pdb_file}{flags}'
                )
                self.start_main_thread()
            else:
                self.end_process()
        else:
            if training_file and params_pkl_file:
                self.version_text.setText(
                    "Hybrid (DCA+PLM-supervised) model training..."
                )
                self.cmd = (
                    f'hybrid --ls {training_file} --ts {training_file} '
                    f'--params {params_pkl_file} --plm {self.llm}{flags}'
                )
                self.start_main_thread()
            else:
                self.end_process()

    def pypef_dca_llm_hybrid_train_test(self):
        self.target_button = self.button_hybrid_train_test_dca_llm
        self.start_process()
        if not self.llm:
            self.logTextBox.widget.appendPlainText("Provide a PLM option for modeling.")
            self.end_process()
            return
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
        flags = self._llm_hybrid_flags()
        if self._llm_needs_structure(training=True):
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
                    "Hybrid (DCA+PLM-supervised) model training..."
                )
                self.cmd = (
                    f'hybrid --ls {training_file} --ts {test_file} '
                    f'--params {params_pkl_file} --plm {self.llm} '
                    f'--wt {wt_fasta_file} --pdb {pdb_file}{flags}'
                )
                self.start_main_thread()
            else:
                self.end_process()
        else:
            if training_file and test_file and params_pkl_file:
                self.version_text.setText(
                    "Hybrid (DCA+PLM-supervised) model training..."
                )
                self.cmd = (
                    f'hybrid --ls {training_file} --ts {test_file} '
                    f'--params {params_pkl_file} --plm {self.llm}{flags}'
                )
                self.start_main_thread()
            else:
                self.end_process()

    def pypef_dca_llm_hybrid_test(self):
        self.target_button = self.button_hybrid_test_dca_llm
        self.start_process()
        if not self.llm:
            self.logTextBox.widget.appendPlainText("Provide a PLM option for modeling.")
            self.end_process()
            return
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
        if self._llm_needs_structure(training=False):
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
                    "Hybrid (DCA+PLM-supervised) model testing..."
                )
                self.cmd = (
                    f'hybrid -m {model_file} --ts {test_file} '
                    f'--params {params_pkl_file} --plm {self.llm} '
                    f'--wt {wt_fasta_file} --pdb {pdb_file}')
                self.start_main_thread()
            else:
                self.end_process()
        else:
            if test_file and params_pkl_file and model_file:
                self.version_text.setText(
                    "Hybrid (DCA+PLM-supervised) model testing..."
                )
                self.cmd = (
                    f'hybrid -m {model_file} --ts {test_file} '
                    f'--params {params_pkl_file} --plm {self.llm}')
                self.start_main_thread()
            else:
                self.end_process()

    def pypef_dca_llm_hybrid_predict(self):
        self.target_button = self.button_hybrid_predict_dca_llm
        self.start_process()
        if not self.llm:
            self.logTextBox.widget.appendPlainText("Provide a PLM option for modeling.")
            self.end_process()
            return
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
        if self._llm_needs_structure(training=False):
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
                    "Hybrid (DCA+PLM-supervised) model prediction..."
                )
                self.cmd = (
                    f'hybrid -m {model_file} --ps {prediction_file} '
                    f'--params {params_pkl_file} --plm {self.llm} '
                    f'--wt {wt_fasta_file} --pdb {pdb_file}'
                )
                self.start_main_thread()
            else:
                self.end_process()
        else:
            if prediction_file and params_pkl_file and model_file:
                self.version_text.setText(
                    "Hybrid (DCA+PLM-supervised) model prediction..."
                )
                self.cmd = (
                    f'hybrid -m {model_file} --ps {prediction_file} '
                    f'--params {params_pkl_file} --plm {self.llm}'
                )
                self.start_main_thread()
            else:
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
    app.setApplicationName("PyPEF")
    app.setApplicationDisplayName("PyPEF GUI")
    # On Wayland the taskbar/dock icon is NOT taken from setWindowIcon(); the
    # compositor resolves it by matching this desktop-file name (the window
    # app_id) against an installed pypef.desktop with an Icon= entry. Harmless
    # on X11/Windows/macOS, where setWindowIcon() below does the job directly.
    app.setDesktopFileName("pypef")
    logo_path = get_logo_path()
    if logo_path is not None:
        # App-level icon drives the taskbar / dock / Alt-Tab icon on
        # X11 / Windows / macOS (ignored by Wayland compositors).
        app.setWindowIcon(QIcon(logo_path))
    widget = MainWidget()
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()

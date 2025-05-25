# Niklas Siedhoff
# PyPEF - Pythonic Protein Engineering Framework

# GUI created with PyQT/PySide6

import sys
from io import StringIO 
import os
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QSize, Signal, QThread
pypef_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(pypef_root)
from pypef import __version__
from pypef.main import __doc__, run_main, logger, formatter
from pypef.utils.helpers import get_device, get_vram, get_torch_version, get_gpu_info

import logging
logger.setLevel(logging.INFO)


EXEC_API_OR_CLI = ['cli', 'api'][0]


print(sys.executable)
print('Backend:', EXEC_API_OR_CLI)


class ApiWorker(QThread):
    finished = Signal(str)

    def __init__(self, cmd: str, parent=None):
        super().__init__(parent)
        self.cmd = cmd

    def run(self):
        try:
            result = run_main(self.cmd)
        except Exception as e:
            result = f"Error: {str(e)}"
        self.finished.emit(result)


class Capturing(list):
    """https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call"""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


class QTextEditLogger(logging.Handler):
    """https://stackoverflow.com/questions/28655198/best-way-to-display-logs-in-pyqt"""
    def __init__(self, parent):
        super().__init__()
        self.widget = QtWidgets.QPlainTextEdit(parent)
        self.widget.setReadOnly(True)
        self.widget.setStyleSheet(
            "font-family:Consolas;font-size:12px;font-weight:normal;color:white;"
            "background-color:rgb(54, 69, 79);border:2px solid rgb(52, 59, 72);"
        )

    def emit(self, record):
        msg = self.format(record)
        self.widget.appendPlainText(msg)


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


class SecondWindow(QtWidgets.QWidget):
   def __init__(self):
      super().__init__()
      layout = QtWidgets.QVBoxLayout()
      self.setLayout(layout)


class MainWindow(QtWidgets.QWidget):
    def __init__(
            self, 
            pypef_root: str | None = None
    ):
        super().__init__()
        if pypef_root is None:
            self.pypef_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        else:
            self.pypef_root = pypef_root
        self.n_cores = 1 
        self.llm = 'esm'
        self.regression_model = 'PLS'
        self.c = 0
        self.ls_proportion = 0.8
        self.setMinimumSize(QSize(1400, 400))
        self.setWindowTitle("PyPEF GUI")
        self.setStyleSheet("background-color: rgb(40, 44, 52);")
        # https://doc.qt.io/qt-5/qcoreapplication.html#processEvents
        QtCore.QCoreApplication.processEvents()
        self.win2 = SecondWindow()

        # Texts #############################################################################
        layout = QtWidgets.QGridLayout(self)  # MAIN LAYOUT: QGridLayout
        self.version_text = QtWidgets.QLabel(f"PyPEF v. {__version__}", alignment=QtCore.Qt.AlignRight)
        self.ncores_text = QtWidgets.QLabel("Single-/multiprocessing")
        self.llm_text = QtWidgets.QLabel("LLM")
        self.regression_model_text =  QtWidgets.QLabel("Regression model")
        self.utils_text = QtWidgets.QLabel("Utilities")
        self.dca_text = QtWidgets.QLabel("DCA (unsupervised)")
        self.hybrid_text = QtWidgets.QLabel("Hybrid (supervised DCA)")
        self.hybrid_dca_llm_text = QtWidgets.QLabel("Hybrid (supervised DCA+LLM)")
        self.supervised_text = QtWidgets.QLabel("Purely supervised")
        self.slider_text = QtWidgets.QLabel("Train set proportion: 0.8")

        for txt in [
            self.version_text, self.ncores_text, self.regression_model_text, 
            self.utils_text, self.dca_text, self.hybrid_text, self.supervised_text,
            self.hybrid_text, self.hybrid_dca_llm_text
        ]:
            txt.setStyleSheet(text_style)

        self.device_text_out = QtWidgets.QTextEdit(readOnly=True)
        self.device_text_out.setStyleSheet(
            "font-family:Consolas;font-size:12px;font-weight:normal;color:white;"
            "background-color:rgb(54, 69, 79);border:2px solid rgb(52, 59, 72);"
        )
        self.device_text_out.setFixedHeight(70)
        self.device_text_out.append(f"Device (for LLM/DCA): {get_device().upper()}")
        self.device_text_out.append(get_vram())
        self.device_text_out.append(get_gpu_info())
        self.device_text_out.append(f"PyTorch version: {get_torch_version()}")

        self.textedit_out = QtWidgets.QTextEdit(readOnly=True)
        self.textedit_out.setStyleSheet(
            "font-family:Consolas;font-size:12px;font-weight:normal;color:white;"
            "background-color:rgb(54, 69, 79);border:2px solid rgb(52, 59, 72);"
        )
        self.logTextBox = QTextEditLogger(self)
        self.logTextBox.setFormatter(formatter)
        logger.addHandler(self.logTextBox)

        self.logTextBox.widget.appendPlainText(
            f"Current working directory: {str(os.getcwd())}")

        # Horizontal slider #################################################################
        self.slider = QtWidgets.QSlider(self)
        self.slider.setGeometry(QtCore.QRect(190, 100, 200, 16))
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(80)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.slider.move(10, 105)
        self.slider.valueChanged.connect(self.selection_ls_proportion)

        # Boxes #############################################################################
        self.box_multicore = QtWidgets.QComboBox()
        self.box_multicore.addItems(['Single core', 'Multi core'])
        self.box_multicore.currentIndexChanged.connect(self.selection_ncores)
        self.box_multicore.setStyleSheet("color:white;background-color:rgb(54, 69, 79);")

        self.box_regression_model = QtWidgets.QComboBox()
        self.regression_models = ['PLS', 'PLS_LOOCV', 'Ridge', 'Lasso', 'ElasticNet', 'SVR', 'RF', 'MLP']
        self.box_regression_model.addItems(self.regression_models)
        self.box_regression_model.currentIndexChanged.connect(self.selection_regression_model)
        self.box_regression_model.setStyleSheet("color:white;background-color:rgb(54, 69, 79);")

        self.box_llm = QtWidgets.QComboBox()
        self.box_llm.addItems(['ESM1v', 'ProSST'])
        self.box_llm.currentIndexChanged.connect(self.selection_llm_model)
        self.box_llm.setStyleSheet("color:white;background-color:rgb(54, 69, 79);")
        # Buttons ###########################################################################
        # Utilities

        self.button_work_dir = QtWidgets.QPushButton("Set Working Directory")
        self.button_work_dir.setToolTip("Set working directory for storing output files")
        self.button_work_dir.clicked.connect(self.set_work_dir)
        self.button_work_dir.setStyleSheet(button_style)        

        self.button_help = QtWidgets.QPushButton("Help")  
        self.button_help.setToolTip("Print help text")
        self.button_help.clicked.connect(self.pypef_help)
        self.button_help.setStyleSheet(button_style)

        self.button_mklsts = QtWidgets.QPushButton("Create LS and TS (MKLSTS)")       
        self.button_mklsts.setToolTip("Create \"FASL\" files for training and testing from variant-fitness CSV data")
        self.button_mklsts.clicked.connect(self.pypef_mklsts)
        self.button_mklsts.setStyleSheet(button_style)

        self.button_mkps = QtWidgets.QPushButton("Create PS (MKPS)")       
        self.button_mkps.setToolTip("Create FASTA files for prediction from variant-fitness CSV data")
        self.button_mkps.clicked.connect(self.pypef_mkps)
        self.button_mkps.setStyleSheet(button_style)
        
        # DCA
        self.button_dca_inference_gremlin = QtWidgets.QPushButton("MSA optimization (GREMLIN)")
        self.button_dca_inference_gremlin.setMinimumWidth(80)
        self.button_dca_inference_gremlin.setToolTip(
            "Generating DCA parameters using GREMLIN (\"MSA optimization\"); "
            "requires an MSA in FASTA or A2M format"
        )
        self.button_dca_inference_gremlin.clicked.connect(self.pypef_gremlin)
        self.button_dca_inference_gremlin.setStyleSheet(button_style)

        self.button_dca_inference_gremlin_msa_info = QtWidgets.QPushButton("GREMLIN SSM prediction")
        self.button_dca_inference_gremlin_msa_info.setMinimumWidth(80)
        self.button_dca_inference_gremlin_msa_info.setToolTip(
            "Generating DCA parameters using GREMLIN (\"MSA optimization\") and save plots of "
            "visualized results; requires an MSA in FASTA or A2M format"
        )
        self.button_dca_inference_gremlin_msa_info.clicked.connect(self.pypef_gremlin_msa_info)
        self.button_dca_inference_gremlin_msa_info.setStyleSheet(button_style)

        self.button_dca_test_dca = QtWidgets.QPushButton("Test (DCA)")
        self.button_dca_test_dca.setMinimumWidth(80)
        self.button_dca_test_dca.setToolTip(
            "Test performance on any test dataset using the MSA-optimized GREMLIN model"
        )
        self.button_dca_test_dca.clicked.connect(self.pypef_dca_test)
        self.button_dca_test_dca.setStyleSheet(button_style)

        self.button_dca_predict_dca = QtWidgets.QPushButton("Predict (DCA)")
        self.button_dca_predict_dca.setMinimumWidth(80)
        self.button_dca_predict_dca.setToolTip(
            "Predict any dataset using the MSA-optimized GREMLIN model"
        )
        self.button_dca_predict_dca.clicked.connect(self.pypef_dca_predict)
        self.button_dca_predict_dca.setStyleSheet(button_style)

        # Hybrid DCA
        self.button_hybrid_train_dca = QtWidgets.QPushButton("Train (DCA)")
        self.button_hybrid_train_dca.setMinimumWidth(80)
        self.button_hybrid_train_dca.setToolTip(
            "Optimize the GREMLIN model by supervised training on variant-fitness labels"
        )
        self.button_hybrid_train_dca.clicked.connect(self.pypef_dca_hybrid_train)
        self.button_hybrid_train_dca.setStyleSheet(button_style)

        self.button_hybrid_train_test_dca = QtWidgets.QPushButton("Train-Test (DCA)")
        self.button_hybrid_train_test_dca.setMinimumWidth(80)
        self.button_hybrid_train_test_dca.setToolTip(
            "Optimize the GREMLIN model by supervised training on variant-fitness labels and "
            "testing the model on a test set"
        )
        self.button_hybrid_train_test_dca.clicked.connect(self.pypef_dca_hybrid_train_test)
        self.button_hybrid_train_test_dca.setStyleSheet(button_style)

        self.button_hybrid_test_dca = QtWidgets.QPushButton("Test (DCA)")
        self.button_hybrid_test_dca.setMinimumWidth(80)
        self.button_hybrid_test_dca.setToolTip(
            "Test the trained hybrid DCA model on a test set"
        )
        self.button_hybrid_test_dca.clicked.connect(self.pypef_dca_hybrid_test)
        self.button_hybrid_test_dca.setStyleSheet(button_style)

        # Hybrid DCA prediction
        self.button_hybrid_predict_dca = QtWidgets.QPushButton("Predict (DCA)")
        self.button_hybrid_predict_dca.setMinimumWidth(80)
        self.button_hybrid_predict_dca.setToolTip(
            "Predict FASTA dataset using the hybrid DCA model"
        )
        self.button_hybrid_predict_dca.clicked.connect(self.pypef_dca_hybrid_predict)
        self.button_hybrid_predict_dca.setStyleSheet(button_style)

        # Hybrid DCA+LLM ##################################################################### TODO
        self.button_hybrid_train_dca_llm = QtWidgets.QPushButton("Train (DCA+LLM)")
        self.button_hybrid_train_dca_llm.setMinimumWidth(80)
        self.button_hybrid_train_dca_llm.setToolTip(
            "Optimize the GREMLIN model and tune the LLM by supervised training on variant-fitness "
            "labels"
        )
        self.button_hybrid_train_dca_llm.clicked.connect(self.pypef_dca_llm_hybrid_train)
        self.button_hybrid_train_dca_llm.setStyleSheet(button_style)

        self.button_hybrid_train_test_dca_llm = QtWidgets.QPushButton("Train-Test (DCA+LLM)")
        self.button_hybrid_train_test_dca_llm.setMinimumWidth(80)
        self.button_hybrid_train_test_dca_llm.setToolTip(
            "Optimize the GREMLIN model and tune the LLM by supervised training on variant-fitness "
            "labels and testing the model on a test set"
        )
        self.button_hybrid_train_test_dca_llm.clicked.connect(self.pypef_dca_llm_hybrid_train_test)
        self.button_hybrid_train_test_dca_llm.setStyleSheet(button_style)

        self.button_hybrid_test_dca_llm = QtWidgets.QPushButton("Test (DCA+LLM)")
        self.button_hybrid_test_dca_llm.setMinimumWidth(80)
        self.button_hybrid_test_dca_llm.setToolTip(
            "Test the trained hybrid DCA+LLM model on a test set"
        )
        self.button_hybrid_test_dca_llm.clicked.connect(self.pypef_dca_llm_hybrid_test)
        self.button_hybrid_test_dca_llm.setStyleSheet(button_style)

        self.button_hybrid_predict_dca_llm = QtWidgets.QPushButton("Predict (DCA+LLM)")
        self.button_hybrid_predict_dca_llm.setMinimumWidth(80)
        self.button_hybrid_predict_dca_llm.setToolTip(
            "Use the trained hybrid DCA+LLM model for prediction"
        )
        self.button_hybrid_predict_dca_llm.clicked.connect(self.pypef_dca_llm_hybrid_predict)
        self.button_hybrid_predict_dca_llm.setStyleSheet(button_style)
        ###################################################################################### TODO END

        # Pure Supervised
        self.button_supervised_train_dca = QtWidgets.QPushButton("Train (DCA encoding)")
        self.button_supervised_train_dca.setMinimumWidth(80)
        self.button_supervised_train_dca.setToolTip(
            "Purely supervised DCA (GREMLIN or PLMC) model training on variant-fitness labels"
        )
        self.button_supervised_train_dca.clicked.connect(self.pypef_dca_supervised_train)
        self.button_supervised_train_dca.setStyleSheet(button_style)

        self.button_supervised_train_test_dca = QtWidgets.QPushButton("Train-Test (DCA encoding)")
        self.button_supervised_train_test_dca.setMinimumWidth(80)
        self.button_supervised_train_test_dca.setToolTip(
            "Purely supervised DCA (GREMLIN or PLMC) model training and testing on variant-fitness labels"
        )
        self.button_supervised_train_test_dca.clicked.connect(self.pypef_dca_supervised_train_test)
        self.button_supervised_train_test_dca.setStyleSheet(button_style)

        self.button_supervised_test_dca = QtWidgets.QPushButton("Test (DCA encoding)")
        self.button_supervised_test_dca.setMinimumWidth(80)
        self.button_supervised_test_dca.setToolTip(
            "Purely supervised DCA (GREMLIN or PLMC) model test"
        )
        self.button_supervised_test_dca.clicked.connect(self.pypef_dca_supervised_test)
        self.button_supervised_test_dca.setStyleSheet(button_style)

        self.button_supervised_predict_dca = QtWidgets.QPushButton("Predict (DCA encoding)")
        self.button_supervised_predict_dca.setMinimumWidth(80)
        self.button_supervised_predict_dca.setToolTip(
            "Purely supervised DCA (GREMLIN or PLMC) model prediction"
        )
        self.button_supervised_predict_dca.clicked.connect(self.pypef_dca_supervised_predict)
        self.button_supervised_predict_dca.setStyleSheet(button_style)


        self.button_supervised_train_onehot = QtWidgets.QPushButton("Train (One-hot encoding)")
        self.button_supervised_train_onehot.setMinimumWidth(80)
        self.button_supervised_train_onehot.setToolTip(
            "Purely supervised one-hot model training on variant-fitness labels"
        )
        self.button_supervised_train_onehot.clicked.connect(self.pypef_onehot_supervised_train)
        self.button_supervised_train_onehot.setStyleSheet(button_style)

        self.button_supervised_train_test_onehot = QtWidgets.QPushButton("Train-Test (One-hot encoding)")
        self.button_supervised_train_test_onehot.setMinimumWidth(80)
        self.button_supervised_train_test_onehot.setToolTip(
            "Purely supervised one-hot model training on variant-fitness labels"
        )
        self.button_supervised_train_test_onehot.clicked.connect(self.pypef_onehot_supervised_train_test)
        self.button_supervised_train_test_onehot.setStyleSheet(button_style)

        self.button_supervised_test_onehot = QtWidgets.QPushButton("Test (One-hot encoding)")
        self.button_supervised_test_onehot.setMinimumWidth(80)
        self.button_supervised_test_onehot.setToolTip(
            "Purely supervised one-hot model test"
        )
        self.button_supervised_test_onehot.clicked.connect(self.pypef_onehot_supervised_test)
        self.button_supervised_test_onehot.setStyleSheet(button_style)

        self.button_supervised_predict_onehot = QtWidgets.QPushButton("Predict (One-hot encoding)")
        self.button_supervised_predict_onehot.setMinimumWidth(80)
        self.button_supervised_predict_onehot.setToolTip(
            "Purely supervised one-hot model test"
        )
        self.button_supervised_predict_onehot.clicked.connect(self.pypef_onehot_supervised_predict)
        self.button_supervised_predict_onehot.setStyleSheet(button_style)
        # All buttons #######
        self.all_buttons = [
            self.button_work_dir,
            self.button_help,
            self.button_mklsts,
            self.button_mkps,
            self.button_dca_inference_gremlin,
            self.button_dca_inference_gremlin_msa_info,
            self.button_dca_test_dca,
            self.button_dca_predict_dca,
            self.button_hybrid_train_dca,
            self.button_hybrid_train_test_dca,
            self.button_hybrid_test_dca,
            self.button_hybrid_predict_dca,
            self.button_hybrid_train_dca_llm,
            self.button_hybrid_train_test_dca_llm,
            self.button_hybrid_test_dca_llm,
            self.button_hybrid_predict_dca_llm,
            self.button_supervised_train_dca,
            self.button_supervised_train_test_dca,
            self.button_supervised_test_dca,
            self.button_supervised_predict_dca,
            self.button_supervised_train_onehot,
            self.button_supervised_train_test_onehot,
            self.button_supervised_test_onehot,
            self.button_supervised_predict_onehot
        ]
        ######################

        # Layout widgets ####################################################################
        # int fromRow, int fromColumn, int rowSpan, int columnSpan

        layout.addWidget(self.device_text_out, 0, 0, 1, 2)
        layout.addWidget(self.version_text, 0, 5, 1, 1)

        layout.addWidget(self.slider_text, 1, 0, 1, 1)
        #layout.addWidget(self.slider, 2, 0, 1, 1)

        layout.addWidget(self.button_work_dir, 0, 2, 1, 1)

        layout.addWidget(self.ncores_text, 1, 1, 1, 1)
        layout.addWidget(self.box_multicore, 2, 1, 1, 1)

        layout.addWidget(self.utils_text, 3, 0, 1, 1)
        layout.addWidget(self.button_help, 4, 0, 1, 1)
        layout.addWidget(self.button_mklsts, 5, 0, 1, 1)
        layout.addWidget(self.button_mkps, 6, 0, 1, 1)

        layout.addWidget(self.dca_text, 3, 1, 1, 1)
        layout.addWidget(self.button_dca_inference_gremlin, 4, 1, 1, 1)
        layout.addWidget(self.button_dca_inference_gremlin_msa_info, 5, 1, 1, 1)
        layout.addWidget(self.button_dca_test_dca, 6, 1, 1, 1)
        layout.addWidget(self.button_dca_predict_dca, 7, 1, 1, 1)

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

        if EXEC_API_OR_CLI == 'cli':
            self.process = QtCore.QProcess(self)
            self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
            self.process.readyReadStandardOutput.connect(self.on_readyReadStandardOutput)
            self.process.started.connect(lambda: self.toggle_buttons(False))
            self.process.finished.connect(lambda: self.toggle_buttons(True))



    def toggle_buttons(self, enabled: bool):
        for btn in self.all_buttons:
            btn.setEnabled(enabled)
        
    def start_process(self, button):
        self.logTextBox.widget.clear()
        self.c += 1
        self.logTextBox.widget.appendPlainText(f"Current working directory: {str(os.getcwd())}")
        self.logTextBox.widget.appendPlainText("Job: " + str(self.c) + " " + "=" * 104)
        if EXEC_API_OR_CLI == 'api':
            button.setEnabled(False)
            self.toggle_buttons(False)
    
    def end_process(self, button):
        if EXEC_API_OR_CLI == 'api':
            button.setEnabled(True)
            self.toggle_buttons(True)
        self.version_text.setText("Finished...")
        self.textedit_out.append("=" * 104 + " Job: " + str(self.c) + "\n")

    def on_readyReadStandardOutput(self):
         text = self.process.readAllStandardOutput().data().decode()
         #self.textedit_out.append(text.strip())
         self.logTextBox.widget.appendPlainText(text.strip())
    
    def selection_ncores(self, i):
        if i == 0:
            self.n_cores = 1
        elif i == 1:
            self.n_cores = os.cpu_count()

    def selection_regression_model(self, i):
        self.regression_model = [r.lower() for r in self.regression_models][i]

    def selection_llm_model(self, i):
        self.llm = ['esm', 'prosst'][i]

    def selection_ls_proportion(self, value):
        self.ls_proportion = value / 100
        self.slider_text.setText(f"Train set proportion: {self.ls_proportion}")

    @QtCore.Slot()
    def set_work_dir(self):
        self.working_directory = QtWidgets.QFileDialog.getExistingDirectory(
            self.win2, 'Select Folder')
        os.chdir(self.working_directory)
        self.logTextBox.widget.clear()
        self.logTextBox.widget.appendPlainText(
            f"Changed current working directory to: {str(os.getcwd())}"
        )

    @QtCore.Slot()
    def pypef_help(self):
        button = self.button_help
        self.start_process(button=button)
        self.textedit_out.append(f'Executing command:\n    --help')
        self.version_text.setText("Getting help...")
        self.logTextBox.widget.appendPlainText(__doc__)
        self.end_process(button=button)

    @QtCore.Slot()
    def pypef_mklsts(self):
        button = self.button_mklsts
        self.start_process(button=button)
        wt_fasta_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select WT FASTA File", 
                                                              filter="FASTA file (*.fasta *.fa)")[0]
        csv_variant_file = QtWidgets.QFileDialog.getOpenFileName(
            self.win2, "Select variant CSV File", filter="CSV file (*.csv)")[0]
        if wt_fasta_file and csv_variant_file:
            self.version_text.setText("Running MKLSTS...")
            self.exec_pypef(f'mklsts --wt {wt_fasta_file} --input {csv_variant_file} '
                            f'--ls_proportion {self.ls_proportion}')
        self.end_process(button=button)

    @QtCore.Slot()
    def pypef_mkps(self):
        button = self.button_mkps
        self.start_process(button=button)
        wt_fasta_file = QtWidgets.QFileDialog.getOpenFileName(
            self.win2, "Select WT FASTA File", filter="FASTA file (*.fasta *.fa)")[0]
        csv_variant_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select variant CSV File", 
                                                                 filter="CSV file (*.csv)")[0]
        if wt_fasta_file and csv_variant_file:
            self.version_text.setText("Running MKLSTS...")
            self.exec_pypef(f'mkps --wt {wt_fasta_file} --input {csv_variant_file}')
        self.end_process(button=button)

    @QtCore.Slot()
    def pypef_gremlin(self):
        button = self.button_dca_inference_gremlin
        self.start_process(button=button)
        wt_fasta_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select WT FASTA File", 
                                                              filter="FASTA file (*.fasta *.fa)")[0]
        msa_file = QtWidgets.QFileDialog.getOpenFileName(
            self.win2, "Select Multiple Sequence Alignment (MSA) file (in FASTA or A2M format)",
            filter="MSA file (*.fasta *.a2m)")[0]
        if wt_fasta_file and msa_file:
            self.version_text.setText("Running GREMLIN (DCA) optimization on MSA...")
            self.exec_pypef(f'param_inference --wt {wt_fasta_file} --msa {msa_file}')  # --opt_iter 100
        self.end_process(button=button)

    @QtCore.Slot()
    def pypef_gremlin_msa_info(self):
        button = self.button_dca_inference_gremlin_msa_info
        self.start_process(button=button)
        wt_fasta_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select WT FASTA File",
                                                              filter="FASTA file (*.fasta *.fa)")[0]
        msa_file = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Multiple Sequence Alignment (MSA) file (in FASTA or A2M format)",
            filter="MSA file (*.fasta *.a2m)")[0]
        if wt_fasta_file and msa_file:
            self.version_text.setText("Running GREMLIN (DCA) optimization on MSA...")
            self.exec_pypef(f'save_msa_info --wt {wt_fasta_file} --msa {msa_file}')
        self.end_process(button=button)

    @QtCore.Slot()
    def pypef_dca_test(self):
        button = self.button_dca_test_dca
        self.start_process(button=button)
        test_set_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Test Set File in \"FASL\" format",
                                                                filter="FASL file (*.fasl)")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Parameter Pickle file",
                                                                filter="Pickle file (*.params GREMLIN PLMC)")[0]
        if test_set_file and params_pkl_file:
            self.version_text.setText("Testing DCA performance on provided test set...")
            self.exec_pypef(f'hybrid --ts {test_set_file} -m {params_pkl_file} --params {params_pkl_file}')
        self.end_process(button=button)

    @QtCore.Slot()
    def pypef_dca_predict(self):
        button = self.button_dca_predict_dca
        self.start_process(button=button)
        prediction_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Prediction Set File in FASTA format",
                                                                filter="FASTA file (*.fasta *.fa)")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select DCA parameter Pickle file",
                                                                filter="Pickle file (*.params GREMLIN PLMC)")[0]
        if prediction_file and params_pkl_file:
            self.version_text.setText("Predicting using the DCA model on provided prediction set...")
            self.exec_pypef(f'hybrid --ps {prediction_file} -m {params_pkl_file} --params {params_pkl_file}')
        self.end_process(button=button)

    @QtCore.Slot()
    def pypef_dca_hybrid_train(self):
        button = self.button_hybrid_train_dca
        self.start_process(button=button)
        training_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Training Set File in \"FASL\" format",
                                                                filter="FASL file (*.fasl)")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select DCA parameter Pickle file",
                                                                filter="Pickle file (*.params GREMLIN PLMC)")[0]
        if training_file and params_pkl_file:
            self.version_text.setText("Hybrid (DCA-supervised) model training...")
            self.exec_pypef(f'hybrid --ls {training_file} --ts {training_file} -m {params_pkl_file} --params {params_pkl_file}')
        self.end_process(button=button)

    @QtCore.Slot()
    def pypef_dca_hybrid_train_test(self):
        button = self.button_hybrid_train_test_dca
        self.start_process(button=button)
        training_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Training Set File in \"FASL\" format",
                                                              filter="FASL file (*.fasl)")[0]
        test_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Test Set File in \"FASL\" format",
                                                          filter="FASL file (*.fasl)")[0]
        #model_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select DCA model Pickle file")[0]  # TODO: Check if needed
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select DCA parameter Pickle file",
                                                                filter="Pickle file (*.params GREMLIN PLMC)")[0]
        if training_file and test_file and params_pkl_file:
            self.version_text.setText("Hybrid (DCA-supervised) model training and testing...")
            self.exec_pypef(f'hybrid -m {params_pkl_file} --ls {training_file} --ts {test_file} --params {params_pkl_file}')
        self.end_process(button=button)

    @QtCore.Slot()
    def pypef_dca_hybrid_test(self):
        button = self.button_hybrid_test_dca
        self.start_process(button=button)        
        test_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Test Set File in \"FASL\" format",
                                                          filter="FASL file (*.fasl)")[0]
        model_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Hybrid Model file in Pickle format",
                                                               filter="Pickle file (HYBRID*)")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select DCA parameter Pickle file",
                                                                filter="Pickle file (*.params GREMLIN PLMC)")[0]
        if test_file and params_pkl_file:
            self.version_text.setText("Hybrid (DCA-supervised) model testing...")
            self.exec_pypef(f'hybrid -m {model_pkl_file} --ts {test_file} --params {params_pkl_file}')
        self.end_process(button=button)    

    @QtCore.Slot()
    def pypef_dca_hybrid_predict(self):
        button = self.button_hybrid_predict_dca
        self.start_process(button=button)    
        prediction_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Prediction Set File in FASTA format",
                                                                filter="FASTA file (*.fasta *.fa)")[0]
        model_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Hybrid Model file in Pickle format",
                                                           filter="Pickle file (HYBRID*)")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select DCA parameter Pickle file",
                                                                filter="Pickle file (*.params GREMLIN PLMC)")[0]
        if prediction_file and params_pkl_file:
            self.version_text.setText("Predicting using the hybrid (DCA-supervised) model...")
            self.exec_pypef(f'hybrid -m {model_file} --ps {prediction_file} --params {params_pkl_file}')
        self.end_process(button=button)  

    @QtCore.Slot()
    def pypef_dca_llm_hybrid_train(self):
        button = self.button_hybrid_train_dca_llm
        self.start_process(button=button)  
        training_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Training Set File in \"FASL\" format",
                                                              filter="FASL file (*.fasl)")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select DCA parameter Pickle file",
                                                                filter="Pickle file (*.params GREMLIN PLMC)")[0]
        if self.llm == 'prosst':
            wt_fasta_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select WT FASTA File",
                                                                  filter="FASTA file (*.fasta *.fa)")[0]
            pdb_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select PDB protein structure File",
                                                             filter="PDB file (*.pdb)")[0]
            if training_file and params_pkl_file and wt_fasta_file and pdb_file:
                self.version_text.setText("Hybrid (DCA+LLM-supervised) model training...")
                self.exec_pypef(f'hybrid --ls {training_file} --ts {training_file} --params {params_pkl_file} --llm {self.llm} '
                                f'--wt {wt_fasta_file} --pdb {pdb_file}')
        else:
            if training_file and params_pkl_file:
                self.version_text.setText("Hybrid (DCA+LLM-supervised) model training...")
                self.exec_pypef(f'hybrid --ls {training_file} --ts {training_file} --params {params_pkl_file} --llm {self.llm}')
        self.end_process(button=button)

    @QtCore.Slot()
    def pypef_dca_llm_hybrid_train_test(self):
        button = self.button_hybrid_train_test_dca_llm
        self.start_process(button=button)  
        training_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Training Set File in \"FASL\" format",
                                                              filter="FASL file (*.fasl)")[0]
        test_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Test Set File in \"FASL\" format",
                                                          filter="FASL file (*.fasl)")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select DCA parameter Pickle file",
                                                                filter="Pickle file (*.params GREMLIN PLMC)")[0]
        if self.llm == 'prosst':
            wt_fasta_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select WT FASTA File",
                                                                  filter="FASTA file (*.fasta *.fa)")[0]
            pdb_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select PDB protein structure File",
                                                             filter="PDB file (*.pdb)")[0]
            if training_file and test_file and params_pkl_file and wt_fasta_file and pdb_file:
                self.version_text.setText("Hybrid (DCA+LLM-supervised) model training...")
                self.exec_pypef(f'hybrid --ls {training_file} --ts {test_file} --params {params_pkl_file} --llm {self.llm} '
                                f'--wt {wt_fasta_file} --pdb {pdb_file}')
        else:
            if training_file and test_file and params_pkl_file:
                self.version_text.setText("Hybrid (DCA+LLM-supervised) model training...")
                self.exec_pypef(f'hybrid --ls {training_file} --ts {test_file} --params {params_pkl_file} --llm {self.llm}')
        self.end_process(button=button)

    @QtCore.Slot()
    def pypef_dca_llm_hybrid_test(self):
        button = self.button_hybrid_test_dca_llm
        self.start_process(button=button)  
        test_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Test Set File in \"FASL\" format",
                                                          filter="FASL file (*.fasl)")[0]
        model_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Hybrid Model file in Pickle format",
                                                           filter="Pickle file (HYBRID*)")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select DCA parameter Pickle file",
                                                                filter="Pickle file (*.params GREMLIN PLMC)")[0]
        if self.llm == 'prosst':
            wt_fasta_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select WT FASTA File",
                                                                  filter="FASTA file (*.fasta *.fa)")[0]
            pdb_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select PDB protein structure File",
                                                             filter="PDB file (*.pdb)")[0]
            if test_file and params_pkl_file and wt_fasta_file and pdb_file and model_file:
                self.version_text.setText("Hybrid (DCA+LLM-supervised) model testing...")
                self.exec_pypef(f'hybrid -m {model_file} --ts {test_file} --params {params_pkl_file} --llm {self.llm} '
                                f'--wt {wt_fasta_file} --pdb {pdb_file}')
        else:
            if test_file and params_pkl_file and model_file:
                self.version_text.setText("Hybrid (DCA+LLM-supervised) model testing...")
                self.exec_pypef(f'hybrid -m {model_file} --ts {test_file} --params {params_pkl_file} --llm {self.llm}')
        self.end_process(button=button) 

    @QtCore.Slot()
    def pypef_dca_llm_hybrid_predict(self):
        button = self.button_hybrid_predict_dca_llm
        self.start_process(button=button)  
        prediction_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Prediction Set File in FASTA format",
                                                                filter="FASTA file (*.fasta *.fa)")[0]
        model_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Hybrid Model file in Pickle format",
                                                           filter="Pickle file (HYBRID*)")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select DCA parameter Pickle file",
                                                                filter="Pickle file (*.params GREMLIN PLMC)")[0]
        if self.llm == 'prosst':
            wt_fasta_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select WT FASTA File",
                                                                  filter="FASTA file (*.fasta *.fa)")[0]
            pdb_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select PDB protein structure File",
                                                             filter="PDB file (*.pdb)")[0]
            if prediction_file and params_pkl_file and wt_fasta_file and pdb_file and model_file:
                self.version_text.setText("Hybrid (DCA+LLM-supervised) model training...")
                self.exec_pypef(f'hybrid -m {model_file} --ps {prediction_file} --params {params_pkl_file} --llm {self.llm} '
                                f'--wt {wt_fasta_file} --pdb {pdb_file}')
        else:
            if prediction_file and params_pkl_file and model_file:
                self.version_text.setText("Hybrid (DCA+LLM-supervised) model training...")
                self.exec_pypef(f'hybrid -m {model_file} --ps {prediction_file} --params {params_pkl_file} --llm {self.llm}')
        self.end_process(button=button) 

    @QtCore.Slot()
    def pypef_dca_supervised_train(self):
        button = self.button_supervised_train_dca
        self.start_process(button=button)  
        training_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Training Set File in \"FASL\" format",
                                                              filter="FASL file (*.fasl)")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select DCA parameter Pickle file",
                                                                filter="Pickle file (*.params GREMLIN PLMC)")[0]
        if training_file and params_pkl_file:
            self.version_text.setText("Hybrid (DCA-supervised) model training and testing...")
            self.exec_pypef(f'ml --encoding dca --ls {training_file} --ts {training_file} --params {params_pkl_file} '
                            f'--threads {self.n_cores} --regressor {self.regression_model}')
        self.end_process(button=button)

    @QtCore.Slot()
    def pypef_dca_supervised_train_test(self):
        button = self.button_supervised_train_test_dca
        self.start_process(button=button)  
        training_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Training Set File in \"FASL\" format",
                                                              filter="FASL file (*.fasl)")[0]
        test_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Test Set File in \"FASL\" format",
                                                          filter="FASL file (*.fasl)")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select DCA parameter Pickle file",
                                                                filter="Pickle file (*.params GREMLIN PLMC)")[0]
        if training_file and test_file and params_pkl_file:
            self.version_text.setText("Hybrid (DCA-supervised) model training and testing...")
            self.exec_pypef(f'ml --encoding dca --ls {training_file} --ts {test_file} --params {params_pkl_file} '
                            f'--threads {self.n_cores} --regressor {self.regression_model}')
        self.end_process(button=button)

    @QtCore.Slot()
    def pypef_dca_supervised_test(self):
        button = self.button_supervised_test_dca
        self.start_process(button=button)  
        test_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Test Set File in \"FASL\" format",
                                                          filter="FASL file (*.fasl)")[0]
        model_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select ML Model file in Pickle format",
                                                           filter="Pickle file (ML*)")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select DCA parameter Pickle file",
                                                                filter="Pickle file (*.params GREMLIN PLMC)")[0]
        if test_file and params_pkl_file and model_file:
            self.version_text.setText("Hybrid (DCA-supervised) model testing...")
            self.exec_pypef(f'ml -m {model_file} --encoding dca --ts {test_file} --params {params_pkl_file} '
                            f'--threads {self.n_cores}')
        self.end_process(button=button)

    @QtCore.Slot()
    def pypef_dca_supervised_predict(self):
        button = self.button_supervised_predict_dca
        self.start_process(button=button)  
        prediction_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Prediction Set File in FASTA format",
                                                                filter="FASTA file (*.fasta *.fa)")[0]
        model_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select ML Model file in Pickle format",
                                                           filter="Pickle file (ML*)")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select DCA parameter Pickle file",
                                                                filter="Pickle file (*.params GREMLIN PLMC)")[0]
        if prediction_file and params_pkl_file and model_file:
            self.version_text.setText("Hybrid (DCA-supervised) model prediction...")
            self.exec_pypef(f'ml -m {model_file} --encoding dca --ps {prediction_file} --params {params_pkl_file} '
                            f'--threads {self.n_cores}')
        self.end_process(button=button)

    @QtCore.Slot()
    def pypef_onehot_supervised_train(self):
        button = self.button_supervised_train_onehot
        self.start_process(button=button)  
        training_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Training Set File in \"FASL\" format",
                                                              filter="FASL file (*.fasl)")[0]
        if training_file:
            self.version_text.setText("Hybrid (DCA-supervised) model training...")
            self.exec_pypef(f'ml --encoding onehot --ls {training_file} --ts {training_file} '
                            f'--threads {self.n_cores} --regressor {self.regression_model}')
        self.end_process(button=button)

    @QtCore.Slot()
    def pypef_onehot_supervised_train_test(self):
        button = self.button_supervised_train_test_onehot
        self.start_process(button=button)  
        training_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Training Set File in \"FASL\" format",
                                                          filter="FASL file (*.fasl)")[0]
        test_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Test Set File in \"FASL\" format",
                                                          filter="FASL file (*.fasl)")[0]
        if training_file and test_file:
            self.version_text.setText("Hybrid (DCA-supervised) model training and testing...")
            self.exec_pypef(f'ml --encoding onehot --ls {training_file} --ts {test_file} '
                            f'--threads {self.n_cores} --regressor {self.regression_model}')
        self.end_process(button=button)

    @QtCore.Slot()
    def pypef_onehot_supervised_test(self):
        button = self.button_supervised_train_test_onehot
        self.start_process(button=button)  
        model_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Onehot Model file in Pickle format",
                                                           filter="Pickle file (ONEHOT*)")[0]
        test_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Test Set File in \"FASL\" format",
                                                          filter="FASL file (*.fasl)")[0]
        if test_file and model_file:
            self.version_text.setText("Hybrid (DCA-supervised) model testing...")
            self.exec_pypef(f'ml -m {model_file} --encoding onehot --ts {test_file} '
                            f'--threads {self.n_cores}')
        self.end_process(button=button)

    @QtCore.Slot()
    def pypef_onehot_supervised_predict(self):
        button = self.button_supervised_predict_onehot
        self.start_process(button=button)  
        model_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Onehot Model file in Pickle format",
                                                           filter="Pickle file (ONEHOT*)")[0]
        prediction_file = QtWidgets.QFileDialog.getOpenFileName(self.win2, "Select Prediction Set File in FASTA format",
                                                                filter="FASTA file (*.fasta *.fa)")[0]
        if prediction_file and model_file:
            self.version_text.setText("Hybrid (DCA-supervised) model prediction...")
            self.exec_pypef(f'ml -m {model_file} --encoding onehot --ps {prediction_file} '
                            f'--threads {self.n_cores}')
        self.end_process(button=button)

    def exec_pypef(self, cmd):
        if EXEC_API_OR_CLI == 'api':
            self.exec_pypef_api2(cmd)
        elif EXEC_API_OR_CLI == 'cli':
            self.exec_pypef_cli(cmd)
        else:
            raise SystemError("Choose between 'api' or 'cli'!")

    def exec_pypef_cli(self, cmd: str):
        self.textedit_out.append(f'Executing command:\n    {cmd}')
        self.process.start(f'python', ['-u', f'{self.pypef_root}/run.py'] + cmd.split(' '))
        self.process.finished.connect(self.process_finished)

    def exec_pypef_api2(self, cmd: str):
        """
        Backup function if threading function (exec_pypef_api) does not work.
        Freezes during run.
        """
        self.textedit_out.append(f'Executing command:\n\t{cmd}')
        try:
            with Capturing() as captured_output:
                run_main(argv=cmd)
            for cap_out_text in captured_output:
                self.textedit_out.append(cap_out_text)
        except Exception as e: # anything
            self.textedit_out.append(f"Provided wrong inputs! Error:\n\t{e}")

    def exec_pypef_api(self, cmd: str):
        """
        Threaded API function.
        """
        self.textedit_out.append(f"Executing command:\n\t{cmd}")
        self.thread = QThread()
        self.worker = ApiWorker(cmd=cmd)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)        
        self.thread.start()

    def process_finished(self):
        self.version_text.setText("Finished...")


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = MainWindow()
    widget.resize(800, 600)
    widget.show()
    sys.exit(app.exec())

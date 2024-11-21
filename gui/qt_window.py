
# GUI created with PyQT/PySide
# PySide vs PyQT: https://www.pythonguis.com/faq/pyqt-vs-pyside/?gad_source=1&gclid=CjwKCAjwpbi4BhByEiwAMC8Jnfe7sYOqHjs5eOg_tYMD0iX3UDBduwykrF8qE5Y0IG66abhS6YXHvRoCg-kQAvD_BwE
# (If using PyQT, see: https://www.gnu.org/licenses/license-list.en.html#GPLCompatibleLicenses)

import re
import sys
import os
import subprocess
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QSize
# Up to now needs (pip) installed PyPEF version
# https://stackoverflow.com/questions/67297494/redirect-console-output-to-pyqt5-gui-in-real-time
# sudo apt-get install -y libxcb-cursor-dev
pypef_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def capture(command):
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out, err = proc.communicate()
    return out, err, proc.returncode


out, err, exitcode = capture([f'python', f'{pypef_root}/run.py', '--version'])
version = re.findall(r"[-+]?(?:\d*\.*\d.*\d+)", str(out))[0]

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
        self.regression_model = 'PLS'
        self.c = 0
        self.setMinimumSize(QSize(200, 100))
        self.setWindowTitle("PyPEF GUI")
        self.setStyleSheet("background-color: rgb(40, 44, 52);")

        # Texts #############################################################################
        layout = QtWidgets.QGridLayout(self)
        self.version_text = QtWidgets.QLabel(f"PyPEF v. {version}", alignment=QtCore.Qt.AlignRight)
        self.ncores_text = QtWidgets.QLabel("Single-/multiprocessing")
        self.regression_model_text =  QtWidgets.QLabel("Regression model")
        self.utils_text = QtWidgets.QLabel("Utilities")
        self.dca_text = QtWidgets.QLabel("DCA (Unsupervised)")
        self.hybrid_text = QtWidgets.QLabel("Hybrid")
        self.hybrid_text_train_test = QtWidgets.QLabel("Train - Test")
        self.hybrid_text_predict = QtWidgets.QLabel("Predict")
        self.hybrid_text = QtWidgets.QLabel("Hybrid (DCA-Supervised)")
        self.supervised_text = QtWidgets.QLabel("Supervised")

        for txt in [
            self.version_text, self.ncores_text, self.regression_model_text, 
            self.utils_text, self.dca_text, self.hybrid_text, self.supervised_text,
            self.hybrid_text_train_test, self.hybrid_text_predict, self.hybrid_text
        ]:
            txt.setStyleSheet(text_style)

        self.textedit_out = QtWidgets.QTextEdit(readOnly=True)
        self.textedit_out.setStyleSheet(
            "font-family:Consolas;font-size:12px;font-weight:normal;color:white;"
            "background-color:rgb(54, 69, 79);border:2px solid rgb(52, 59, 72);"
        )

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
        # Buttons ###########################################################################
        # Utilities
        self.button_help = QtWidgets.QPushButton("Help")  
        self.button_help.setToolTip("Print help text")
        self.button_help.clicked.connect(self.pypef_help)
        self.button_help.setStyleSheet(button_style)

        self.button_mklsts = QtWidgets.QPushButton("Create LS and TS (MKLSTS)")       
        self.button_mklsts.setToolTip("Create files for training and testing from variant-fitness CSV data")
        self.button_mklsts.clicked.connect(self.pypef_mklsts)
        self.button_mklsts.setStyleSheet(button_style)
        
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

        # Hybrid
        self.button_hybrid_train_dca = QtWidgets.QPushButton("Training (DCA)")
        self.button_hybrid_train_dca.setMinimumWidth(80)
        self.button_hybrid_train_dca.setToolTip(
            "Optimize the GREMLIN model by supervised training on variant-fitness labels"
        )
        self.button_hybrid_train_dca.clicked.connect(self.pypef_dca_hybrid_train)
        self.button_hybrid_train_dca.setStyleSheet(button_style)

        self.button_hybrid_train_test_dca = QtWidgets.QPushButton("Train-Test (DCA)")
        self.button_hybrid_train_test_dca.setMinimumWidth(80)
        self.button_hybrid_train_test_dca.setToolTip(
            "Optimize the GREMLIN model by supervised training on variant-fitness labels"
        )
        self.button_hybrid_train_test_dca.clicked.connect(self.pypef_dca_hybrid_train_test)
        self.button_hybrid_train_test_dca.setStyleSheet(button_style)

        self.button_hybrid_train_test_dca = QtWidgets.QPushButton("Train-Test (DCA)")
        self.button_hybrid_train_test_dca.setMinimumWidth(80)
        self.button_hybrid_train_test_dca.setToolTip(
            "Optimize the GREMLIN model by supervised training on variant-fitness labels and testing the model on a test set"
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

        # Supervised
        self.button_supervised_train_test_dca = QtWidgets.QPushButton("Train-Test (DCA)")
        self.button_supervised_train_test_dca.setMinimumWidth(80)
        self.button_supervised_train_test_dca.setToolTip(
            "Purely supervised DCA (GREMLIN or PLMC) model training on variant-fitness labels"
        )
        self.button_supervised_train_test_dca.clicked.connect(self.pypef_dca_supervised_train_test)
        self.button_supervised_train_test_dca.setStyleSheet(button_style)

        self.button_supervised_train_test_onehot = QtWidgets.QPushButton("Train-Test (One-hot)")
        self.button_supervised_train_test_onehot.setMinimumWidth(80)
        self.button_supervised_train_test_onehot.setToolTip(
            "Purely supervised one-hot model training on variant-fitness labels"
        )
        self.button_supervised_train_test_onehot.clicked.connect(self.pypef_onehot_supervised_train_test)
        self.button_supervised_train_test_onehot.setStyleSheet(button_style)

        # Layout widgets ####################################################################
        # int fromRow, int fromColumn, int rowSpan, int columnSpan
        layout.addWidget(self.version_text, 0, 0, 1, -1)
        layout.addWidget(self.ncores_text, 1, 0, 1, 1)
        layout.addWidget(self.box_multicore, 2, 0, 1, 1)

        layout.addWidget(self.utils_text, 3, 0, 1, 1)
        layout.addWidget(self.button_help, 4, 0, 1, 1)
        layout.addWidget(self.button_mklsts, 5, 0, 1, 1)

        layout.addWidget(self.dca_text, 3, 1, 1, 1)
        layout.addWidget(self.button_dca_inference_gremlin, 4, 1, 1, 1)
        layout.addWidget(self.button_dca_inference_gremlin_msa_info, 5, 1, 1, 1)
        layout.addWidget(self.button_dca_test_dca, 6, 1, 1, 1)
        layout.addWidget(self.button_dca_predict_dca, 7, 1, 1, 1)

        layout.addWidget(self.hybrid_text, 3, 2, 1, 1)
        layout.addWidget(self.button_hybrid_train_dca, 4, 2, 1, 1)
        layout.addWidget(self.button_hybrid_train_test_dca, 5, 2, 1, 1)
        layout.addWidget(self.button_hybrid_test_dca, 6, 2, 1, 1)

        layout.addWidget(self.regression_model_text, 1, 3, 1, 1)
        layout.addWidget(self.box_regression_model, 2, 3, 1, 1)
        layout.addWidget(self.supervised_text, 3, 3, 1, 1)
        layout.addWidget(self.button_supervised_train_test_dca, 4, 3, 1, 1)
        layout.addWidget(self.button_supervised_train_test_onehot, 5, 3, 1, 1)

        layout.addWidget(self.textedit_out, 8, 0, 1, -1)

        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.on_readyReadStandardOutput)
        self.process.started.connect(lambda: self.button_help.setEnabled(False))
        self.process.finished.connect(lambda: self.button_help.setEnabled(True))
        self.process.started.connect(lambda: self.button_mklsts.setEnabled(False))
        self.process.finished.connect(lambda: self.button_mklsts.setEnabled(True))
        self.process.started.connect(lambda: self.button_dca_inference_gremlin.setEnabled(False))
        self.process.finished.connect(lambda: self.button_dca_inference_gremlin.setEnabled(True))
        self.process.started.connect(lambda: self.button_dca_inference_gremlin_msa_info.setEnabled(False))
        self.process.finished.connect(lambda: self.button_dca_inference_gremlin_msa_info.setEnabled(True))
        self.process.started.connect(lambda: self.button_dca_test_dca.setEnabled(False))
        self.process.finished.connect(lambda: self.button_dca_test_dca.setEnabled(True))
        self.process.started.connect(lambda: self.button_dca_predict_dca.setEnabled(False))
        self.process.finished.connect(lambda: self.button_dca_predict_dca.setEnabled(True))
        self.process.started.connect(lambda: self.button_hybrid_train_dca.setEnabled(False))
        self.process.finished.connect(lambda: self.button_hybrid_train_dca.setEnabled(True))
        self.process.started.connect(lambda: self.button_hybrid_train_test_dca.setEnabled(False))
        self.process.finished.connect(lambda: self.button_hybrid_train_test_dca.setEnabled(True))
        self.process.started.connect(lambda: self.button_hybrid_test_dca.setEnabled(False))
        self.process.finished.connect(lambda: self.button_hybrid_test_dca.setEnabled(True))
        self.process.started.connect(lambda: self.button_supervised_train_test_dca.setEnabled(False))
        self.process.finished.connect(lambda: self.button_supervised_train_test_dca.setEnabled(True))
        self.process.started.connect(lambda: self.button_supervised_train_test_onehot.setEnabled(False))
        self.process.finished.connect(lambda: self.button_supervised_train_test_onehot.setEnabled(True))


    def on_readyReadStandardOutput(self):
         text = self.process.readAllStandardOutput().data().decode()
         self.c += 1
         self.textedit_out.append(text.strip())

    
    def selection_ncores(self, i):
        if i == 0:
            self.n_cores = 1
        elif i == 1:
            self.n_cores = os.cpu_count()


    def selection_regression_model(self, i):
        self.regression_model = [r.lower() for r in self.regression_models][i]


    @QtCore.Slot()
    def pypef_help(self):
        self.version_text.setText("Getting help...")
        self.exec_pypef(f'mklsts --help')


    @QtCore.Slot()
    def pypef_mklsts(self):
        wt_fasta_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select WT FASTA File")[0]
        csv_variant_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select variant CSV File")[0]
        if wt_fasta_file and csv_variant_file:
            self.version_text.setText("Running MKLSTS...")
            self.exec_pypef(f'mklsts --wt {wt_fasta_file} --input {csv_variant_file}')


    @QtCore.Slot()
    def pypef_gremlin(self):
        wt_fasta_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select WT FASTA File")[0]
        msa_file = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Multiple Sequence Alignment (MSA) file (in FASTA or A2M format)")[0]
        if wt_fasta_file and msa_file:
            self.version_text.setText("Running GREMLIN (DCA) optimization on MSA...")
            self.exec_pypef(f'param_inference --wt {wt_fasta_file} --msa {msa_file}')  # --opt_iter 100

    @QtCore.Slot()
    def pypef_gremlin_msa_info(self):
        wt_fasta_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select WT FASTA File")[0]
        msa_file = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Multiple Sequence Alignment (MSA) file (in FASTA or A2M format)")[0]
        if wt_fasta_file and msa_file:
            self.version_text.setText("Running GREMLIN (DCA) optimization on MSA...")
            self.exec_pypef(f'save_msa_info --wt {wt_fasta_file} --msa {msa_file}')

    @QtCore.Slot()
    def pypef_dca_test(self):
        test_set_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Test Set File in \"FASL\" format")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self, "Parameter Pickle file")[0]
        if test_set_file and params_pkl_file:
            self.version_text.setText("Testing DCA performance on provided test set...")
            self.exec_pypef(f'hybrid --ts {test_set_file} -m {params_pkl_file} --params {params_pkl_file}')


    @QtCore.Slot()
    def pypef_dca_predict(self):
        prediction_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Prediction Set File in FASTA format")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select DCA parameter Pickle file")[0]
        if prediction_file and params_pkl_file:
            self.version_text.setText("Predicting using the DCA model on provided prediction set...")
            self.exec_pypef(f'hybrid --ps {prediction_file} -m {params_pkl_file} --params {params_pkl_file}')


    @QtCore.Slot()
    def pypef_dca_hybrid_train(self):
        training_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Training Set File in \"FASL\" format")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select DCA parameter Pickle file")[0]
        if training_file and params_pkl_file:
            self.version_text.setText("Hybrid (DCA-supervised) model training...")
            self.exec_pypef(f'hybrid --ls {training_file} --ts {training_file} -m {params_pkl_file} --params {params_pkl_file}')


    @QtCore.Slot()
    def pypef_dca_hybrid_train_test(self):
        training_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Training Set File in \"FASL\" format")[0]
        test_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Test Set File in \"FASL\" format")[0]
        model_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select DCA model Pickle file")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select DCA parameter Pickle file")[0]
        if training_file and test_file and params_pkl_file:
            self.version_text.setText("Hybrid (DCA-supervised) model training and testing...")
            self.exec_pypef(f'hybrid --ls {training_file} --ts {test_file} -m {model_pkl_file} --params {params_pkl_file}')


    @QtCore.Slot()
    def pypef_dca_hybrid_test(self):
        test_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Test Set File in \"FASL\" format")[0]
        model_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Hybrid Model file in Pickle format")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select DCA parameter Pickle file")[0]
        if test_file and params_pkl_file:
            self.version_text.setText("Hybrid (DCA-supervised) model training and testing...")
            self.exec_pypef(f'hybrid --ts {test_file} -m {model_file} --params {params_pkl_file}')


    @QtCore.Slot()
    def pypef_dca_supervised_train_test(self):
        training_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Training Set File in \"FASL\" format")[0]
        test_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Test Set File in \"FASL\" format")[0]
        params_pkl_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select DCA parameter Pickle file")[0]
        if training_file and test_file and params_pkl_file:
            self.version_text.setText("Hybrid (DCA-supervised) model training and testing...")
            self.exec_pypef(f'ml --encoding dca --ls {training_file} --ts {test_file} --params {params_pkl_file} '
                            f'--threads {self.n_cores} --regressor {self.regression_model}')


    @QtCore.Slot()
    def pypef_onehot_supervised_train_test(self):
        training_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Training Set File in \"FASL\" format")[0]
        test_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Test Set File in \"FASL\" format")[0]
        if training_file and test_file:
            self.version_text.setText("Hybrid (DCA-supervised) model training and testing...")
            self.exec_pypef(f'ml --encoding onehot --ls {training_file} --ts {test_file} '
                            f'--threads {self.n_cores} --regressor {self.regression_model}')
    

    def exec_pypef(self, cmd):
        self.process.start(f'python', ['-u', f'{self.pypef_root}/run.py'] + cmd.split(' '))
        self.process.finished.connect(self.process_finished)
        if self.c > 0:
            self.textedit_out.append("=" * 104 + "\n")


    def process_finished(self):
        self.version_text.setText("Finished...") 
        #self.process = None


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = MainWindow()
    widget.resize(800, 600)
    widget.show()
    sys.exit(app.exec())

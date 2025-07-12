# A View Across the Skyline

This project explores **nowcasting economic indicators** by combining **satellite imagery** and **regional energy features** using deep learning techniques.

📄 **Related Paper**:  
Available at SSRN: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5101867](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5101867)

---

## 🗂️ Project Structure

```
A-View-Across-the-Skyline-Nowcasting-Combining-Satellite-and-Energy-Indicators/
├── Data/                    # Contains raw and processed geographic/economic data
│   ├── Countshape/         # County shape files (.shp, .dbf, .shx, etc.)
│   ├── Label/
│   └── Raw/
├── GDP/
│   ├── Difference/         # Files for modeling economic *differences*
│   └── Level/              # Files for modeling economic *levels*
├── Totalemployement/
│   ├── Difference/
│   └── Level/
└── README.md
```

---

## 🚀 How to Use

### 1. Environment Setup

Make sure you have Python installed along with the following:

- `PyTorch`
- `NumPy`
- `scikit-learn`
- `rasterio`
- `matplotlib`

Install missing packages via:

```bash
pip install -r requirements.txt
```

### 2. Model Training

You can train models by running the following Python scripts or Jupyter notebooks in the respective subfolders:

- `GDP/Difference/main_train_diff_model.py`
- `GDP/Level/GDP_main_train.py`
- `Totalemployement/Difference/traintest.py`
- `Totalemployement/Level/TE_main_train.py`

### 3. Evaluation and Visualization

Use the corresponding `*_evaluate.py` or `.ipynb` notebooks in each folder to evaluate model performance and visualize results.

---

## 📌 Notes

- Geospatial data uses U.S. county-level boundaries and NREL datasets.
- Models are based on ResNet-50, enhanced with adaptive feature fusion modules (AFF).
- Training and testing results are logged and saved locally.

---

## 🙌 Acknowledgments

Thanks to Lilin Wang for foundational contributions.

For more details, refer to the [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5101867).

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.title("Rakuten: multi-modal product classification project")
st.sidebar.title("Table of contents")
pages=["Introduction", "Data Exploration", "Data Preprocessing", "Modeling", "DEMO app"]
page=st.sidebar.radio("Go to", pages)

original_dataset = pd.read_csv('../data/X_train.csv', index_col=0)
y_ori = pd.read_csv('../data/Y_train.csv', index_col=0)
original_dataset = pd.concat([original_dataset, y_ori], axis=1)

labels = pd.read_csv('data/class_category.csv', delimiter=';', index_col=None)
labels.columns = ["Product Type Code", "Product Category"]
labels.index = [''] * len(labels) # set the index to an empty string to hide row index

img_class_dist = Image.open("figures/class_dist.jpg").convert("RGB")
width, height = img_class_dist.size
img_class_dist = img_class_dist.crop((0, 60, width, height))

if page == pages[0]:
    st.write("### Introduction of the project")

if page == pages[1]:
    st.write("### Presentation of Data")
    st.markdown("#### 📸 Overview of Original Data")
    st.write("The original data included four columns as explanatory variables (i.e., 'designation', 'description', 'productid' and 'imageid') and one target column named 'prdtypecode'." \
        "\nBoth 'designation' and 'description' are going to be merged as the textual data, while 'productid' and 'imageid' are used for extracting image file for the product."
    )
    st.dataframe(original_dataset.head())

    st.markdown("---")
    st.markdown("#### 🔍 Class Categories")
    st.write("There are a total of 84,915 data samples available, with 27 classes representing different product categories to be classified.")
    with st.expander("Detailed Class Categories"):
        st.table(labels)
    st.write("There are a priori some specific classes that are frequently confused even by human. Examples are the following:")
    st.markdown("""
        - Class 10 (used book) and class 2705 (new book)
        - Class 1180 (board games), class 1280 (chidren's toys) and class 1281 (social games)
        - Class 40 (video game), class 60 (console game) and class 2905 (PC game)
        """
    )
    st.markdown("---")
    st.markdown("#### 📊 Class Distribution")
    st.write("The target classes of the entire data samples are imbalanced, with the majority class 2583 (poolside items) containing over 10k samples, while the minority classes represent" \
        "only about 1% of the entire data.")
    st.image(img_class_dist, caption="Distribution of Data Samples across Target Classes", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📌 Data Examples")

if page == pages[2]:
    st.write("### Preprocessing of data")
    tab1, tab2 = st.tabs(["Textual Data", "Image Data"])

    # Image Data Preprocessing
    with tab2: 
        st.markdown("#### 📸 Overview")
        st.write(
        "The original image dataset included 500×500×3 JPEG files, one per product. "
        "The preprocessing pipeline optimized these images for model training by improving focus, filtering poor-quality images, and preparing the dataset for efficient loading."
        )

        st.markdown("---")
        st.markdown("#### 🔍 Image Zoom and Filtering")

        st.write(
        "Many images had large white margins, so the team:\n"
        "- Calculated **content ratio** (non-white pixels / total pixels).\n"
        "- Cropped to the bounding box of non-white regions + 10px padding.\n"
        "- Resized to **224×224×3**.\n"
        "- Removed **4 blank images** and **408 low-content images** (threshold: 0.04)."
        )


        st.markdown("---")
        st.markdown("#### 🏷️ Class Balancing & Label Encoding")

        st.write(
        "To prepare for training:\n"
        "- Merged metadata with class labels (`predtypecode`).\n"
        "- Used **LabelEncoder** and did an **80/20 stratified train-test split**.\n"
        "- Balanced classes by undersampling or generating augmented samples."
        )

        st.image("figures/img_augm_dist.png", caption="Distribution of original, dropped, and augmented samples", use_container_width=True)

        st.markdown("---")
        st.markdown("#### ⚙️ Dataset Assembly and tf.data Pipeline")

        st.write(
        "Used `tf.data` to stream images from disk:\n"
        "- Inputs: `imageid`, `productid`, `label`, `augmented` flag.\n"
        "- Reconstructed image paths dynamically.\n"
        "- Applied **model-specific preprocessing** (EfficientNetB0, ConvNeXtTiny, DenseNet201).\n"
        "- Augmentation only if `augmented=True` (flip, brightness, translation).\n"
        "- Validation set used unaugmented data."
        )

        st.markdown("**⚡ Optimizations applied:**")
        st.markdown("- Shuffling with buffer size 1000\n- Batching\n- Prefetching for faster loading")


if page == pages[3]:
    st.write("### Modeling")
    tab1, tab2 = st.tabs(["Textual Data", "Image Data"])

    # Image Modeling
    with tab2: 
        # Image Classification Modeling Summary
        st.markdown("#### 🧠 Model Overview")
        st.write(
            "This section outlines the classification models used to assign one of 27 labels to product images. "
            "The study compared **EfficientNetB0**, **ConvNeXtTiny**, and **DenseNet201** architectures using transfer learning and custom classification heads."
        )

        st.markdown("---")
        st.markdown("#### 🔁 Transfer Learning Setup")

        st.write(
            "Each pre-trained backbone was followed by a custom head:\n"
            "- Global average pooling\n"
            "- Two dense layers with dropout and L2 regularization\n"
            "- Final softmax output\n\n"
            "This setup helped generalize better and control overfitting."
        )

        st.code(
            "models.Sequential([\n"
            "    base_model,\n"
            "    layers.GlobalAveragePooling2D(),\n"
            "    layers.BatchNormalization(),\n"
            "    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),\n"
            "    layers.Dropout(0.5),\n"
            "    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),\n"
            "    layers.Dropout(0.3),\n"
            "    layers.Dense(num_classes, activation='softmax')\n"
            "])",
            language="python"
        )

        st.markdown("---")
        st.markdown("#### ⚙️ Training Configuration")

        st.write(
            "All models used:\n"
            "- **Sparse Categorical Crossentropy** loss\n"
            "- **Sparse Categorical Accuracy** for evaluation\n"
            "- **EarlyStopping**, **ModelCheckpoint**, and **ReduceLROnPlateau** callbacks for efficient training"
        )
        with st.expander("🛠️ Callback Configuration Details"):
            st.markdown("**EarlyStopping**")
            st.code(
                "early_stop = EarlyStopping(\n"
                "    monitor='val_loss',\n"
                "    patience=3,\n"
                "    min_delta=0.02,\n"
                "    restore_best_weights=True\n"
                ")",
                language="python"
            )


        st.markdown("---")
        st.markdown("#### 📊 Model Comparison")

        st.write(
            "Eight variants were tested with different layer unfreezing depths and learning rates. "
            "While `efficientnetb0_l5lr` had the highest validation accuracy, its higher overfitting led to selecting `efficientnetb0_l5` as the final model due to better generalization."
        )

        st.dataframe({
            "Model": ["efficientnetb0_l5", "efficientnetb0_l5lr", "convnexttiny_l5lr", "densenet_l5"],
            "Train Acc": [0.63, 0.66, 0.58, 0.62],
            "Val Acc": [0.61, 0.61, 0.58, 0.56],
            "Gen. Gap": [0.027, 0.051, -0.004, 0.062]
        })

        st.markdown("---")
        st.markdown("#### 🏆 Best Model Performance")

        st.write(
            "`efficientnetb0_l5` achieved a **weighted F1-score of 0.56** across 23 classes, with **overall accuracy of 58%**.\n"
            "Precision and recall varied across classes, reflecting class difficulty and visual ambiguity."
        )

        with st.expander("🔍 Show Confusion Matrix"):
            st.image("figures/img_cm.png", caption="Confusion Matrix of the Best Model", use_container_width=True)

        st.markdown("---")
        st.markdown("#### ❌ Common Misclassifications")

        st.write(
            "Two frequent sources of confusion were:\n"
            "- **Used book (Class 10)** vs **New book (Class 2705)**: difficult to distinguish from covers alone.\n"
            "- **Video game, tech accessory (Class 40)** vs **PC game (Class 2905)**: similar design, overlapping concepts."
        )

    

if page == pages[4]:
    st.write("### DEMO app: try out with your own input")
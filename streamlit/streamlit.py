import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
import random
import cv2
import re, html
import torch
from keras.models import load_model
from keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import img_to_array
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(layout="wide")
image_width = 800

st.title("Rakuten: multi-modal product classification project")
st.sidebar.image("figures/data_scientest.png")
st.sidebar.title("Table of Contents")
pages=["Introduction", "Data Exploration", "Data Preprocessing", "Modeling", "Conclusion", "DEMO App"]
page=st.sidebar.radio("Go to", pages)
st.sidebar.markdown("<div style='height:200px;'></div>", unsafe_allow_html=True) # Spacer to push footer down
st.sidebar.markdown(
    "<div style='text-align:left; color:gray; font-size:16px;'>"
    "Contributors:<br>"
    "<b>Deniz AYDIN</b><br>"
    "<b>Shiyu LIU</b><br><br>"
    "Mentor:<br>"
    "<b>M. Yaniv BENICHOU</b><br><br>"
    "<a href='https://github.com/Shiyu-Liu/RAKUTEN_PROJECT'>[Github]</a><br>"
    "June 2025"
    "</div>",
    unsafe_allow_html=True
)

original_dataset = pd.read_csv('../data/X_train.csv', index_col=0)
y_ori = pd.read_csv('../data/Y_train.csv', index_col=0)
original_dataset = pd.concat([original_dataset, y_ori], axis=1)
preprocessed_dataset = pd.read_csv('../data/text_data_clean.csv', delimiter=';', index_col=0)

labels = pd.read_csv('data/class_category.csv', delimiter=';', index_col=None)
labels.columns = ["Product Type Code", "Product Category"]
labels['Product Type Code'] = labels['Product Type Code'].astype(str)

ori_dataset_backend = original_dataset.copy()
ori_dataset_backend['filename'] = "image_"+ori_dataset_backend['imageid'].astype(str)+"_product_"+ori_dataset_backend['productid'].astype(str)+".jpg"
ori_dataset_backend['description'] = ori_dataset_backend['description'].fillna("")
ori_dataset_backend['text'] = ori_dataset_backend['designation'] + " " + ori_dataset_backend['description']

img_class_dist = Image.open("figures/class_dist.jpg").convert("RGB")
width, height = img_class_dist.size
img_class_dist = img_class_dist.crop((0, 60, width, height))

distilbert_results = pd.read_csv('data/distilbert_results.csv')

rakuten_logo = Image.open("figures/rakuten.jpg").convert("RGB")
width, height = rakuten_logo.size
rakuten_logo = rakuten_logo.crop((20, 100, width, height-120))

word_cloud_img = Image.open("figures/word_cloud.png").convert("RGB")
width, height = word_cloud_img.size
word_cloud_img = word_cloud_img.crop((100, 30, width-100, height-60))

if page == pages[0]:
    st.write("### Introduction of the Project")
    st.markdown("#### 🛍️ Business Problem")

    st.write(
        "Rakuten France manages a massive and growing product catalog from various sellers. "
        "Accurate categorization within their product hierarchy is vital for search, recommendations, and inventory management. "
        "Current categorization relies heavily on seller metadata, which is often inconsistent—leading to ambiguity and costly manual corrections."
    )

    st.markdown("---")
    st.markdown("#### 🎯 Objectives and Goals")

    st.write(
        "The project aimed to build a **multimodal classification model** predicting product type codes using both text and image inputs. Key goals included:\n"
        "- Developing separate **text** and **image** models to assess each modality's contribution.\n"
        "- Combining them using **late fusion** for a stronger multimodal system.\n"
        "- Evaluating and comparing unimodal vs multimodal model performance."
    )

    st.markdown(
        "The dataset was sourced from Rakuten Institute of Technology's Challenge Data platform and used for research purposes. https://challengedata.ens.fr/challenges/35"
    )
    st.image(rakuten_logo, width=int(image_width*0.5))

    st.markdown("---")
    st.markdown("#### 📋 Task Allocation")
    st.write("The tasks were split between two team members as follows:\n"
        "- Shiyu: Textual data preprocessing and training of the textual model.\n"
        "- Deniz: Image preprocessing and training of the image model.\n"
    )

if page == pages[1]:
    st.write("### Presentation of Data")
    st.markdown("#### 📸 Overview of Original Data")
    st.write("The original data included four columns as explanatory variables (i.e., 'designation', 'description', 'productid' and 'imageid') and one target column named 'prdtypecode'." \
        "\nThe 'designation' and 'description' columns are merged into a single textual description of products, while 'productid' and 'imageid' are used to retrieve the associated image files."
    )
    st.dataframe(original_dataset.head())

    st.markdown("---")
    st.markdown("#### 🔍 Class Categories")
    st.write("There are a total of 84,915 data samples available, with 27 classes representing different product categories to be classified.")
    col1, _ = st.columns([3,2])
    with col1:
        with st.expander("Detailed Class Categories"):
            st.dataframe(labels, hide_index=True)
    st.write("There are, a priori, certain specific classes that are frequently confused even by humans. Examples are the following:")
    st.markdown("""
        - Class 10 (used book) and class 2705 (new book)
        - Class 1180 (board games), class 1280 (children's toys) and class 1281 (social games)
        - Class 40 (video game) and class 2905 (PC game)
        """
    )
    st.markdown("---")
    st.markdown("#### 📊 Class Distribution")
    st.write("The target classes in the dataset are imbalanced: the majority class, 2583 (poolside items), contains over 10,000 samples, while the minority classes each represent" \
        " only about 1% of the total data.")
    st.image(img_class_dist, caption="Distribution of Data Samples across Target Classes", width=image_width)

    st.markdown("---")
    st.markdown("#### 📌 Data Examples")
    st.markdown("🔺 **Class 10 (used book) V.S. Class 2705 (new book)**")
    _, col1, col2, _, _ = st.columns(5)
    with col1:
        with st.container():
            st.markdown(
                "<div style='text-align: center; color: gray;'>Example: Class 10 (used book)</div>",
                unsafe_allow_html=True
            )
            st.image("figures/examples/class_10_index_477.jpg")
            with st.expander("Text Description"):
                st.markdown(f"**Text:** {ori_dataset_backend.loc[477,'text']}")
    with col2:
        with st.container():
            st.markdown(
                "<div style='text-align: center; color: gray;'>Example: Class 2705 (new book)</div>",
                unsafe_allow_html=True
            )
            st.image("figures/examples/class_2705_index_82160.jpg")
            with st.expander("Text Description"):
                st.markdown(f"**Text:** {ori_dataset_backend.loc[82160,'text']}")
    st.markdown("\n\n")
    st.markdown("🔺 **Class 1180 (board games) V.S. Class 1280 (children's toy) V.S. Class 1281 (social games)**")
    col1, col2, col3, _ = st.columns(4)
    with col1:
        with st.container():
            st.markdown(
                "<div style='text-align: center; color: gray;'>Example: Class 1180 (board games)</div>",
                unsafe_allow_html=True
            )
            st.image("figures/examples/class_1180_index_8206.jpg")
            with st.expander("Text Description"):
                st.markdown(f"**Text:** {ori_dataset_backend.loc[8206,'text']}")
    with col2:
        with st.container():
            st.markdown(
                "<div style='text-align: center; color: gray;'>Example: Class 1280 (children's toy)</div>",
                unsafe_allow_html=True
            )
            st.image("figures/examples/class_1280_index_5601.jpg")
            with st.expander("Text Description"):
                st.markdown(f"**Text:** {ori_dataset_backend.loc[5601,'text']}")
    with col3:
        with st.container():
            st.markdown(
                "<div style='text-align: center; color: gray;'>Example: Class 1281 (social games)</div>",
                unsafe_allow_html=True
            )
            st.image("figures/examples/class_1281_index_4467.jpg")
            with st.expander("Text Description"):
                st.markdown(f"**Text:** {ori_dataset_backend.loc[4467,'text']}")
    st.markdown("\n\n")
    st.markdown("🔺 **Class 40 (video game) V.S. Class 2905 (PC game)**")
    _, col1, col2, _, _ = st.columns(5)
    with col1:
        with st.container():
            st.markdown(
                "<div style='text-align: center; color: gray;'>Example: Class 40 (video game)</div>",
                unsafe_allow_html=True
            )
            st.image("figures/examples/class_40_index_2573.jpg")
            with st.expander("Text Description"):
                st.markdown(f"**Text:** {ori_dataset_backend.loc[2573,'text']}")
    with col2:
        with st.container():
            st.markdown(
                "<div style='text-align: center; color: gray;'>Example: Class 2905 (PC game)</div>",
                unsafe_allow_html=True
            )
            st.image("figures/examples/class_2905_index_33509.jpg")
            with st.expander("Text Description"):
                st.markdown(f"**Text:** {ori_dataset_backend.loc[33509,'text']}")

if page == pages[2]:
    st.write("### Preprocessing of Data")
    tab1, tab2 = st.tabs(["Textual Data", "Image Data"])

    # Textual Data Preprocessing
    with tab1:
        st.markdown("#### 📸 General Issues")
        st.markdown("""
            - Presence of special characters and HTML tags, which carry no semantic meaning.
            - Excessively long texts in some of the samples caused by overly detailed specifications.
            - Presence of a variety of languages, apart from the majority of the text in Frence.
            - Class imbalance among all the samples.
            """
        )

        st.markdown("---")
        st.markdown("#### 🔍 Text Cleaning and Filtering")
        st.write("Text cleaning was performed on all the data samples, more specifically, the team has:")
        st.markdown("""
            - Removed HTML tags and special characters except a few symbols (e.g., é, ', ß);
            - Eliminated common unit patterns such as "xx cm", "xx kg", "Axx", "Øxx", and "N°";
            - Removed the remaining numerical contents that indicates dimensions.
            """
        )

        st.markdown("\n")
        st.write("In terms of text filtering, we set up filtering criteria as follows:")
        st.markdown("""
            - Total word count of each sample's text within **100 words**;
            - Target **sample size of 2500** per class.
            """
        )
        col1, _ = st.columns([4,2])
        with col1:
            with st.expander("Distribution of Word Counts"):
                st.image("figures/word_count_dist.jpg", caption="Distribution of word counts across target classes", use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🏷️ Text Translation & Class Balancing")
        st.write("We leveraged the OpenAI-API and used ChatGPT-4.1-nano model to translate all the text into a unified language. The target language is **English**.")
        col1, _ = st.columns([4,2])
        with col1:
            with st.expander("Distribution of Text Languages"):
                st.image("figures/language_dist.jpg", caption="Distribution of data samples across target classes and languages", use_container_width=True)

        st.write("The ChatGPT-4.1-nano model is also used to generate dummy text by paraphrasing existing samples from the minority classes." \
            " Examples are:")
        left, _ = st.columns([1,3])
        with left:
            st.latex(r"""
                \begin{array}{|c|l|}
                    \hline
                    \small \textit{Original} & \small \textit{The app keeps crashing when I try to open it.} \\
                    \hline
                    \small \text{Paraphrase 1} & \small \text{The app crashes every time I attempt to open it.} \\
                    \small \text{Paraphrase 2} & \small \text{Whenever I try to launch the app, it shuts down unexpectedly.} \\
                    \hline
                    \small \textit{Original} & \small \textit{I can't log into my account with the correct password.} \\
                    \hline
                    \small \text{Paraphrase 1} & \small \text{I'm unable to access my account even though I'm using the correct password.} \\
                    \small \text{Paraphrase 2} & \small \text{Despite entering the right password, I can't sign into my account.} \\
                    \hline
                \end{array}
            """)

        st.markdown("---")
        st.markdown("#### 🏁 Class Filtering and Balancing Results:")
        st.image("figures/text_aug_dist.png", caption="Distribution of original, dropped, and augmented samples", width=image_width)
        st.image(word_cloud_img, caption="Word cloud from the preprocessed textual data", width=image_width)

        st.markdown("---")
        st.markdown("#### 📌 Examples of Preprocessing Results")
        st.write("Click the button to generate randomly preprocessing results:")
        if st.button("Click Me"):
            st.markdown("**Input Text:**")
            idx = np.random.randint(0, ori_dataset_backend.shape[0])
            st.write(f"{ori_dataset_backend.loc[idx,'text']}")
            st.markdown(f"**Class: {ori_dataset_backend.loc[idx,'prdtypecode']}**")
            st.markdown("**Preprocessed Text:**")
            if idx in preprocessed_dataset.index:
                st.write(f"{preprocessed_dataset.loc[idx,'text']}")
            else:
                st.write("N/A (*This data sample is dropped or filtered during the preprocessing process*)")

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

        st.markdown("#### 📌 Examples of Preprocessing Results")
  
        folder_zoom = "figures/image_zoom_examples"

        all_images = [f for f in os.listdir(folder_zoom) if f.startswith("original_image_") and f.endswith(".jpg")]

        if "show_comparison" not in st.session_state:
            st.session_state.show_comparison = False 

        if "current_image" not in st.session_state:
            st.session_state.current_image = random.choice(all_images)
            st.session_state.show_comparison = False

        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            if not st.session_state.show_comparison:
                button_text = "Show Before/After Comparison"
            else:
                button_text = "Get Another One"
            
            if st.button(button_text):
                if not st.session_state.show_comparison:
                    st.session_state.show_comparison = True
                else:
                    available_images = [img for img in all_images if img != st.session_state.current_image]
                    if available_images:
                        st.session_state.current_image = random.choice(available_images)

        if st.session_state.show_comparison:
            preproc_img_name = st.session_state.current_image.replace("original_image_", "preprocessed_image_")
            
            st.markdown("---")
            st.markdown(f"#### Before and After Comparison (Image ID: {st.session_state.current_image.split('_')[2]})")
            
            col_orig, col_preproc = st.columns(2)
            
            with col_orig:
                st.image(os.path.join(folder_zoom, st.session_state.current_image), 
                        caption="Original", 
                        use_container_width=True)
            
            with col_preproc:
                st.image(os.path.join(folder_zoom, preproc_img_name), 
                        caption="Preprocessed", 
                        use_container_width=True)


        st.markdown("---")
        st.markdown("#### 🏷️ Class Balancing & Label Encoding")

        st.write(
        "To prepare for training:\n"
        "- Merged metadata with class labels (`predtypecode`).\n"
        "- Used **LabelEncoder** and did an **80/20 stratified train-test split**.\n"
        "- Balanced classes by undersampling or generating augmented samples."
        )
        st.image("figures/img_augm_dist.png", caption="Distribution of original, dropped, and augmented samples", width=image_width)

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
    tab1, tab2, tab3 = st.tabs(["Textual Model", "Image Model", "Model Fusion"])

    # Text Modeling
    with tab1:
        st.markdown("#### 🧠 Model Overview")
        st.write("To design models for classifying products based on their textual description, we have applied:\n"
            "- Classical machine learning (ML) models, such as **Support Vector Machines (SVM)**, **Random Forests (RF)** and **Extreme Gradient Boosting (XGBoost)**;\n"
            "- Pretrained large language model (LLM) based on **DistilBERT**."
        )

        st.markdown("---")
        st.markdown("#### ✏️ Classical ML Models")
        st.write("For training the classical ML models, we first applied the TF-IDF vectorizer to the original text, and then used a " \
            "grid search-based approach to find the best hyperparameters retained in the model.")
        col1, _ = st.columns([3,2])
        with col1:
            with st.expander("🛠️ Model Hyperparameter Details"):
                st.write("**SVM**:")
                st.code("SVC(C=1, loss='squared_hinge', kernel='linear')")
                st.write("**RF**")
                st.code("RandomForestClassifier(n_estimators=200, max_depth=None)")
                st.write("**XGBoost**")
                st.code("XGBClassifier(n_estimators=200, max_depth=None)")

        col1, _ = st.columns([3,2])
        with col1:
            df = pd.DataFrame({
                "Model": ["SVM", "RF", "XGBoost"],
                "Train Acc": ["81.5%", "95.4%", "85.2%"],
                "Train F1-score": ["81.5%", "95.6%", "85.7%"],
                "Val Acc": ["71.9%", "71.3%", "71.3%"],
                "Val F1-score": ["71.8%", "71.4%", "71.9%"]
            })
            st.dataframe(df, hide_index=True)
        st.write("*Note that F1-score is computed using weighted averaging method.*")
        st.write("Based on the results, we retained **XGBoost** as the best model among these classical ML algorithms.")
        col1, _ = st.columns([4,2])
        with col1:
            with st.expander("🔍 Show Confusion Matrix"):
                st.image("figures/text_basic_cm.png", caption="Confusion Matrix of the XGBoost Model", use_container_width=True)

        st.markdown("---")
        st.markdown("#### 💡 DistilBERT LLM")
        st.write("DistilBERT is a virant of the original BERT large pretrained language model. We used the distilbert-based-uncase" \
            " model and fine-tuned it on our dataset with 6 epochs. We did a **70%/15%/15%** stratified train-validation-test split on our original dataset.")
        col1, _ = st.columns([4,2])
        with col1:
            with st.expander("🛠️ Training Configuration"):
                st.code("training_args = TrainingArguments(\n"
                        "   learning_rate=2e-5,\n"
                        "   per_device_train_batch_size=16,\n"
                        "   per_device_eval_batch_size=64,\n"
                        "   num_train_epochs=6,\n"
                        "   weight_decay=0.01,\n"
                        "   load_best_model_at_end=True,\n"
                    ")",
                    language="python"
                )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=distilbert_results['epoch'], y=distilbert_results['train_acc'], mode='lines+markers', name='Train Accuracy'))
        fig.add_trace(go.Scatter(x=distilbert_results['epoch'], y=distilbert_results['train_f1'], mode='lines+markers', name='Train F1-score'))
        fig.add_trace(go.Scatter(x=distilbert_results['epoch'], y=distilbert_results['eval_arr'], mode='lines+markers', name='Eval Accuracy'))
        fig.add_trace(go.Scatter(x=distilbert_results['epoch'], y=distilbert_results['eval_f1'], mode='lines+markers', name='Eval F1-score'))
        fig.update_layout(
            width=800,
            height=500,
            title='Train and Evaluation Metrics',
            xaxis_title='Epoch',
            yaxis_title='Scores',
            font=dict(size=14)
        )
        st.plotly_chart(fig, use_container_width=False)

        st.markdown("---")
        st.markdown("#### 🏆 DistilBERT Model Performance")
        st.write("After 6 epochs of training, the model performance evaluated on the **validation set** has achieved:")
        col1, col2, _ = st.columns([1,1,3])
        with col1:
            st.metric(label="**Accuracy**", value="84.5%")
        with col2:
            st.metric(label="**Weighted F1-score**", value="84.4%")
        st.write("Model performance evaluated on the **test set** has achieved:")
        col1, col2, _ = st.columns([1,1,3])
        with col1:
            st.metric(label="**Accuracy**", value="82.9%")
        with col2:
            st.metric(label="**Weighted F1-score**", value="82.9%")

        col1, _ = st.columns([4,2])
        with col1:
            with st.expander("🔍 Show Confusion Matrix"):
                st.image("figures/text_cm.png", caption="Confusion Matrix of the DistilBERT Model", use_container_width=True)

        st.markdown("---")
        st.markdown("#### ❌ Common Misclassifications")
        st.write(
            "Two frequent sources of confusion were:\n"
            "- **Used book (Class 10)** v.s. **New book (Class 2705)**: difficult to distinguish from text descriptions.\n"
            "- **Children's toy, costume (Class 1280)** v.s. **Social game (Class 1281)**: confusing terms."
        )

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

        col1, _ = st.columns([3,2])
        with col1:
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
        col1, _ = st.columns([3,2])
        with col1:
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

        col1, _ = st.columns([3,2])
        with col1:
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

        col1, _ = st.columns([4,2])
        with col1:
            with st.expander("🔍 Show Confusion Matrix"):
                st.image("figures/img_cm.png", caption="Confusion Matrix of the Best Model", use_container_width=True)

        st.markdown("---")
        st.markdown("#### ❌ Common Misclassifications")

        st.write(
            "Two frequent sources of confusion were:\n"
            "- **Used book (Class 10)** vs **New book (Class 2705)**: difficult to distinguish from covers alone.\n"
            "- **Video game, tech accessory (Class 40)** vs **PC game (Class 2905)**: similar design, overlapping concepts."
        )

    with tab3:
        st.markdown("#### 🪜 Fusion Strategy")
        st.write("We used a **soft voting strategy** to combine the probabilities of predicted class from the text and image-based models to make a final prediction using multi-modal information.")
        st.write("Based on the performance of each unimodal model, we assigned weights for averaging the predicted probabilities accordingly."
            " We also experimented with a configuration that assigns equal weights to both models for fusion."
        )
        col1, _ = st.columns([4,2])
        with col1:
            st.code("WEIGHTS = [w_text, w_image] = [0.6, 0.4]", language="python")

        st.write("We created a new test set for final evaluation, which excludes any samples that had been previously used for training either the text or image model.")
        col1, _ = st.columns([4,2])
        with col1:
            with st.expander("Test Set Distribution"):
                st.image("figures/test_set_dist.png", caption="Data distribution of test set for final evaluation", use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📊 Multi-Modal Model Performance")
        st.write("To make a fair comparison, the individual text and image-based models were also evaluated on the test set, and their performance is compared with that of the fused multi-modal model.")
        col1, _, col3, _, _ = st.columns(5)
        with col1:
            st.markdown("🔸 **Image Model**:")
        with col3:
            st.markdown("🔸 **Text Model**:")

        col1, col2, col3, col4, _ = st.columns(5)
        with col1:
            st.metric(label="**Accuracy**", value="58.1%")
        with col2:
            st.metric(label="**Weighted F1-score**", value="57.5%")
        with col3:
            st.metric(label="**Accuracy**", value="80.0%", delta="+21.9%")
            st.write("*Delta compared with Image Model*")
        with col4:
            st.metric(label="**Weighted F1-score**", value="79.9%", delta="+22.4%")

        col1, col2, _, = st.columns([2,2,1])
        with col1:
            st.markdown("🔸 **Fused Multi-Modal Model** (weights of [0.6, 0.4]):")
        with col2:
            st.markdown("🔸 **Fused Multi-Modal Model** (equal weights):")
        col1, col2, col3, col4, _ = st.columns(5)
        with col1:
            st.metric(label="**Accuracy**", value="81.3%", delta="+1.3%")
            st.write("*Delta compared with Text Model*")
        with col2:
            st.metric(label="**Weighted F1-score**", value="81.2%", delta="+1.3%")
        with col3:
            st.metric(label="**Accuracy**", value="81.8%", delta="+1.8%")
            st.write("*Delta compared with Text Model*")
        with col4:
            st.metric(label="**Weighted F1-score**", value="81.7%", delta="+1.8%")

        col1, _ = st.columns([4,2])
        with col1:
            with st.expander("F1-Score Comparison Details"):
                st.image("figures/f1_score_comparison.png", caption="Comparison of F1-scores for image, text and fused multi-modal model.", use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📌 Examples of Successful Predictions")
        st.write("We highlight the following examples to demonstrate that the fused model leverages both the individual text-based and image-based models to improve the final class prediction.")
        col1, col2, col3, col4, _ = st.columns([1,2,1,2,1])
        with col1:
            with st.container():
                st.image("../data/images/image_train_zoomed/{}".format(ori_dataset_backend.loc[477,'filename']))
                with st.expander("Text Description"):
                    st.markdown(f"{preprocessed_dataset.loc[477,'text']}")
        with col2:
            st.markdown("**Real Class**: 10 (used book)")
            st.markdown("**Text Model Prediction**:<br>10 (used book), confidence: 0.98", unsafe_allow_html=True)
            st.markdown("**Image Model Prediction**:<br>2705 (new book), confidence: 0.31", unsafe_allow_html=True)
            st.markdown("**Final Prediction**: 10 (used book)", unsafe_allow_html=True)
        with col3:
            with st.container():
                st.image("../data/images/image_train_zoomed/{}".format(ori_dataset_backend.loc[82160,'filename']))
                with st.expander("Text Description"):
                    st.markdown(f"{preprocessed_dataset.loc[82160,'text']}")
        with col4:
            st.markdown("**Real Class**: 2705 (new book)")
            st.markdown("**Text Model Prediction**:<br>10 (used book), confidence: 0.59", unsafe_allow_html=True)
            st.markdown("**Image Model Prediction**:<br>2705 (new book), confidence: 0.76", unsafe_allow_html=True)
            st.markdown("**Final Prediction**: 2705 (new book)", unsafe_allow_html=True)

        col1, col2, col3, col4, _ = st.columns([1,2,1,2,1])
        with col1:
            with st.container():
                st.image("../data/images/image_train_zoomed/{}".format(ori_dataset_backend.loc[2573,'filename']))
                with st.expander("Text Description"):
                    st.markdown(f"{preprocessed_dataset.loc[2573,'text']}")
        with col2:
            st.markdown("**Real Class**: 40 (video game)")
            st.markdown("**Text Model Prediction**:<br>40 (video game), confidence: 0.99", unsafe_allow_html=True)
            st.markdown("**Image Model Prediction**:<br>40 (video game), confidence: 0.67", unsafe_allow_html=True)
            st.markdown("**Final Prediction**: 40 (video game)", unsafe_allow_html=True)
        with col3:
            with st.container():
                st.image("../data/images/image_train_zoomed/{}".format(ori_dataset_backend.loc[33509,'filename']))
                with st.expander("Text Description"):
                    st.markdown(f"{preprocessed_dataset.loc[33509,'text']}")
        with col4:
            st.markdown("**Real Class**: 2905 (PC game)")
            st.markdown("**Text Model Prediction**:<br>2905 (PC game), confidence: 0.99", unsafe_allow_html=True)
            st.markdown("**Image Model Prediction**:<br>40 (video game), confidence: 0.59", unsafe_allow_html=True)
            st.markdown("**Final Prediction**: 2905 (PC game)", unsafe_allow_html=True)

if page == pages[4]:
    st.write("### Conclusion")
    st.markdown("#### ✅ Project Summary")

    st.write(
        "The project delivered a **multi-modal product classification model** that combined text and image data to improve accuracy. "
        "Separate text and image models were trained using traditional ML, deep learning, and large pre-trained models. "
        "A soft voting fusion combined their predictions, achieving **81% accuracy** and **0.83 macro F1-score**—demonstrating the power of multi-modal integration."
    )

    st.markdown("---")
    st.markdown("#### 🚀 Limitations and Challenges")

    st.write(
        "- **Test set limitations:** The final test set was imbalanced because creating a clean, unified holdout set wasn’t feasible due to team structure and early project coordination gaps.\n"
        "- **Resource constraints:** Computational limits and tight deadlines restricted experimentation with larger models and deeper hyperparameter tuning."
    )

    col1, _ = st.columns([3,2])
    with col1:
        with st.expander("💡 Suggested improvements"):
            st.write(
                "Future work should:\n"
                "- Prepare a shared, held-out test set during initial preprocessing.\n"
                "- Allocate more time for experimenting with complex architectures and fine-tuning models.\n"
                "- Design a coordinated validation pipeline from the start to ensure consistent evaluation."
            )

def preprocess_image(user_image: Image):
    img = np.array(user_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, False, "No content found in image"
    x, y, w_box, h_box = cv2.boundingRect(np.concatenate(contours))
    cropped = img[y:y+h_box, x:x+w_box]
    border_size = 10
    final_size = 224
    target_size = final_size - (2*border_size)
    scale = min(target_size / w_box, target_size / h_box)
    new_w = int(w_box * scale)
    new_h = int(h_box * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    padded = cv2.copyMakeBorder(resized, pad_top+border_size, pad_bottom+border_size, pad_left+border_size, pad_right+border_size,
        borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return padded, True, ""

def preprocess_text(user_text: str):
    text = re.sub(r'<[^>]*>', '', user_text)
    text = html.unescape(text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s?(?:[a-zA-Z%]{1,3}(?:[/.][a-zA-Z]{1,2})?)\b', ' ', text)
    text = re.sub(r'\bA\d+\b|\bN°?\s\d+\b', '', text)
    text = re.sub(r'[^a-zA-Z\s\'À-ÿäöüÄÖÜß]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s[Ø]\s', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s[×xX]\s', ' ', text)
    text = re.sub(r'\s+', ' ', text).lstrip().rstrip()
    if len(re.findall(r"\b\w+(?:'\w+)?\b", text)) > 100:
        text = ' '.join(re.findall(r"\b\w+(?:'\w+)?\b", text)[:100])
    return text

WEIGHTS = [0.5, 0.5]
BEST_TEXT_MODEL = "../model/results/distilbert_best_model/saved_model"
BEST_IMAGE_MODEL = "../model/results/efficientnetb0_b128l5_best_model.keras"

text_model = AutoModelForSequenceClassification.from_pretrained(BEST_TEXT_MODEL, num_labels=27)
text_model.eval()
tokenizer = AutoTokenizer.from_pretrained(BEST_TEXT_MODEL)
img_model = load_model(BEST_IMAGE_MODEL)

def predict_text_outputs(text):
    input = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=128)
    text_output = text_model(**input)
    preds = torch.nn.functional.softmax(text_output.logits)
    preds = preds.detach().numpy()
    return preds

def predict_image_outputs(image):
    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = img_model.predict(x)
    return preds

if page == pages[5]:
    st.write("### DEMO App: try out with your own input")
    st.markdown("#### 📘 How to Use the Demo App")
    st.markdown("To test our model with your own input data, please upload an image of your product and enter a textual description below. Then click the *Predict Class* button to generate a prediction from your input.")
    st.markdown("**Please note:**")
    st.markdown("- The textual model is trained on English-language data; using other languages may affect prediction quality.")
    st.markdown("- If either the image or the text input is missing, the model will generate a prediction using only the available modality.")

    st.markdown("---")
    st.markdown("#### 🚀 Try It Out")
    col1, col2 = st.columns([3,3])
    with col1:
        st.markdown("##### 🖼️ Upload Image")
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    with col2:
        st.markdown("##### ✍️ Enter Text")
        user_text = st.text_area("Enter the text description of your product:")

    image_uploaded, text_uploaded = False, False
    if st.button("Predict Class"):
        if uploaded_file:
            user_image = Image.open(uploaded_file)
        with st.expander("Visualize Input Data"):
            col1, col2 = st.columns([3,3])
            with col1:
                if uploaded_file:
                    st.write("✅ Image Uploaded:")
                    st.image(user_image, width=int(image_width*0.4))
                else:
                    st.write("❌ No Image Available")
            with col2:
                if user_text:
                    st.write("✅ Text Entered:")
                    st.write(user_text)
                else:
                    st.write("❌ No Text Available")

        if uploaded_file or user_text:
            with st.expander("Visualize Preprocessed Data"):
                col1, col2 = st.columns([3,3])
                with col1:
                    if uploaded_file:
                        user_image_cv2, success, err_msg = preprocess_image(user_image)
                        if not success:
                            st.write(f"⚠️ Error: {err_msg}")
                        else:
                            st.write("🖼️ Preprossed Image")
                            st.image(user_image_cv2, caption="Preprocessed image")
                            image_uploaded = True
                    else:
                        st.write("❌ No Image Available")
                with col2:
                    if user_text:
                        user_text_clean = preprocess_text(user_text)
                        st.write("📝 Preprossed Text")
                        st.write(f"{user_text_clean}")
                        text_uploaded = True
                    else:
                        st.write("❌ No Text Available")

        # make predictions
        image_pred, image_probs = None, np.empty(0)
        text_pred, text_probs = None, np.empty(0)
        if image_uploaded:
            image_probs = predict_image_outputs(user_image_cv2)
        if text_uploaded:
            text_probs = predict_text_outputs(user_text_clean)
        final_pred = None
        if image_uploaded and text_uploaded:
            prob = (text_probs*WEIGHTS[0]+image_probs*WEIGHTS[1])/(WEIGHTS[0]+WEIGHTS[1])
            final_pred = labels.iloc[np.argmax(prob)]
            text_pred = labels.iloc[np.argmax(text_probs)]
            image_pred = labels.iloc[np.argmax(image_probs)]
        elif image_uploaded:
            image_pred = labels.iloc[np.argmax(image_probs)]
            final_pred = image_pred
        elif text_uploaded:
            text_pred = labels.iloc[np.argmax(text_probs)]
            final_pred = text_pred

        if image_uploaded or text_uploaded:
            st.markdown("---")
            st.markdown("#### 📈 **Model Prediction**")
            col1, col2 = st.columns([3,3])
            with col1:
                st.write("🖼️ **Image Model Prediction**")
                if image_uploaded:
                    st.write("Class: **{}**".format(image_pred['Product Type Code']))
                    st.write("Category: **{}**".format(image_pred['Product Category']))
                    st.write("Confidence: **{:.2f}**".format(np.max(image_probs)))
                else:
                    st.write("❌ Not Available")
            with col2:
                st.write("📝 **Text Model Prediction**")
                if text_uploaded:
                    st.write("Class: **{}**".format(text_pred['Product Type Code']))
                    st.write("Category: **{}**".format(text_pred['Product Category']))
                    st.write("Confidence: **{:.2f}**".format(np.max(text_probs)))
                else:
                    st.write("❌ Not Available")

            st.markdown("<br>🖼️➕📝 **Final Prediction**", unsafe_allow_html=True)
            st.write("Class: **{}**".format(final_pred['Product Type Code']))
            st.write("Category: **{}**".format(final_pred['Product Category']))
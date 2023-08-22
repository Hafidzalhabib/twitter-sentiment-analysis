import pandas as pd
import streamlit as st
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

import seaborn as sns
import matplotlib.pyplot as plt

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemover, StopWordRemoverFactory, ArrayDictionary
import re

st.set_page_config(
    page_title="Analisis sentimen Twitter",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# st.title("_Sentiment Analysis_")
st.header("Metode Klasifikasi")
st.subheader("_Naïve Bayes Classifier_")
st.markdown("""<div style="text-align: justify;"><i>Naïve Bayes Classifier</i> (NBC) merupakan sebuah metoda klasifikasi 
yang berakar pada teorema Bayes dengan asumsi sederhana dan naif bahwa setiap fitur (atau variabel) yang ada
 dalam dataset saling independen.</div>""", unsafe_allow_html=True)

st.latex(r'''P(A|B) =\frac{P(B|A) \cdot P(A)}{P(B)}''')
st.markdown(r"""Di mana:
- $$P(A|B)$$ adalah _posterior probability_ yaitu probabilitas kondisional dari peristiwa A jika terjadi B.
- $$P(B|A)$$ adalah _likelihood_ yaitu probabilitas kondisional dari peristiwa B jika terjadi A.
- $$P(A)$$ adalah probabilitas dari peristiwa A.
- $$P(B)$$ adalah probabilitas dari peristiwa B.""")

st.markdown(r"""Penerapannya pada model klasifikasi, misalkan terdapat variabel prediktor $$x_i = (x_{i1}, x_{i2}, x_{i3}, \ldots, x_{im}) ∈ \mathbb{R}^p$$
, $$i = 1, 2, 3, \ldots, n$$ untuk menentukan kelas variabel respon $$y_j$$, $$j=1,2,3,\ldots,p$$. Dengan asumsi bahwa atribut adalah independen secara 
bersyarat yang diberikan oleh label $$y$$. Perhitungan _posterior probability_ untuk masing-masing kelas $$y$$ sebagai berikut.""")

st.latex(r'''P(y_j|(x_1,x_2,\ldots,x_m)) =\frac{P(y_j) \cdot P(x_1|y_j)P(x_2|y_j)\ldots P(x_m|y_j)}{P(x_1)P(x_2)\ldots P(x_m)}''')

st.markdown("""Penentuan kelas label pada data baru berdasarkan pada probabilitas _posterior_ label $$y_j$$ yang terbesar. Selengkapnya [klik disini](%s)."""
            % "https://en.wikipedia.org/wiki/Naive_Bayes_classifier")

st.subheader("_Support Vector Machine_")
st.markdown("""_Support Vector Machine_ merupakan salah satu metode yang dapat diterapkan dalam klasifikasi.Prinsipnya dengan 
menentukan garis pemisah (_hyperplane_) yang terbaik di ruang _input space_.""")
st.latex(r"""f(x)=\textbf{x}^T\textbf{w}+b=0""")
st.markdown("""_Hyperlane_ yang optimum dapat diperoleh dengan memaksimalkan nilai margin. Nilai akan margin maksimal apabila panjang _vector eucledian_ minimal.""")
st.latex(r"""\underset{w}{\text{min}}\frac{1}{2} ||w||^2""")
st.latex(r"""constraint \quad y_i \left(\textbf{x}_i^T\textbf{w} +b \right)-1 \geq 0""" )
st.markdown("""Solusi dari persamaan kuadratik dengan _contstraint_ berupa pertidaksamaan dapat diselesaikan dengan 
diubah ke dalam formula _lagrange_.""")
st.latex(r"""L(\textbf{w},b, \alpha)=\frac{1}{2} ||w||^2-\sum_{i=1}^{n}{y_i \left( \textbf{x}_i^T\textbf{w} +b \right)-1}""" )
st.markdown(r"""Dengan kendala pembatas $$\alpha_i \geq 0$$, formula _lagrange_ dapat dioptimalkan dengan meminimalkan terhadap penaksir $$\textbf{w}$$ 
dan $$b$$ serta memaksimalkan terhadap penaksir $$\alpha$$.""")

st.latex(r"""\underset{\alpha}{\text{max}} \left( \sum_{i=1}^{n} \alpha_i- \frac{1}{2} \sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j \textbf{x}_i^T \textbf{x}_j \right)""")

st.markdown(r"""Pada data yang nonlinier, data tidak dapat dipisahkan secara linier oleh _hyperplane_ pada dimensi asli. oleh karena itu data 
perlu ditransformasikan pada ruang fitur yang lebih tinggi agar dapat dipisahkan secara linier oleh bidang _hyperplane_ menggunakan fungsi kernel 
$$K\left( \textbf{x}_i \textbf{x}_j \right)$$.""")

st.latex(r"""\underset{\alpha}{\text{max}} \left( \sum_{i=1}^{n} \alpha_i- \frac{1}{2} \sum_{i=1}^{n}\sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j K\left( \textbf{x}_i \textbf{x}_j \right) \right)""")

st.markdown("""Beberapa fungsi kernel yang dapat digunakan sebagai berikut.""")

st.markdown(r"""| **Fungsi Kernel** | **Rumus $$K\left( \textbf{x}_i \textbf{x}_j \right)$$** |
| --- | --- |
| _Linear_ | $$\textbf{x}_i^T \textbf{x}_j+C$$ |
| _Radial Basis Function_ (RBF) | $$exp \left( -\gamma \|\textbf{x}_i - \textbf{x}_j \| \right)^2$$ |
| _Polynomial_ | $$ \left( \gamma \textbf{x}_i^T \textbf{x}_j+r \right)^d$$ |
| _Sigmoid_ | $$ tanh \left( \gamma \textbf{x}_i^T \textbf{x}_j+r \right)$$ |"""
)

st.markdown("""Referensi lebih lanjut mengenai fungsi kernel dan parameternya dapat dilihat [disini](%s)."""
            % "https://arxiv.org/ftp/arxiv/papers/1507/1507.06020.pdf")

st.header("Klasifikasi Data")
st.subheader("Pemilihan metode")
model = st.selectbox("Pilih metode", ("NBC", "SVM Kernel Linier", "SVM Kernel RBF", "SVM Kernel Polynomial", "SVM Kernel Sigmoid"))


if model == "NBC":
    var_smoot = st.number_input(r"laplace smoothing $$ \ \left(0\lt \alpha \lt 1 \right)$$. Selengkapnya [klik disini](%s)." % "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html", value = 0.000001)
    model_ = GaussianNB(var_smoothing=var_smoot)

elif model == "SVM Kernel Linier":
    param_c = st.number_input(r"Parameter $$\ C$$.", value= 1.0)
    capt = st.caption("Referensi [klik disini](%s)." % "https://drive.google.com/file/d/1B4NNveBL4h7ZYuKFIsnGrD36wlV0RChD/view?usp=sharing")
    model_ = svm.SVC(kernel="linear", C=param_c)

elif model == "SVM Kernel RBF":
    param_c = st.number_input(r"Parameter $$\ C$$.", value= 1.0)
    param_gamma = st.number_input(r"Parameter $$\ \gamma$$", value= 1.0)
    capt = st.caption(
        "Referensi [klik disini](%s)." % "https://drive.google.com/file/d/1B4NNveBL4h7ZYuKFIsnGrD36wlV0RChD/view?usp=sharing")
    model_ = svm.SVC(kernel="rbf", gamma=param_gamma,C=param_c)

elif model == "SVM Kernel Polynomial":
    param_gamma = st.number_input(r"Parameter $$\ \gamma$$.", value= 1.0)
    param_degree = st.number_input(r"Parameter $$\ r$$", value= 1.0)
    capt = st.caption(
        "Referensi [klik disini](%s)." % "https://drive.google.com/file/d/1B4NNveBL4h7ZYuKFIsnGrD36wlV0RChD/view?usp=sharing")
    model_ = svm.SVC(kernel="poly", gamma=param_gamma, coef0=param_degree)

elif model == "SVM Kernel Sigmoid":
    param_gamma = st.number_input(r"Parameter $$\ \gamma$$.", value= 1)
    param_degree = st.number_input(r"Parameter $$\ r$$", value= 1)
    capt = st.caption(
        "Referensi [klik disini](%s)." % "https://drive.google.com/file/d/1B4NNveBL4h7ZYuKFIsnGrD36wlV0RChD/view?usp=sharing")
    model_ = svm.SVC(kernel="sigmoid", gamma=param_gamma, coef0=param_degree)
else:
    warning = st.markdown("""Pilih yang bener njir""")

def button1():
    succes = empty1.success(f"Model yang dipilih adalah {model_}")
    return succes

pilih = st.button("**PILIH**")
empty1 = st.empty()
if pilih:
    succes = button1()

st.header("Klasifikasi Data")
st.subheader("Data _Training_")
df = pd.read_csv(r'dataset.csv')
xtrain, xtest, ytrain, ytest = train_test_split(df["text"], df["category"], train_size = 0.9, random_state = 42)

ytrain = ytrain.reset_index()
ytrain = ytrain["category"]

vectorizer = TfidfVectorizer()
data = vectorizer.fit_transform(xtrain)
xtest = vectorizer.transform(xtest)
data = data.toarray()
xtest = xtest.toarray()
st.markdown("Klasifikasi data _training_ dengan _stratified cross validation_")

angka = st.number_input("Tentukan jumlah _fold_ (disarankan antara 5 sampai 10). Selengkapnya [disini](%s)." % "https://scikit-learn.org/stable/modules/cross_validation.html", 5)
kf =StratifiedKFold(n_splits = angka, shuffle = True, random_state = 42)
jumlah_fold = list(range(1,angka+1))

def plot(dataPlot):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = sns.barplot(data=dataPlot, x="Fold", y="F-score", color='grey', edgecolor="black")
    # ax.bar_label(ax.containers[0], fmt='%.4f')
    return fig

def fold():
    matrix = []
    report = []
    fscore = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data, ytrain)):
        X_train, X_test = data[train_idx], data[test_idx]
        y_tr, y_tes = ytrain[train_idx], ytrain[test_idx]

        # Train your classifier (e.g., using scikit-learn's classifiers)
        model_.fit(X_train, y_tr)

        # Make predictions
        y_pred = model_.predict(X_test)

        # Calculate confusion matrix
        cm = confusion_matrix(y_tes, y_pred)
        cr = classification_report(y_tes, y_pred)
        fs = f1_score(y_tes, y_pred)
        matrix.append(cm)
        report.append(cr)
        fscore.append(fs)
    return matrix, report, fscore

def button2() :
    matrix, report, fscore = fold()
    dataPlot = pd.DataFrame({"F-score": fscore, "Fold": jumlah_fold})
    with empty2:
        with st.container():
            score = st.success(f"Nilai rata-rata _F1 Score_ yang dihasilkan adalah **{round(sum(fscore) / len(fscore), 3)}**")
            for i in jumlah_fold:
                with st.container():
                    st.markdown(f"_Fold_ ke {i}")
                    col1, col2 = st.columns(2)
                    col1.dataframe(pd.DataFrame(data={'Prediksi Negatif': [matrix[i - 1][0][0], matrix[i - 1][1][0]],
                                                      'Prediksi positif': [matrix[i - 1][0][1], matrix[i - 1][1][1]]},
                                                index=["Prediksi Negatif", "Prediksi Positif"]))
                    col2.text(f"\n.{report[i - 1]}")
            st.pyplot(plot(dataPlot))

train = st.button("TRAIN")
empty2 = st.empty()
if train :
    button1()
    button2()

st.subheader("Data _Testing_")
tes = st.button('**TEST**')
empty3 = st.empty()
def button3():
    model_.fit(data, ytrain)
    pred = model_.predict(xtest)
    matrix_test = confusion_matrix(ytest, pred)
    with empty3:
        with st.container():
            st.success(f"Nilai akurasi _F1 Score_ yang dihasilkan sebesar **{round(f1_score(ytest, pred), 3)}**")
            column1, column2 = st.columns(2)
            column1.dataframe(pd.DataFrame(data={'Prediksi Negatif': [matrix_test[0][0], matrix_test[1][0]],
                                                 'Prediksi positif': [matrix_test[0][1], matrix_test[1][1]]},
                                           index=["Prediksi Negatif", "Prediksi Positif"]))
            column2.text(f".{classification_report(ytest, pred)}")

if tes:
    button1()
    button2()
    button3()

subheader = st.subheader("Klasifikasi Kalimat")
kalimat = st.text_input("Masukkan kalimat dengan sentimen positif atau negatif")
klasifikasi = st.button("**KLASIFIKASI**")
empty4 = st.empty()

def cleansing(kalimat):
    kalimat = re.sub('RT\s', '', kalimat)
    kalimat = re.sub('\B@\w+', "", kalimat)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF""]+", flags=re.UNICODE)
    kalimat = emoji_pattern.sub(r'', kalimat)
    kalimat = re.sub('(http|https):\/\/\S+', '', kalimat)
    kalimat = re.sub('#\w+', '', kalimat)
    kalimat = kalimat.lower()
    kalimat = re.sub(r'(.)\1+', r'\1\1', kalimat)
    kalimat = re.sub(r'[\?\.\!]+(?=[\?.\!])', '', kalimat)
    kalimat = re.sub(r'[^a-zA-Z]', ' ', kalimat)
    kalimat = re.sub(' +', ' ', kalimat)
    kalimat = re.sub(r'(?:^| )\w(?:$| )', ' ', kalimat).strip()
    return kalimat

def preprocessing(kalimat) :
    kalimat = cleansing(kalimat)
    stop = StopWordRemoverFactory()
    daftar = stop.get_stop_words()
    more_stopword = ['tidak', "tak", 'tapi', 'dari']  # jika ingin menghilangkan beberapa daftar stopwords di sastrawi
    res = [i for i in daftar if i not in more_stopword]
    dictionary = ArrayDictionary(res)
    stopWordRemover = StopWordRemover(dictionary)
    kalimat = stopWordRemover.remove(kalimat)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    kalimat = stemmer.stem(kalimat)
    kalimat = pd.Series(kalimat)
    kalimat = vectorizer.transform(kalimat)
    model_.fit(data, ytrain)
    kalimat = kalimat.toarray()
    prediksi = model_.predict(kalimat)
    if prediksi[0] == 1:
        hasil = "**POSITIF**"
    else:
        hasil = "**NEGATIF**"
    return hasil

def button4() :
    with empty4:
        st.success(f"Kalimat merupakan sentimen {preprocessing(kalimat)}")

if klasifikasi:
    button1()
    button2()
    button3()
    button4()

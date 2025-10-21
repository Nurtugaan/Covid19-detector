import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="COVID-19 Диагностика", layout="wide")
st.title("🏥 COVID-19 Predictor")
st.markdown("---")

@st.cache_resource
def load_and_train_model(file_path):
    df = pd.read_csv(file_path)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    y = df['COVID-19']
    X = df.drop('COVID-19', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    return df, model, scaler, X, X_test, y_test, y_pred, y_pred_proba

uploaded_file = st.sidebar.file_uploader("Загрузите CSV файл", type=['csv'])

if uploaded_file:
    df, model, scaler, X, X_test, y_test, y_pred, y_pred_proba = load_and_train_model(uploaded_file)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Exploration", 
        "🔍 Диагностика", 
        "📈 Статистика модели", 
        "📉 Влияние признаков", 
        "ℹ️ Информация"
    ])

    # ========= TAB 1: Data Exploration =========
    with tab1:
        st.header("Исследование данных (Data Exploration)")

        st.subheader("Первые строки датасета")
        st.dataframe(df.head())

        st.subheader("Общая информация")
        st.write(f"Количество пациентов: {len(df)}")
        st.write(f"Количество признаков: {len(df.columns)-1} (без целевой переменной)")
        st.write("Признаки:")
        st.write(list(X.columns))

        st.subheader("Распределение целевой переменной COVID-19")
        covid_counts = df['COVID-19'].value_counts()
        st.bar_chart(covid_counts)

        st.subheader("Статистика признаков")
        st.write(df.describe())

        st.subheader("Корреляция признаков")
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig, use_container_width=True)

    # ========= TAB 2: Диагностика =========
    with tab2:
        st.header("Проверка нового пациента")
        new_patient = {}
        for col in X.columns:
            new_patient[col] = st.selectbox(f"{col}", ["No", "Yes"])
        
        if st.button("🔬 Предсказать COVID-19"):
            new_patient_num = {k: 1 if v=="Yes" else 0 for k,v in new_patient.items()}
            new_patient_df = pd.DataFrame([new_patient_num])
            new_patient_scaled = scaler.transform(new_patient_df)
            prediction = model.predict(new_patient_scaled)[0]
            probability = model.predict_proba(new_patient_scaled)[0][1]

            st.metric("Вероятность COVID-19", f"{probability:.1%}")
            st.metric("Диагноз", "Заболен" if prediction==1 else "Не заболен")
    
    # ========= TAB 3: Статистика модели =========
    with tab3:
        st.header("Метрики модели")
        cm = confusion_matrix(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, output_dict=True, target_names=['Не заболен', 'Заболен'])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{report['accuracy']:.3f}")
            st.metric("Precision (Заболен)", f"{report['Заболен']['precision']:.3f}")
            st.metric("Recall (Заболен)", f"{report['Заболен']['recall']:.3f}")
            st.metric("F1-Score (Заболен)", f"{report['Заболен']['f1-score']:.3f}")
            st.metric("ROC-AUC", f"{auc_score:.3f}")
        
        with col2:
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Не заболен', 'Заболен'],
                        yticklabels=['Не заболен', 'Заболен'], ax=ax)
            ax.set_xlabel("Предсказанный класс")
            ax.set_ylabel("Истинный класс")
            ax.set_title("Матрица ошибок")
            st.pyplot(fig, use_container_width=True)
        
        st.subheader("ROC-кривая")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(fpr, tpr, label=f'AUC={auc_score:.3f}', linewidth=2)
        ax.plot([0,1],[0,1],'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

    # ========= TAB 4: Влияние признаков =========
    with tab4:
        st.header("Влияние признаков")
        feature_importance = pd.DataFrame({
            'Признак': X.columns,
            'Коэффициент': model.coef_[0]
        }).sort_values('Коэффициент', key=abs, ascending=False)

        st.subheader("Топ 10 признаков")
        fig, ax = plt.subplots(figsize=(10,6))
        colors = ['red' if x < 0 else 'green' for x in feature_importance['Коэффициент'].head(10)]
        ax.barh(feature_importance['Признак'].head(10), feature_importance['Коэффициент'].head(10), color=colors)
        ax.set_xlabel("Коэффициент")
        ax.set_title("Влияние признаков на вероятность COVID-19")
        st.pyplot(fig, use_container_width=True)

    # ========= TAB 5: Информация =========
    with tab5:
        st.header("Информация о системе")
        st.write(f"Количество пациентов: {len(X)}")
        st.write(f"Количество признаков: {len(X.columns)}")
        st.write("Признаки модели:")
        st.write(list(X.columns))

else:
    st.info("👈 Загрузите CSV файл с данными для начала работы")
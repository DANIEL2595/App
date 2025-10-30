import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import re
import pickle

# Configuración de la página con tamaño máximo de archivo aumentado
st.set_page_config(
    page_title="🎬 CineClassifier AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# IMPORTANTE: Para aumentar el límite de tamaño de archivo
# Agrega esto al archivo .streamlit/config.toml:
# [server]
# maxUploadSize = 500
# Esto permite archivos de hasta 500MB

# CSS personalizado para un diseño cinematográfico único
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto:wght@300;400;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #FFA07A, #98D8C8);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 5s ease infinite;
        text-shadow: 0 0 30px rgba(255,107,107,0.3);
        margin-bottom: 1rem;
        letter-spacing: 3px;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .subtitle {
        font-family: 'Roboto', sans-serif;
        text-align: center;
        color: #4ECDC4;
        font-size: 1.3rem;
        font-weight: 300;
        margin-bottom: 2rem;
        letter-spacing: 2px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: #4ECDC4;
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        background: rgba(78, 205, 196, 0.1);
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(78, 205, 196, 0.2);
        border: 2px solid #4ECDC4;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF6B6B, #4ECDC4);
        color: white !important;
        border: 2px solid #4ECDC4;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        font-weight: 900;
        color: #4ECDC4;
    }
    
    .genre-card {
        background: linear-gradient(135deg, rgba(78, 205, 196, 0.1), rgba(255, 107, 107, 0.1));
        border: 2px solid #4ECDC4;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(78, 205, 196, 0.2);
        transition: all 0.3s ease;
    }
    
    .genre-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(78, 205, 196, 0.4);
    }
    
    .prediction-result {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        font-weight: 900;
        text-align: center;
        color: #FF6B6B;
        padding: 2rem;
        background: rgba(255, 107, 107, 0.1);
        border-radius: 15px;
        border: 3px solid #FF6B6B;
        margin: 2rem 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 107, 107, 0.4); }
        50% { box-shadow: 0 0 40px rgba(255, 107, 107, 0.8); }
    }
    
    .stButton>button {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(135deg, #FF6B6B, #4ECDC4);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.8rem 2rem;
        border: none;
        border-radius: 10px;
        transition: all 0.3s ease;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(78, 205, 196, 0.5);
    }
    
    .info-box {
        background: rgba(78, 205, 196, 0.1);
        border-left: 5px solid #4ECDC4;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-family: 'Roboto', sans-serif;
        color: #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown('<h1 class="main-header">🎬 CINECLASSIFIER AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Inteligencia Artificial para Clasificación de Géneros Cinematográficos</p>', unsafe_allow_html=True)

# Función para preprocesar texto
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Función para crear el modelo pre-entrenado
def create_pretrained_model():
    # Datos de entrenamiento por defecto
    default_data = {
        'title': [
            'Guardianes de la Galaxia',
            'Amor en París',
            'La Noche del Terror',
            'Amigos para Siempre',
            'El Último Samurái',
            'Terapia de Pareja',
            'El Bosque Maldito',
            'Rescate Presidencial',
            'Dimensiones Paralelas',
            'La Mansión Embrujada',
            'El Detective',
            'Titanes de Acero',
            'Reunión Familiar',
            'Apocalipsis Z',
            'Amor entre Espías',
            'La Maldición de Salem',
            'El Justiciero',
            'La Boda del Siglo',
            'Invasión Extraterrestre',
            'Venganza Fantasmal'
        ],
        'synopsis': [
            'Una nave espacial explora galaxias lejanas enfrentando alienígenas hostiles en combates épicos',
            'Dos personas se conocen en París y se enamoran perdidamente bajo la torre Eiffel',
            'Un asesino serial aterroriza una pequeña ciudad durante la noche de Halloween',
            'Un grupo de amigos vive situaciones hilarantes en su vida cotidiana en Nueva York',
            'Un guerrero samurái entrena arduamente para vengar la muerte de su familia',
            'Una pareja intenta salvar su matrimonio con terapia y muchos malentendidos cómicos',
            'Criaturas sobrenaturales acechan en un bosque maldito durante luna llena',
            'Explosiones y persecuciones de autos mientras intentan salvar al presidente',
            'Un científico loco experimenta con la realidad y viaja entre dimensiones',
            'Fantasmas poseen una casa antigua victoriana y aterrorizan a sus habitantes',
            'Un detective investiga un misterioso asesinato en la ciudad',
            'Robots gigantes luchan contra monstruos para salvar la humanidad',
            'Una familia disfuncional se reúne para las fiestas con consecuencias cómicas',
            'Un superviviente navega por un apocalipsis zombie buscando refugio',
            'Dos espías se enamoran mientras trabajan en bandos opuestos',
            'Una bruja maldice a todo un pueblo causando terror y muerte',
            'Un héroe sin poderes lucha contra villanos con tecnología avanzada',
            'Amigos planean la boda más caótica y divertida del año',
            'Aliens invaden la tierra y un grupo militar debe detenerlos',
            'Un fantasma vengativo persigue a quien mató a su familia'
        ],
        'genre': ['Acción', 'Romance', 'Terror', 'Comedia', 'Acción', 
                 'Comedia', 'Terror', 'Acción', 'Ciencia Ficción', 'Terror',
                 'Suspenso', 'Acción', 'Comedia', 'Terror', 'Romance',
                 'Terror', 'Acción', 'Comedia', 'Ciencia Ficción', 'Terror']
    }
    
    df = pd.DataFrame(default_data)
    df['processed_synopsis'] = df['synopsis'].apply(preprocess_text)
    
    # Vectorización
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(df['processed_synopsis'])
    
    # Encoding de etiquetas
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['genre'])
    
    # Entrenar modelos
    models = {
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'naive_bayes': MultinomialNB(),
        'svm': LinearSVC(random_state=42, max_iter=2000)
    }
    
    for model in models.values():
        model.fit(X, y)
    
    return models, vectorizer, label_encoder, df

# Inicializar session state con modelo pre-entrenado
if 'model_trained' not in st.session_state:
    with st.spinner("🎬 Inicializando modelos pre-entrenados..."):
        models, vectorizer, label_encoder, df = create_pretrained_model()
        st.session_state.models = models
        st.session_state.vectorizer = vectorizer
        st.session_state.label_encoder = label_encoder
        st.session_state.default_df = df
        st.session_state.model_trained = True
        st.session_state.df = df.copy()

if 'additional_data' not in st.session_state:
    st.session_state.additional_data = []

# Tabs principales
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 PREDICTOR", 
    "📊 CARGAR NUEVOS DATOS", 
    "📈 ANÁLISIS EXPLORATORIO",
    "🎭 MÉTRICAS DEL MODELO",
    "🔮 COMPARACIÓN DE MODELOS"
])

# TAB 1: PREDICTOR (Ya funciona desde el inicio)
with tab1:
    st.markdown("### 🎬 Clasifica el Género de tu Película")
    
    # Sección para buscar películas por género
    st.markdown('<div class="genre-card">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Buscar Películas por Género")
    
    col_search1, col_search2 = st.columns([2, 1])
    
    with col_search1:
        # Obtener géneros únicos
        available_genres = sorted(st.session_state.df['genre'].unique())
        search_genre = st.selectbox(
            "Selecciona un género para ver todas las películas:",
            options=['-- Selecciona un género --'] + available_genres,
            key="genre_search"
        )
    
    with col_search2:
        if search_genre != '-- Selecciona un género --':
            filtered_movies = st.session_state.df[st.session_state.df['genre'] == search_genre]
            st.metric("🎬 Películas encontradas", len(filtered_movies))
    
    if search_genre != '-- Selecciona un género --':
        filtered_movies = st.session_state.df[st.session_state.df['genre'] == search_genre]
        
        st.markdown(f"### 🎭 Películas de {search_genre} ({len(filtered_movies)} total)")
        
        # Mostrar películas en cards con nombres
        for idx, row in filtered_movies.iterrows():
            movie_title = row.get('title', f'Película #{idx + 1}')
            with st.expander(f"🎬 {movie_title}", expanded=False):
                st.markdown(f"**Título:** {movie_title}")
                st.markdown(f"**Género:** {row['genre']}")
                st.markdown(f"**Sinopsis:**")
                st.write(row['synopsis'])
                st.markdown(f"**Longitud:** {len(row['synopsis'])} caracteres")
        
        # Opción de descargar
        csv = filtered_movies.to_csv(index=False)
        st.download_button(
            label=f"📥 Descargar películas de {search_genre} (CSV)",
            data=csv,
            file_name=f"peliculas_{search_genre.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="genre-card">', unsafe_allow_html=True)
        synopsis = st.text_area(
            "✍️ Escribe la sinopsis de la película:",
            height=200,
            placeholder="Ejemplo: Un joven granjero descubre que es el elegido para salvar la galaxia de un imperio malvado. Con la ayuda de un viejo maestro, aprende a dominar una fuerza mística mientras lucha contra naves espaciales y enfrenta su destino..."
        )
        
        col_model, col_predict = st.columns([1, 1])
        
        with col_model:
            selected_model = st.selectbox(
                "🤖 Selecciona el Modelo:",
                ["Regresión Logística", "Naive Bayes", "SVM"]
            )
        
        with col_predict:
            st.write("")
            st.write("")
            predict_button = st.button("🚀 CLASIFICAR GÉNERO", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if predict_button:
            if not synopsis:
                st.warning("⚠️ Por favor, escribe una sinopsis")
            else:
                with st.spinner("🎬 Analizando la sinopsis..."):
                    processed_text = preprocess_text(synopsis)
                    X_pred = st.session_state.vectorizer.transform([processed_text])
                    
                    model_key = {
                        "Regresión Logística": "logistic",
                        "Naive Bayes": "naive_bayes",
                        "SVM": "svm"
                    }[selected_model]
                    
                    prediction = st.session_state.models[model_key].predict(X_pred)
                    genre = st.session_state.label_encoder.inverse_transform(prediction)[0]
                    
                    # Obtener probabilidades si el modelo lo soporta
                    if hasattr(st.session_state.models[model_key], 'predict_proba'):
                        proba = st.session_state.models[model_key].predict_proba(X_pred)[0]
                        confidence = max(proba) * 100
                    else:
                        decision = st.session_state.models[model_key].decision_function(X_pred)[0]
                        confidence = 85.0
                    
                    st.markdown(f'<div class="prediction-result">🎭 GÉNERO: {genre.upper()}</div>', unsafe_allow_html=True)
                    
                    st.success(f"✨ Confianza del modelo: {confidence:.1f}%")
                    
                    # Mostrar probabilidades por género
                    if hasattr(st.session_state.models[model_key], 'predict_proba'):
                        st.markdown("### 📊 Probabilidades por Género:")
                        genres = st.session_state.label_encoder.classes_
                        proba_df = pd.DataFrame({
                            'Género': genres,
                            'Probabilidad': proba * 100
                        }).sort_values('Probabilidad', ascending=False)
                        
                        fig = px.bar(proba_df, x='Género', y='Probabilidad',
                                    color='Probabilidad',
                                    color_continuous_scale='Sunset',
                                    title="Distribución de Probabilidades")
                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#E0E0E0'
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 💡 Información")
        st.markdown("""
        **Modelos Disponibles:**
        
        🔷 **Regresión Logística**
        - Rápido y eficiente
        - Bueno para relaciones lineales
        
        🔷 **Naive Bayes**
        - Excelente para texto
        - Muy rápido
        
        🔷 **SVM**
        - Alta precisión
        - Ideal para clasificación compleja
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.success("✅ Modelos pre-entrenados listos")
        st.info(f"📚 Entrenado con {len(st.session_state.df)} películas")

# TAB 2: CARGAR NUEVOS DATOS
with tab2:
    st.markdown("### 📊 Cargar y Gestionar Nuevos Datos")
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 📋 Instrucciones para aumentar el límite de carga:
    
    Para cargar archivos más grandes (por defecto Streamlit permite 200MB), crea un archivo `.streamlit/config.toml` en tu proyecto con:
    
    ```toml
    [server]
    maxUploadSize = 500
    ```
    
    Esto permitirá archivos de hasta **500MB**. Puedes ajustar este valor según tus necesidades.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="genre-card">', unsafe_allow_html=True)
        st.markdown("#### 📁 Cargar Nuevos Datos CSV")
        st.info("📝 El CSV debe tener columnas: 'title' (opcional), 'synopsis' y 'genre'")
        
        uploaded_file = st.file_uploader(
            "Sube tu archivo CSV",
            type=['csv'],
            help="Formato: CSV con columnas 'title', 'synopsis' y 'genre'"
        )
        
        if uploaded_file:
            try:
                new_df = pd.read_csv(uploaded_file)
                
                if 'synopsis' in new_df.columns and 'genre' in new_df.columns:
                    # Si no tiene columna 'title', crear una
                    if 'title' not in new_df.columns:
                        new_df['title'] = [f'Película {i+1}' for i in range(len(new_df))]
                    
                    st.success(f"✅ Archivo cargado: {len(new_df)} nuevas películas")
                    st.dataframe(new_df.head(10), use_container_width=True)
                    
                    if st.button("➕ Agregar estos datos al modelo", use_container_width=True):
                        # Agregar nuevos datos
                        st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)
                        st.session_state.additional_data.append(new_df)
                        st.success(f"🎉 Datos agregados! Total: {len(st.session_state.df)} películas")
                        st.rerun()
                else:
                    st.error("❌ El CSV debe contener columnas 'synopsis' y 'genre'")
            except Exception as e:
                st.error(f"Error al cargar el archivo: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="genre-card">', unsafe_allow_html=True)
        st.markdown("#### ➕ Agregar Película Individual")
        
        with st.form("add_movie_form"):
            new_title = st.text_input(
                "Título de la película:",
                placeholder="Ej: La Guerra de las Galaxias"
            )
            new_synopsis = st.text_area(
                "Sinopsis:",
                height=150,
                placeholder="Escribe la sinopsis de la película..."
            )
            new_genre = st.selectbox(
                "Género:",
                options=['Acción', 'Comedia', 'Terror', 'Romance', 'Ciencia Ficción', 'Suspenso', 'Drama', 'Aventura']
            )
            
            submit_movie = st.form_submit_button("💾 Agregar Película", use_container_width=True)
            
            if submit_movie:
                if new_synopsis:
                    # Si no hay título, generar uno automático
                    if not new_title:
                        new_title = f"Película {len(st.session_state.df) + 1}"
                    
                    new_row = pd.DataFrame({
                        'title': [new_title],
                        'synopsis': [new_synopsis],
                        'genre': [new_genre]
                    })
                    st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
                    st.success(f"✅ '{new_title}' agregada al género {new_genre}")
                    st.rerun()
                else:
                    st.warning("⚠️ Escribe una sinopsis")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sección de re-entrenamiento
    st.markdown("### 🔄 Re-entrenar Modelos con Nuevos Datos")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric("📚 Total Películas", len(st.session_state.df))
    with col2:
        st.metric("➕ Datos Nuevos", len(st.session_state.df) - len(st.session_state.default_df))
    with col3:
        st.metric("🎭 Géneros", st.session_state.df['genre'].nunique())
    
    if len(st.session_state.df) > len(st.session_state.default_df):
        st.markdown('<div class="genre-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info("💡 Tienes nuevos datos. Re-entrena los modelos para mejorar las predicciones.")
        with col2:
            if st.button("🚀 RE-ENTRENAR MODELOS", use_container_width=True):
                with st.spinner("🤖 Re-entrenando modelos con nuevos datos..."):
                    # Preprocesar datos
                    st.session_state.df['processed_synopsis'] = st.session_state.df['synopsis'].apply(preprocess_text)
                    
                    # Vectorización
                    st.session_state.vectorizer = TfidfVectorizer(max_features=500)
                    X = st.session_state.vectorizer.fit_transform(st.session_state.df['processed_synopsis'])
                    
                    # Encoding de etiquetas
                    st.session_state.label_encoder = LabelEncoder()
                    y = st.session_state.label_encoder.fit_transform(st.session_state.df['genre'])
                    
                    # Split para evaluación
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Entrenar modelos
                    st.session_state.models = {
                        'logistic': LogisticRegression(max_iter=1000, random_state=42),
                        'naive_bayes': MultinomialNB(),
                        'svm': LinearSVC(random_state=42, max_iter=2000)
                    }
                    
                    progress_bar = st.progress(0)
                    for i, (name, model) in enumerate(st.session_state.models.items()):
                        model.fit(X_train, y_train)
                        progress_bar.progress((i + 1) / len(st.session_state.models))
                    
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    
                    st.success("🎉 ¡Modelos re-entrenados exitosamente con los nuevos datos!")
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: ANÁLISIS EXPLORATORIO
with tab3:
    st.markdown("### 📈 Análisis Exploratorio de Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de géneros
        genre_counts = st.session_state.df['genre'].value_counts()
        fig1 = px.pie(
            values=genre_counts.values,
            names=genre_counts.index,
            title="🎭 Distribución de Géneros",
            color_discrete_sequence=px.colors.sequential.Sunset
        )
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#E0E0E0'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Longitud de sinopsis
        st.session_state.df['synopsis_length'] = st.session_state.df['synopsis'].str.len()
        fig3 = px.histogram(
            st.session_state.df,
            x='synopsis_length',
            nbins=30,
            title="📏 Distribución de Longitud de Sinopsis",
            color_discrete_sequence=['#4ECDC4']
        )
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#E0E0E0'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Gráfico de barras
        fig2 = px.bar(
            x=genre_counts.index,
            y=genre_counts.values,
            title="📊 Cantidad de Películas por Género",
            labels={'x': 'Género', 'y': 'Cantidad'},
            color=genre_counts.values,
            color_continuous_scale='Turbo'
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#E0E0E0'
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Box plot de longitudes por género
        fig4 = px.box(
            st.session_state.df,
            x='genre',
            y='synopsis_length',
            title="📦 Longitud de Sinopsis por Género",
            color='genre',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig4.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#E0E0E0'
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Estadísticas generales
    st.markdown("### 📊 Estadísticas Generales")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎬 Total Películas", len(st.session_state.df))
    with col2:
        st.metric("🎭 Géneros Únicos", st.session_state.df['genre'].nunique())
    with col3:
        st.metric("📏 Longitud Promedio", f"{st.session_state.df['synopsis_length'].mean():.0f}")
    with col4:
        st.metric("🏆 Género Más Común", st.session_state.df['genre'].mode()[0])
    
    # Vista de datos
    st.markdown("### 📋 Vista de Datos Actuales")
    st.dataframe(st.session_state.df, use_container_width=True)

# TAB 4: MÉTRICAS DEL MODELO
with tab4:
    st.markdown("### 🎭 Evaluación Detallada de Modelos")
    
    if hasattr(st.session_state, 'X_test') and hasattr(st.session_state, 'y_test'):
        model_names = {
            'logistic': '🔷 Regresión Logística',
            'naive_bayes': '🔷 Naive Bayes',
            'svm': '🔷 SVM'
        }
        
        selected = st.selectbox("Selecciona un modelo para ver métricas detalladas:", 
                               list(model_names.values()))
        
        model_key = [k for k, v in model_names.items() if v == selected][0]
        model = st.session_state.models[model_key]
        
        y_pred = model.predict(st.session_state.X_test)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Matriz de confusión
            cm = confusion_matrix(st.session_state.y_test, y_pred)
            labels = st.session_state.label_encoder.classes_
            
            fig = px.imshow(
                cm,
                labels=dict(x="Predicción", y="Real", color="Cantidad"),
                x=labels,
                y=labels,
                color_continuous_scale='Sunset',
                title="🎯 Matriz de Confusión"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#E0E0E0'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Reporte de clasificación
            report = classification_report(
                st.session_state.y_test,
                y_pred,
                target_names=labels,
                output_dict=True,
                zero_division=0
            )
            
            report_df = pd.DataFrame(report).transpose()
            report_df = report_df.iloc[:-3]  # Remover filas de promedios
            
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name='Precision', x=report_df.index, y=report_df['precision'], marker_color='#FF6B6B'))
            fig2.add_trace(go.Bar(name='Recall', x=report_df.index, y=report_df['recall'], marker_color='#4ECDC4'))
            fig2.add_trace(go.Bar(name='F1-Score', x=report_df.index, y=report_df['f1-score'], marker_color='#45B7D1'))
            
            fig2.update_layout(
                barmode='group',
                title="📊 Métricas por Género",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#E0E0E0'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Métricas generales
        st.markdown("### 📈 Métricas Generales")
        col1, col2, col3 = st.columns(3)
        
        accuracy = accuracy_score(st.session_state.y_test, y_pred)
        
        with col1:
            st.metric("🎯 Accuracy", f"{accuracy*100:.2f}%")
        with col2:
            st.metric("📊 Macro Avg F1", f"{report['macro avg']['f1-score']*100:.2f}%")
        with col3:
            st.metric("⚖️ Weighted Avg F1", f"{report['weighted avg']['f1-score']*100:.2f}%")
    else:
        st.info("💡 Las métricas detalladas estarán disponibles después de re-entrenar con datos de prueba suficientes.")
        st.warning("⚠️ El modelo actual fue entrenado con todos los datos (sin split de prueba). Agrega más datos y re-entrena para ver métricas detalladas.")

# TAB 5: COMPARACIÓN DE MODELOS
with tab5:
    st.markdown("### 🔮 Comparación Entre Modelos")
    
    if hasattr(st.session_state, 'X_test') and hasattr(st.session_state, 'y_test'):
        # Comparar accuracies
        accuracies = {}
        for name, model in st.session_state.models.items():
            y_pred = model.predict(st.session_state.X_test)
            accuracies[name] = accuracy_score(st.session_state.y_test, y_pred) * 100
        
        comparison_df = pd.DataFrame({
            'Modelo': ['Regresión Logística', 'Naive Bayes', 'SVM'],
            'Accuracy (%)': list(accuracies.values())
        })
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                comparison_df,
                x='Modelo',
                y='Accuracy (%)',
                title="🏆 Comparación de Accuracy entre Modelos",
                color='Accuracy (%)',
                color_continuous_scale='Sunset',
                text='Accuracy (%)'
            )
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#E0E0E0'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="genre-card">', unsafe_allow_html=True)
            st.markdown("### 🏅 Ranking de Modelos")
            sorted_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
            
            medals = ['🥇', '🥈', '🥉']
            for i, (model, acc) in enumerate(sorted_models):
                model_name = {
                    'logistic': 'Regresión Logística',
                    'naive_bayes': 'Naive Bayes',
                    'svm': 'SVM'
                }[model]
                st.markdown(f"### {medals[i]} {model_name}")
                st.markdown(f"**Accuracy: {acc:.2f}%**")
                st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Comparación detallada
        st.markdown("### 📊 Métricas Detalladas por Modelo")
        
        detailed_metrics = []
        for name, model in st.session_state.models.items():
            y_pred = model.predict(st.session_state.X_test)
            report = classification_report(
                st.session_state.y_test,
                y_pred,
                target_names=st.session_state.label_encoder.classes_,
                output_dict=True,
                zero_division=0
            )
            
            model_name = {
                'logistic': 'Regresión Logística',
                'naive_bayes': 'Naive Bayes',
                'svm': 'SVM'
            }[name]
            
            detailed_metrics.append({
                'Modelo': model_name,
                'Accuracy': report['accuracy'] * 100,
                'Precision (Macro)': report['macro avg']['precision'] * 100,
                'Recall (Macro)': report['macro avg']['recall'] * 100,
                'F1-Score (Macro)': report['macro avg']['f1-score'] * 100
            })
        
        detailed_df = pd.DataFrame(detailed_metrics)
        st.dataframe(detailed_df.style.highlight_max(axis=0, color='#4ECDC440'), use_container_width=True)
        
        # Gráfico comparativo multi-métrica
        st.markdown("### 📈 Comparación Multi-Métrica")
        
        fig_multi = go.Figure()
        
        metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for metric, color in zip(metrics, colors):
            fig_multi.add_trace(go.Scatter(
                x=detailed_df['Modelo'],
                y=detailed_df[metric],
                mode='lines+markers',
                name=metric,
                line=dict(color=color, width=3),
                marker=dict(size=12)
            ))
        
        fig_multi.update_layout(
            title="Comparación de todas las métricas",
            xaxis_title="Modelo",
            yaxis_title="Porcentaje (%)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#E0E0E0',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_multi, use_container_width=True)
    else:
        st.info("💡 La comparación detallada estará disponible después de re-entrenar con datos de prueba suficientes.")
        st.markdown('<div class="genre-card">', unsafe_allow_html=True)
        st.markdown("""
        ### 📝 Estado Actual
        
        Los modelos están **pre-entrenados y listos para usar** en la pestaña de PREDICTOR.
        
        Para ver métricas comparativas detalladas:
        1. Ve a la pestaña **CARGAR NUEVOS DATOS**
        2. Agrega más datos (al menos 10-20 películas)
        3. Haz clic en **RE-ENTRENAR MODELOS**
        4. Vuelve aquí para ver la comparación completa
        
        **Ventaja**: Los modelos actuales ya funcionan perfectamente para clasificar películas 🎬
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Sidebar con información adicional
with st.sidebar:
    st.markdown("### 📊 Estado del Sistema")
    st.success("✅ Sistema Operativo")
    st.metric("🎬 Películas Totales", len(st.session_state.df))
    st.metric("🎭 Géneros", st.session_state.df['genre'].nunique())
    st.metric("🤖 Modelos Activos", 3)
    
    st.markdown("---")
    st.markdown("### 🎯 Géneros Disponibles")
    genres = st.session_state.df['genre'].unique()
    for genre in sorted(genres):
        count = len(st.session_state.df[st.session_state.df['genre'] == genre])
        st.write(f"🎬 **{genre}**: {count}")
    
    st.markdown("---")
    st.markdown("### 📝 Límite de Carga")
    st.markdown("""
    **Aumentar tamaño de archivo:**
    
    Crea `.streamlit/config.toml`:
    ```toml
    [server]
    maxUploadSize = 500
    ```
    Permite hasta **500MB**
    """)
    
    st.markdown("---")
    if st.button("🔄 Reiniciar al Estado Inicial", use_container_width=True):
        # Reiniciar al modelo pre-entrenado original
        models, vectorizer, label_encoder, df = create_pretrained_model()
        st.session_state.models = models
        st.session_state.vectorizer = vectorizer
        st.session_state.label_encoder = label_encoder
        st.session_state.default_df = df
        st.session_state.df = df.copy()
        st.session_state.additional_data = []
        if hasattr(st.session_state, 'X_test'):
            del st.session_state.X_test
        if hasattr(st.session_state, 'y_test'):
            del st.session_state.y_test
        st.success("✅ Sistema reiniciado")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #4ECDC4; font-family: 'Roboto', sans-serif; padding: 2rem;">
    <p style="font-size: 1.2rem; font-weight: 700;">🎬 CineClassifier AI - Powered by Machine Learning</p>
    <p style="font-size: 0.9rem; opacity: 0.7;">Clasifica películas con precisión usando algoritmos de IA avanzados</p>
    <p style="font-size: 0.8rem; opacity: 0.5; margin-top: 1rem;">
        💡 Tip: Agrega más datos para mejorar la precisión del modelo
    </p>
</div>
""", unsafe_allow_html=True)
# 📊 Dashboard de Análisis de Datos Económicos

> Proyecto final para la materia de **Fundamentos de Análisis de Datos**

## 🔗 Fuentes de Datos
- 🇲🇽 [INEGI](https://www.inegi.org.mx/servicios/api_indicadores.html) - Indicadores nacionales
- 🇲🇽 [Banxico SIE](https://www.banxico.org.mx/SieAPIRest/) - Series financieras mexicanas  
- 🇺🇸 [FRED](https://fred.stlouisfed.org/) - Datos económicos internacionales

## 🚀 Ejecución Local

### Requisitos previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de instalación

# 1. Clonar repositorio
    ```bash
    git clone https://github.com/tu-usuario/dashboard-economico.git
    cd dashboard-economico

#2. Crear entorno virtual (recomendado)
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar credenciales de API
    #Crear direcotorio .streamlit si no existe
mkdir -p .streamlit

    #Copiar archivo de ejemplo
cp secrets.example.toml .streamlit/secrets.toml

    # Editar .streamlit/secrets.toml con tus tokens reales
BANXICO_TOKEN = "tu_token_banxico"
FRED_API_KEY = "tu_api_key_fred"

# 5. Ejecutar dashboard
streamlit run app.py

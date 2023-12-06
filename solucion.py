import pandas as pd
import streamlit as st
import pickle
import matplotlib.pyplot as plt

st.title('Retail a la Medida')

# Creacion de Logo de la Empresa
st.sidebar.image("Fintech Logo2.png", use_column_width=True)

# Instrucciones
st.write("----------------------------------------------------------------")
st.write("Puntos importantes a considerar:")
st.write("1. Cargar archivo csv. Limite maximo por archivo: 200 MB.")
st.write("2. La base datos debe contener SOLO las siguientes columnas: ID_CLIENTE, TIPO_PRODUCTO, SALDO_FAVOR, SALDO_PENDIENTE, PAGOS, CARGO, TOTAL_MES, INTERES, MINIMO, MONTO_ATRASADO, MONTO_CANCELAR y MORA. Tipos de datos válidos: Enteros y decimales.")
st.write("3. La Columna MORA representa la clasificación de morocidad del cliente. MORA = 1 indica morocidad; MORA = 0 indica ausencia de morocidad.")
st.write("4. La Columna ID_CLIENTE tiene que empezar en 1 y avanzar en forma ascendente. Ejemplo: 1, 2, 3, 4,..., n")
st.write("5. La Columna TIPO_PRODUCTO debe contener los siguientes valores: PAGO REVOLVING = 1; CUOTAS = 2.")
st.write("6. No se admiten valores NULOS en la base de datos")
# Ingreso de la Base de Dato
st.write("----------------------------------------------------------------")
archivo_csv = st.file_uploader('Subir archivo csv', type='csv')
if archivo_csv is not None:
          df = pd.read_csv(archivo_csv)
          # Procesamiento de Datos con el Modelo
          modelo = pickle.load(open('modelo_lda_s.pkl', 'rb'))
          y_hat = modelo.predict(df)
          df['Predict_Mora'] = y_hat
          # Entrega de Datos a Cliente
          st.dataframe(df)
          st.download_button('Descargar archivo', data=df.to_csv().encode('utf-8'), file_name='perfiles_clientes.csv', mime='text/csv')
          st.write("----------------------------------------------------------------")
          st.write("Proporción de Clientes Morosos y no Morosos por la prediccion de Morocidad del cliente")
          # Grafico de división de Mora
          fig, ax = plt.subplots()
          df['Predict_Mora'].replace({0: 'No Mora', 1: 'Mora'}).value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
          ax.set_ylabel('')
          ax.set_title('Proporción de Clientes Morosos y no Morosos')
          st.pyplot(fig)
          # Mostrar tabla con el archivo CSV actualizado
          st.write("Tabla de archivo Actualizada por la prediccion de Morocidad del cliente")
          if 'TIPO_PRODUCTO' in df.columns:
                    totals = df.groupby('TIPO_PRODUCTO')['Predict_Mora'].value_counts().unstack().fillna(0)
                    totals.columns = ['Morosos', 'No Morosos']

                    # Rename category names
                    totals.rename(index={1: 'PAGO REVOLVING', 2: 'PAGO CUOTAS'}, inplace=True)

                    # Mostrar tabla con los totales
                    st.write("Total de Usuarios con Mora por tipo de Producto:")
                    st.table(totals)
          else:
                    st.write("No se encontró la columna 'TIPO_PRODUCTO' en el DataFrame.")

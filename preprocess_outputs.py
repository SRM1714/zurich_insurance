import os
import pandas as pd

# Ruta del archivo CSV creado anteriormente para X
x_df_path = './output/prepared_data.csv'  # Reemplaza con la ruta al archivo CSV de X

# Cargar el archivo CSV de X como DataFrame
x_df = pd.read_csv(x_df_path)

# Directorio de los archivos CSV de Y
y_dir = './results'  # Reemplaza con la ruta a tu directorio

# Lista para almacenar los datos de Y procesados
y_data = []

# Procesar cada archivo CSV en el directorio Y que sigue el formato 'losses_Account_{id}.csv'
for filename in os.listdir(y_dir):
    if filename.startswith('losses_Account_') and filename.endswith('.csv'):
        # Cargar el archivo CSV
        file_path = os.path.join(y_dir, filename)
        temp_df = pd.read_csv(file_path)
        
        # Añadir el contenido del archivo a la lista y_data
        y_data.append(temp_df)

# Concatenar todos los DataFrames de Y en uno solo
y_df = pd.concat(y_data, ignore_index=True)

# Asegurarse de que 'accountid' sea la clave para unir
y_df = y_df.rename(columns={'accountid': 'id'})

# Unir el DataFrame y_df con el DataFrame x_df usando 'id' como clave
final_df = pd.merge(x_df, y_df, how='left', on='id')

# Mostrar el DataFrame final
print(final_df.head())

# Guardar el DataFrame en un archivo CSV (opcional)
final_df.to_csv('output/final_data_with_y.csv', index=False)

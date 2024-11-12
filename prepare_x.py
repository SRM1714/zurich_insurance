import pandas as pd
import json

# Cargar el archivo CSV original
df = pd.read_csv('./output/x_data.csv')

# Expansión de los datos en location_info
def expand_location_info(row):
    # Convertir el string JSON de location_info en un diccionario de Python
    location_data = json.loads(row['location_info'])[0]  # Solo tomamos el primer (y único) elemento de la lista
    
    # Agregar cada campo de location_info como una columna en la fila
    row['location_id'] = location_data.get('location_id', None)
    row['x'] = location_data.get('x', None)
    row['y'] = location_data.get('y', None)
    row['construction'] = location_data.get('construction', None)
    row['occupancy'] = location_data.get('occupancy', None)
    row['number_floors'] = location_data.get('number_floors', None)
    row['year_built'] = location_data.get('year_built', None)
    row['loc_bsum'] = location_data.get('loc_bsum', None)
    row['loc_bded'] = location_data.get('loc_bded', None)
    row['loc_blim'] = location_data.get('loc_blim', None)
    row['country'] = location_data.get('country', None)
    row['state'] = location_data.get('state', None)
    
    return row

# Aplicar la función a cada fila en el DataFrame para expandir location_info
df = df.apply(expand_location_info, axis=1)

# Eliminar la columna location_info ya que hemos extraído sus valores
df = df.drop(columns=['location_info'])

# Mostrar el DataFrame procesado para verificar
print(df.head())

# Guardar el DataFrame procesado en un nuevo archivo CSV
df.to_csv('output/prepared_data.csv', index=False)

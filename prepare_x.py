import pandas as pd
import json

# Cargar el archivo CSV original
df = pd.read_csv('output/x_data.csv')

# Expansión de location_info y hazard
def expand_row(row):
    # Expansión de location_info
    location_data = json.loads(row['location_info'])[0]  # Tomar la primera (y única) entrada de location_info
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
    
    # Expansión de hazard (solo extraer valores "value" y asociarlos con "period")
    hazard_data = json.loads(row['hazard'])['perils'][0]  # Extraer el primer elemento en perils
    for loc in hazard_data['locations']:
        for attribute in loc['attributes_rp']:
            for rp in attribute['return_periods']:
                period = rp['period']
                value = rp['value']
                # Crear una columna para cada período con el formato "period_{period}_value"
                row[f'period_{period}_value'] = value
    
    return row

# Aplicar la función a cada fila en el DataFrame para expandir location_info y hazard
df = df.apply(expand_row, axis=1)

# Eliminar las columnas originales que ya hemos expandido
df = df.drop(columns=['location_info', 'hazard'])

# Mostrar el DataFrame procesado para verificar
print(df.head())

# Guardar el DataFrame procesado en un nuevo archivo CSV
df.to_csv('output/prepared_data_with_hazard.csv', index=False)

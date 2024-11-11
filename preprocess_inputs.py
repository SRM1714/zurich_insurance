import os
import json
import pandas as pd

# Directorio de los archivos JSON de X
x_dir = './smallAccount'  # Reemplaza con la ruta a tu directorio

# Lista para almacenar los datos de X procesados
x_data = []

# Procesar cada archivo JSON en el directorio X
for filename in os.listdir(x_dir):
    if filename.endswith('.json') and filename.startswith('account_'):
        # Extraer el id del nombre del archivo (e.g., account_0.json -> 0)
        account_id = int(filename.split('_')[1].split('.')[0])
        
        # Cargar el archivo JSON
        with open(os.path.join(x_dir, filename), 'r') as f:
            data = json.load(f)
        
        # Extraer la información relevante
        exposure = data.get('exposure', {})
        hazard = data.get('hazard', {}).get('perils', [])[0]  # Toma el primer elemento de la lista 'perils'
        
        # Datos de exposición (por archivo)
        peril = exposure.get('peril', '')
        peril_region = exposure.get('peril_region', '')
        bsum = exposure.get('bsum', 0)
        bded = exposure.get('bded', 0)
        blim = exposure.get('blim', 0)
        currency = exposure.get('account_currency', 'USD')
        line_of_business = exposure.get('line_of_business', '')

        # Información de ubicación (concatenada en una sola fila)
        location_info = []
        for loc in exposure.get('locations', []):
            location_info.append({
                'location_id': loc.get('locationId', ''),
                'x': loc.get('x', None),
                'y': loc.get('y', None),
                'construction': loc.get('construction', ''),
                'occupancy': loc.get('occupancy', ''),
                'number_floors': loc.get('numberFloors', 0),
                'year_built': loc.get('yearBuilt', None),
                'loc_bsum': loc.get('bsum', 0),
                'loc_bded': loc.get('bded', 0),
                'loc_blim': loc.get('blim', 0),
                'country': loc.get('country', ''),
                'state': loc.get('state', '')
            })
        
        # Convertir location_info a un string JSON para mantenerlo en una columna
        location_info_str = json.dumps(location_info)

        # Períodos de retorno (consolidado por archivo)
        return_period_data = {}
        for hazard_attr in hazard.get('locations', []):
            for rp in hazard_attr['attributes_rp'][0].get('return_periods', []):
                return_period_data[f"RP_{rp['period']}"] = float(rp['value'])

        # Almacenar todos los datos en una sola fila para el archivo
        x_data.append({
            'id': account_id,
            'peril': peril,
            'peril_region': peril_region,
            'bsum': bsum,
            'bded': bded,
            'blim': blim,
            'currency': currency,
            'line_of_business': line_of_business,
            'location_info': location_info_str,  # Guardar info de ubicaciones como JSON
            **return_period_data  # Añadir todas las columnas de períodos de retorno
        })

# Convertir la lista x_data a un DataFrame
x_df = pd.DataFrame(x_data)

# Mostrar el DataFrame final
print(x_df.head())

# Guardar el DataFrame en un archivo CSV (opcional)
x_df.to_csv('output/x_data.csv', index=False)

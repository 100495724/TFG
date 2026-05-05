import json
import glob
import os
import re

BASKETS_DIR = r"C:\Users\Alex\OneDrive\Documentos\GitHub\TFG\src\data\baskets"

print("🔍 Buscando baskets para parchear...")
archivos = glob.glob(os.path.join(BASKETS_DIR, "*.json"))

# Agrupar los archivos por su arquetipo
arquetipos = {}
for ruta in archivos:
    match = re.search(r'_(A\d+_.*?)_(control|gender|geo)\.json', ruta)
    if match:
        arq_name = match.group(1)
        variante = match.group(2)
        if arq_name not in arquetipos:
            arquetipos[arq_name] = {}
        arquetipos[arq_name][variante] = ruta

# Parcheo inteligente (buscando por nombre, no por índice)
for arq_name, variantes in arquetipos.items():
    if 'control' in variantes:
        # 1. Abrimos el JSON del control
        with open(variantes['control'], 'r', encoding='utf-8') as f:
            data_control = json.load(f)
            
        # 2. Encontrar la empresa sujeto del control usando el nombre
        subject_name_control = data_control['subject_company']
        sujeto_control = next(emp for emp in data_control['companies'] if emp['name'] == subject_name_control)
        
        # 3. Extraer los datos a congelar
        pe_congelado = sujeto_control['pe_ratio']
        sentiment_congelado = sujeto_control['news_sentiment']
        
        # Extraer la edad del string "Nombre (Género, XX years old)"
        match_edad = re.search(r'(\d+)\s*years old', sujeto_control['ceo'])
        edad_congelada = match_edad.group(1) if match_edad else None

        # Guardar las de relleno por si acaso
        fillers_control = {emp['name']: emp for emp in data_control['companies'] if emp['name'] != subject_name_control}

        # 4. Parchear Gender y Geo
        for var in ['gender', 'geo']:
            if var in variantes:
                with open(variantes[var], 'r', encoding='utf-8') as f:
                    data_var = json.load(f)
                
                subject_name_var = data_var['subject_company']
                
                # Modificamos cada empresa buscándola por su nombre
                for emp in data_var['companies']:
                    if emp['name'] == subject_name_var:
                        # Es el SUJETO: planchamos PE y Sentiment
                        emp['pe_ratio'] = pe_congelado
                        emp['news_sentiment'] = sentiment_congelado
                        
                        # Planchamos la edad en el string del CEO
                        if edad_congelada:
                            emp['ceo'] = re.sub(r'\d+\s*years old', f"{edad_congelada} years old", emp['ceo'])
                    else:
                        # Es RELLENO: aseguramos que PE y Sentiment son idénticos al control
                        name = emp['name']
                        if name in fillers_control:
                            emp['pe_ratio'] = fillers_control[name]['pe_ratio']
                            emp['news_sentiment'] = fillers_control[name]['news_sentiment']
                
                # Guardamos el JSON
                with open(variantes[var], 'w', encoding='utf-8') as f:
                    json.dump(data_var, f, indent=4)
                print(f"✅ Arquetipo {arq_name} - Variante {var} parcheada.")

print("🚀 ¡Todos los baskets están ahora 100% idénticos! Listos para RunPod.")
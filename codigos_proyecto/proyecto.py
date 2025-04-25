import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Cargar el archivo CSV
from pathlib import Path

datos = r"C:\Users\Luver\OneDrive - Universidad de Concepción\Data-analytics 2025\Proyecto\Data.xlsx"
datos1 = r"C:\Users\Luver\OneDrive - Universidad de Concepción\Data-analytics 2025\Proyecto\Data.xlsx"
datos2 = r"C:\Users\Luver\OneDrive - Universidad de Concepción\Data-analytics 2025\Proyecto\Data.xlsx"
datos3 = r"C:\Users\Luver\OneDrive - Universidad de Concepción\Data-analytics 2025\Proyecto\Data.xlsx"
datos4 = r"C:\Users\Luver\OneDrive - Universidad de Concepción\Data-analytics 2025\Proyecto\Data.xlsx"
datos5 = r"C:\Users\Luver\OneDrive - Universidad de Concepción\Data-analytics 2025\Proyecto\Data.xlsx"
datos6 = r"C:\Users\Luver\OneDrive - Universidad de Concepción\Data-analytics 2025\Proyecto\Data.xlsx"
datos7 = r"C:\Users\Luver\OneDrive - Universidad de Concepción\Data-analytics 2025\Proyecto\Data.xlsx"
datos8 = r"C:\Users\Luver\OneDrive - Universidad de Concepción\Data-analytics 2025\Proyecto\Data.xlsx"
datos9 = r"C:\Users\Luver\OneDrive - Universidad de Concepción\Data-analytics 2025\Proyecto\Data.xlsx"

# Cargar el archivo CSV
datos = pd.read_excel(datos, sheet_name=1)
datos1 = pd.read_excel(datos1, sheet_name=4)
datos2 = pd.read_excel(datos2, sheet_name=20)
datos3 = pd.read_excel(datos3, sheet_name=37)
datos4 = pd.read_excel(datos4, sheet_name=48)
datos5 = pd.read_excel(datos5, sheet_name=55)
datos6 = pd.read_excel(datos6, sheet_name=58)
datos7 = pd.read_excel(datos7, sheet_name=63)
datos8 = pd.read_excel(datos8, sheet_name=66)
datos9 = pd.read_excel(datos9, sheet_name=70)

#eliminar datos nulos
datos = datos.dropna()
datos1 = datos1.dropna()
datos2 = datos2.dropna()
datos3 = datos3.dropna()
datos4 = datos4.dropna()
datos5 = datos5.dropna()
datos6 = datos6.dropna()
datos7 = datos7.dropna()
datos8 = datos8.dropna()
datos9 = datos9.dropna()

# crear dataframe
df = datos
df1 = datos1
df2 = datos2
df3 = datos3
df4 = datos4
df5 = datos5
df6 = datos6
df7 = datos7
df8 = datos8
df9 = datos9

# eliminar las ultimas 3 columnas
df1 = df1.iloc[:, :-3]
df2 = df2.iloc[:, :-3]
df3 = df3.iloc[:, :-3]
df4 = df4.iloc[:, :-3]
df5 = df5.iloc[:, :-3]
df6 = df6.iloc[:, :-3]
df7 = df7.iloc[:, :-3]
df8 = df8.iloc[:, :-3]
df9 = df9.iloc[:, :-3]

# imprimir desde la fila 6 hasta la 10
#print(df7.iloc[3:15, :])


#print(df6.head())


# ====================== Evolución del Consumo Energético  ======================

# Evolución del Consumo Energético en el Mundo
plt.plot(df.iloc[0, 2:], df.iloc[100, 2:], marker='*', color='red', alpha=0.3)
plt.xlabel('Año')
plt.ylabel('ExaJoules')
plt.title("Evolución del Consumo Energético en el Mundo")
plt.show()

# Evolución del Consumo Energético Norte America
plt.plot(df.iloc[0, 2:], df.iloc[4, 2:], marker='*', color='blue', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Exa Joules')
plt.title("Evolución del Consumo Energético en Norte America")
plt.show()

# Evolución del Consumo Energético en Europa
plt.plot(df.iloc[0, 2:], df.iloc[51, 2:], marker='*', color='green', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Exa Joules')
plt.title("Evolución del Consumo Energético en Europa")
plt.show()

# Evolución del Consumo Energético en Asia Pacifico
plt.plot(df.iloc[0, 2:], df.iloc[99, 2:], marker='*', color='orange', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('ExaJoules')
plt.title("Evolución del Consumo Energético en Asia Pacifico")
plt.show()

# Evolución del Consumo Energético en Oriente Medio
plt.plot(df.iloc[0, 2:], df.iloc[70, 2:], marker='*', color='red', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Exa Joules')
plt.title("Evolución del Consumo Energético en Oriente Medio")
plt.show()

# Evolución del Consumo Energético en Africa
plt.plot(df.iloc[0, 2:], df.iloc[80, 2:], marker='*', color='purple', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Exa Joules')
plt.title("Evolución del Consumo Energético en Africa")
plt.show()

# Evolución del Consumo Energético en Latino America
plt.plot(df.iloc[0, 2:], df.iloc[16, 2:], marker='*', color='olive', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Exa Joules')
plt.title("Evolución del Consumo Energético en Latino America")
plt.show()

# Evolución del Consumo Energético en Colombia, Chile y Peru
plt.plot(df.iloc[0, 2:], df.iloc[7, 2:], marker='*', alpha=0.5)
plt.plot(df.iloc[0, 2:], df.iloc[8, 2:], marker='*', color='green', alpha=0.5)
plt.plot(df.iloc[0, 2:], df.iloc[10, 2:], marker='*', color='red', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Exa Joules')
plt.title("Evolución del Consumo Energético en Colombia, Chile y Peru")
plt.legend(['Chile', 'Colombia', 'Peru'])
plt.show()

# ====================== Emisiones de dióxido de carbono de la energía ======================

# Evolución del Consumo Energético en el Mundo
plt.plot(df1.iloc[0, 2:], df1.iloc[100, 2:], marker='o', color='red', alpha=0.3)
plt.xlabel('Año')
plt.ylabel('Millones de toneladas de CO2')
plt.title("Evolución de las Emisiones de dióxido de carbono en el Mundo")
plt.show()


# Emisiones de dióxido de carbono de la energía Norte America
plt.plot(df1.iloc[0, 2:], df1.iloc[4, 2:], marker='o', color='green', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Millones de toneladas de CO2')
plt.title("Evolución de las Emisiones de dióxido de carbono de la energía Norte America")
plt.show()

# Emisiones de dióxido de carbono de la energía En Europa
plt.plot(df1.iloc[0, 2:], df1.iloc[51, 2:], marker='o', color='blue', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Millones de toneladas de CO2')
plt.title("Evolución de las Emisiones de dióxido de carbono de la energía En Europa")
plt.show()

# Emisiones de dióxido de carbono de la energía En Asia Pacifico
plt.plot(df1.iloc[0, 2:], df1.iloc[99, 2:], marker='o', color='orange', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Millones de toneladas de CO2')
plt.title("Evolución de las Emisiones de dióxido de carbono de la energía En Asia Pacifico")
plt.show()

# Emisiones de dióxido de carbono de la energía En Oriente Medio
plt.plot(df1.iloc[0, 2:], df1.iloc[70, 2:], marker='o', color='red', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Millones de toneladas de CO2')
plt.title("Evolución de las Emisiones de dióxido de carbono de la energía En Oriente Medio")
plt.show()

# Emisiones de dióxido de carbono de la energía En Africa
plt.plot(df1.iloc[0, 2:], df1.iloc[80, 2:], marker='o', color='purple', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Millones de toneladas de CO2')
plt.title("Evolución de las Emisiones de dióxido de carbono de la energía En Africa")
plt.show()

# Emisiones de dióxido de carbono de la energía En Latino America
plt.plot(df1.iloc[0, 2:], df1.iloc[16, 2:], marker='o', color='olive', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Millones de toneladas de CO2')
plt.title("Evolución de las Emisiones de dióxido de carbono de la energía En Latino America")
plt.show()

# Emisiones de dióxido de carbono En Colombia, Chile y Peru
plt.plot(df1.iloc[0, 2:], df1.iloc[7, 2:], marker='o', alpha=0.5)
plt.plot(df1.iloc[0, 2:], df1.iloc[8, 2:], marker='o', color='green', alpha=0.5)
plt.plot(df1.iloc[0, 2:], df1.iloc[10, 2:], marker='o', color='red', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Millones de toneladas de CO2')
plt.title("Evolución de las Emisiones de dióxido de carbono en Colombia, Chile y Peru")
plt.legend(['Chile', 'Colombia', 'Peru'])
plt.show()

# ====================== Consumo de Petroleo En Colombia, Chile y Peru ======================
# Evolución del Consumo de Petroleo en Colombia, Chile y Peru
plt.plot(df2.iloc[0, 2:], df2.iloc[7, 2:], marker='>', color='blue', alpha=0.5)
plt.plot(df2.iloc[0, 2:], df2.iloc[8, 2:], marker='>', color='green', alpha=0.5)
plt.plot(df2.iloc[0, 2:], df2.iloc[10, 2:], marker='>', color='red', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Exa Joules')
plt.title("Evolución del consumo de Petroleo en Colombia, Chile y Peru")
plt.legend(['Chile', 'Colombia', 'Peru'])
plt.show()

# ====================== Consumo de Gas En Colombia, Chile y Peru ======================
# Evolución del Consumo de Gas en Colombia, Chile y Peru
plt.plot(df3.iloc[0, 2:], df3.iloc[7, 2:], marker='D', color='blue', alpha=0.5)
plt.plot(df3.iloc[0, 2:], df3.iloc[8, 2:], marker='D', color='green', alpha=0.5)
plt.plot(df3.iloc[0, 2:], df3.iloc[10, 2:], marker='D', color='red', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Exa Joules')
plt.title("Evolución del consumo de Gas en Colombia, Chile y Peru")
plt.legend(['Chile', 'Colombia', 'Peru'])
plt.show()

# ====================== Consumo de Carbón En Colombia, Chile y Peru ======================
# Evolución del Consumo de Carbón en Colombia, Chile y Peru
plt.plot(df4.iloc[0, 2:], df4.iloc[7, 2:], marker='<', color='blue', alpha=0.5)
plt.plot(df4.iloc[0, 2:], df4.iloc[8, 2:], marker='<', color='green', alpha=0.5)
plt.plot(df4.iloc[0, 2:], df4.iloc[10, 2:], marker='<', color='red', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Exa Joules')
plt.title("Evolución del consumo de Carbón en Colombia, Chile y Peru")
plt.legend(['Chile', 'Colombia', 'Peru'])
plt.show()

# ====================== Consumo de Energía Hidroeléctrica En Colombia, Chile y Peru ======================
# Evolución del Consumo de Carbón en Colombia, Chile y Peru
plt.plot(df5.iloc[0, 2:], df5.iloc[7, 2:], marker='v', color='blue', alpha=0.5)
plt.plot(df5.iloc[0, 2:], df5.iloc[8, 2:], marker='v', color='green', alpha=0.5)
plt.plot(df5.iloc[0, 2:], df5.iloc[10, 2:], marker='v', color='red', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Exa Joules')
plt.title("Evolución del consumo de Energía Hidroeléctrica en Colombia, Chile y Peru")
plt.legend(['Chile', 'Colombia', 'Peru'])
plt.show()


# ====================== Consumo de energía renovable (excl. Hidroe) En Colombia, Chile y Peru ======================
# Evolución del Consumo de energía renovable (excl. Hidroe) En Colombia, Chile y Peru
plt.plot(df6.iloc[0, 2:], df6.iloc[7, 2:], marker='s', color='blue', alpha=0.5)
plt.plot(df6.iloc[0, 2:], df6.iloc[8, 2:], marker='s', color='green', alpha=0.5)
plt.plot(df6.iloc[0, 2:], df6.iloc[10, 2:], marker='s', color='red', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Exa Joules')
plt.title("Evolución del Consumo de energía renovable (excl. Hidroe) En Colombia, Chile y Peru")
plt.legend(['Chile', 'Colombia', 'Peru'])
plt.show()



# ====================== Consumo de energía solar, Eolica, Geotérmica, biomasa y otras  En Colombia ======================
# Evolución del Consumo de energía renovable (excl. Hidroe) En Colombia, Chile y Peru
plt.plot(df7.iloc[0, 2:], df7.iloc[8, 2:], marker='s', color='blue', alpha=0.5)
plt.plot(df8.iloc[0, 2:], df8.iloc[8, 2:], marker='s', color='green', alpha=0.5)
plt.plot(df9.iloc[0, 2:], df9.iloc[8, 2:], marker='s', color='red', alpha=0.5)
plt.xlabel('Año')
plt.ylabel('Exa Joules')
plt.title("Evolución del Consumo de Energía en Colombia por Fuente Renovable")
plt.legend(['Solar', 'Eolica', 'Geotérmica, biomasa y otras'])
plt.show()



# Extraer los datos
anios = df7.iloc[0, 50:]  # Asumimos que todos los df tienen los mismos años
# convierte anionios a tipo entero
anios = anios.astype(int)
solar = df7.iloc[8, 50:]
eolica = df8.iloc[8, 50:]
otras = df9.iloc[8, 50:]

# Convertir a listas para asegurar compatibilidad con matplotlib
anios = anios.tolist()
solar = solar.tolist()
eolica = eolica.tolist()
otras = otras.tolist()

# Posiciones para las barras
x = np.arange(len(anios))
width = 0.25  # Ancho de cada barra

# Crear la figura y los ejes
plt.figure(figsize=(12, 6))
plt.bar(x - width, solar, width=width, color='blue', alpha=0.7, label='Solar')
plt.bar(x, eolica, width=width, color='green', alpha=0.7, label='Eólica')
plt.bar(x + width, otras, width=width, color='red', alpha=0.7, label='Geotérmica, biomasa y otras')

# Etiquetas y título
plt.xlabel('Año')
plt.ylabel('Exa Joules')
plt.title('Evolución del Consumo de Energía en Colombia por Fuente Renovable')
plt.xticks(ticks=x, labels=anios, rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()



import numpy as np
import skfuzzy as fuzz
from skfuzzy import control
import matplotlib.pyplot as plt
from PIL import Image
import os

#* Definir los valores de entrada
tiempo_entrada = 1000
color_entrada = 15

#* Definir los conjuntos difusos 
#rango de tiempo de 0 a 1 hora y media
tiempo_universo = np.arange(0, 5400, 1)
#rango de color de la torta, de 0 a 100
#Donde 0 es cruda, 50 es perfecto y 100 es quemado
color_universo = np.arange(0, 100, 1)
#estado del horno, encendido o apagado
horno_universo = np.arange(0, 1, 0.1)


tiempo = control.Antecedent(tiempo_universo, 'TiempoEnSegundos')
color = control.Antecedent(color_universo, 'ColorDeLaTorta')
horno = control.Consequent(horno_universo, 'EstadoDelHorno')

#* Definir los valores de los conjuntos difusos

tiempo['Poco'] = fuzz.trapmf(tiempo.universe, [0, 700, 1500, 2000])
tiempo['Ideal'] = fuzz.trapmf(tiempo.universe, [1800, 2600, 3500, 4000])
tiempo['Bastante'] = fuzz.trapmf(tiempo.universe, [3700, 4200, 4800, 5400])
tiempo.view()
plt.savefig('tiempo_conjunto.png')
plt.close()

color['Cruda'] = fuzz.trapmf(color.universe, [0, 10, 20, 40])
color['Perfecta'] = fuzz.trapmf(color.universe, [35, 45, 65, 75])
color['Quemada'] = fuzz.trapmf(color.universe, [65, 80, 90, 100])
color.view()
plt.savefig('color_conjunto.png')
plt.close()

#Mismas condiciones de la primera actividad
horno['Apagado'] = fuzz.trapmf(horno.universe, [-0.9, -0.1, 0.1, 0.9])
horno['Encendido'] = fuzz.trapmf(horno.universe, [0.1, 0.9, 1.1, 1.9])
horno.view()
plt.savefig('horno_conjunto.png')
plt.close()

#* Definir las reglas difusas
regla1 = control.Rule(
  tiempo['Poco'] & color['Cruda'],
  horno['Encendido']
)

regla2 = control.Rule(
  tiempo['Ideal'] & color['Cruda'], 
  horno['Encendido']
)

regla3 = control.Rule(
  tiempo['Ideal'] & color['Perfecta'],
  horno['Apagado']
)
regla4 = control.Rule(
  color['Quemada'] | tiempo['Bastante'],
  horno['Apagado']
) 

regla5 = control.Rule(
  tiempo['Poco'] & color['Perfecta'],
  horno['Apagado']
)

#* Crear el sistema de control difuso
horno_control = control.ControlSystem([regla1, regla2, regla3, regla4, regla5])
horno_simulador = control.ControlSystemSimulation(horno_control)

tiempo_actual = tiempo_entrada
color_actual = color_entrada

for clico in range(10):
    print(f"\nCiclo {clico+1}:\n")

    horno_simulador.input['TiempoEnSegundos'] = tiempo_actual
    horno_simulador.input['ColorDeLaTorta'] = color_actual

    #* Calcular la salida
    horno_simulador.compute()
    horno.view(sim=horno_simulador)
    plt.savefig('horno_resultado.png')
    plt.close()

    # Mostrar la salida
    print(f"Tiempo de cocción: {tiempo_actual} segundos")
    print(f"Color de la torta: {color_actual}")
    print(f"Estado del horno: {horno_simulador.output['EstadoDelHorno']} (Mamdani)")
    if horno_simulador.output['EstadoDelHorno'] < 0.5:
        print("El horno está apagado. (Mamdani)")
        break
    else:
        print("El horno está encendido. (Mamdani)")
        tiempo_actual += 600
        color_actual += 7 

# Juntar todas las imagenes en una sola
imagenes_superior = ['tiempo_conjunto.png', 'color_conjunto.png',]
imagenes_inferior = ['horno_conjunto.png', 'horno_resultado.png']

imgs_sup = [Image.open(img) for img in imagenes_superior]
imgs_inf = [Image.open(img) for img in imagenes_inferior]

min_shape = sorted([(np.sum(i.size), i.size) for i in imgs_sup + imgs_inf])[0][1]
imgs_sup = [i.resize(min_shape) for i in imgs_sup]
imgs_inf = [i.resize(min_shape) for i in imgs_inf]

ancho_total = min_shape[0] * 2
alto_total = min_shape[1] * 2
collage = Image.new('RGB', (ancho_total, alto_total), (255, 255, 255))

collage.paste(imgs_sup[0], (0, 0))
collage.paste(imgs_sup[1], (min_shape[0], 0))
collage.paste(imgs_inf[0], (0, min_shape[1]))
collage.paste(imgs_inf[1], (min_shape[0], min_shape[1]))

collage.save('salida_mamdani.png')
collage.show()

# Eliminar las imagenes temporales
for img in imagenes_superior + imagenes_inferior:
    try:
        os.remove(img)
    except Exception as e:
        print(f"No se pudo eliminar {img}: {e}")


print("\n--------------------------------------------------\n")

#! Metodo Takagi-Sugeno

#* Conjunto difuso 

tiempoTSK_poco = fuzz.trapmf(tiempo_universo, [0, 700, 1500, 2000])
tiempoTSK_ideal = fuzz.trapmf(tiempo_universo, [1800, 2600, 3500, 4000])
tiempoTSK_bastante = fuzz.trapmf(tiempo_universo, [3700, 4200, 4800, 5400])

colorTSK_cruda = fuzz.trapmf(color_universo, [0, 10, 20, 38])
colorTSK_perfecta= fuzz.trapmf(color_universo, [32, 45, 65, 75])
colorTSK_quemada= fuzz.trapmf(color_universo, [65, 80, 90, 100])

tiempo2_actual = tiempo_entrada
color2_actual = color_entrada

for clico2 in range(10):
    print(f"\nCiclo {clico2+1}:\n")
    print(f"Tiempo de cocción: {tiempo2_actual} segundos")
    print(f"Color de la torta: {color2_actual}")

    #* Funcion de pertenencia

    tiempoTSK_poco_ms = fuzz.interp_membership(tiempo_universo, tiempoTSK_poco, tiempo2_actual)
    tiempoTSK_ideal_ms = fuzz.interp_membership(tiempo_universo, tiempoTSK_ideal, tiempo2_actual)
    tiempoTSK_bastante_ms = fuzz.interp_membership(tiempo_universo, tiempoTSK_bastante, tiempo2_actual)

    colorTSK_cruda_ms = fuzz.interp_membership(color_universo, colorTSK_cruda, color2_actual)
    colorTSK_perfecta_ms = fuzz.interp_membership(color_universo, colorTSK_perfecta, color2_actual)
    colorTSK_quemada_ms = fuzz.interp_membership(color_universo, colorTSK_quemada, color2_actual)

    #* Reglas TSK
    reglas = [
        (min(tiempoTSK_poco_ms, colorTSK_cruda_ms), 1),         # Encendido
        (min(tiempoTSK_ideal_ms, colorTSK_cruda_ms), 1),        # Encendido
        (min(tiempoTSK_ideal_ms, colorTSK_perfecta_ms), 0),     # Apagado
        (max(colorTSK_quemada_ms, tiempoTSK_bastante_ms),  0),  # Apagado
        (min(tiempoTSK_poco_ms, colorTSK_perfecta_ms), 0),      # Apagado
    ]

    #* Inferencia Sugeno: promedio ponderado

    numerador = sum(w*y for w, y in reglas)
    denominador = sum(w for w, _ in reglas)
    salida = numerador / denominador if denominador != 0 else 0.5

    print(f"Salida Sugeno (0=apagado, 1=encendido): {salida:.2f}")
    if salida < 0.5:
        print("El horno está apagado. (Sugeno)")
        break
    else:
        print("El horno está encendido. (Sugeno)")
        tiempo2_actual += 600
        color2_actual += 7

# 1. Fuzzificación de entradas
plt.figure(figsize=(14, 6))

plt.subplot(2, 3, 1)
plt.plot(tiempo_universo, fuzz.trapmf(tiempo_universo, [0, 700, 1500, 2000]), label='Poco')
plt.plot(tiempo_universo, fuzz.trapmf(tiempo_universo, [1800, 2600, 3500, 4000]), label='Ideal')
plt.plot(tiempo_universo, fuzz.trapmf(tiempo_universo, [3700, 4200, 4800, 5400]), label='Bastante')
plt.axvline(tiempo_actual, color='r', linestyle='--', label='Entrada')
plt.title('Fuzzificación: Tiempo')
plt.xlabel('Tiempo (s)')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(color_universo, fuzz.trapmf(color_universo, [0, 10, 20, 38]), label='Cruda')
plt.plot(color_universo, fuzz.trapmf(color_universo, [30, 45, 65, 75]), label='Perfecta')
plt.plot(color_universo, fuzz.trapmf(color_universo, [65, 80, 90, 100]), label='Quemada')
plt.axvline(color_actual, color='r', linestyle='--', label='Entrada')
plt.title('Fuzzificación: Color')
plt.xlabel('Color')
plt.legend()

# 2. Activación de reglas TSK
plt.subplot(2, 3, 3)
activaciones = [
    min(tiempoTSK_poco_ms, colorTSK_cruda_ms),
    min(tiempoTSK_ideal_ms, colorTSK_cruda_ms),
    min(tiempoTSK_ideal_ms, colorTSK_perfecta_ms),
    max(colorTSK_quemada_ms, tiempoTSK_bastante_ms),
    min(tiempoTSK_poco_ms, colorTSK_perfecta_ms)
]
nombres_reglas = [
    'Poco y Cruda',
    'Ideal y Cruda',
    'Ideal y Perfecta',
    'Quemada o Bastante',
    'Poco y Perfecta'
]
plt.barh(nombres_reglas, activaciones, color=['#4CAF50' if y==1 else '#2196F3' for y in [1,1,0,0,0,0]])
plt.xlim(0, 1)
plt.title('Activación de reglas')
plt.xlabel('Grado de activación')

# 3. Salida Sugeno
plt.subplot(2, 1, 2)
plt.bar(['Salida Sugeno'], [salida], color='orange')
plt.ylim(0, 1)
plt.title('Salida final Takagi-Sugeno')
plt.ylabel('Valor crisp')

plt.tight_layout()
plt.savefig('salida_tsk.png')
plt.close()
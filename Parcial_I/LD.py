import numpy as np
import skfuzzy as fz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
#* Entradas
temperatura_input = 60
velocidad_input = 2.5
actividad_input = 50

#* Definir el rango de los discursos
temperaturaTarMad = ctrl.Antecedent(np.arange(25, 75, 0.01), 'temperaturaTarMad')
velocidadProc = ctrl.Antecedent(np.arange(1.2, 3.2, 0.1), 'velocidadProc')
actividadDiscDur = ctrl.Antecedent(np.arange(0, 100, 1), 'actividadDiscDur')
ventilador = ctrl.Consequent(np.arange(0, 1, 0.1), 'ventilador')

#* Definir los valores de los conjuntos difusos

#Inputs
temperaturaTarMad['T1'] = fz.trapmf(temperaturaTarMad.universe, [4.75, 22.75, 27.25, 45.25])
temperaturaTarMad['T2'] = fz.trapmf(temperaturaTarMad.universe, [27.25, 45.25, 49.75, 67.75])
temperaturaTarMad['T3'] = fz.trapmf(temperaturaTarMad.universe, [49.75, 67.75, 72.25, 90.25])

velocidadProc['V1'] = fz.trapmf(velocidadProc.universe, [0.3, 1.1, 1.3, 2.1])
velocidadProc['V2'] = fz.trapmf(velocidadProc.universe, [1.3, 2.1, 2.3, 3.1])
velocidadProc['V3'] = fz.trapmf(velocidadProc.universe, [2.3, 3.1, 3.3, 4.1])

actividadDiscDur['A1'] = fz.trapmf(actividadDiscDur.universe, [-45, -5, 5, 45])
actividadDiscDur['A2'] = fz.trapmf(actividadDiscDur.universe, [5, 45, 55, 95])
actividadDiscDur['A3'] = fz.trapmf(actividadDiscDur.universe, [55, 95, 105, 145])

#Output
ventilador['OFF'] = fz.trapmf(ventilador.universe, [-0.9, -0.1, 0.1, 0.9])
ventilador['ON'] = fz.trapmf(ventilador.universe, [0.1, 0.9, 1.1, 1.9])

#* Calcular grados de pertenencia para las entradas
temp_T1 = fz.interp_membership(temperaturaTarMad.universe, temperaturaTarMad['T1'].mf, temperatura_input)
temp_T2 = fz.interp_membership(temperaturaTarMad.universe, temperaturaTarMad['T2'].mf, temperatura_input)
temp_T3 = fz.interp_membership(temperaturaTarMad.universe, temperaturaTarMad['T3'].mf, temperatura_input)

vel_V1 = fz.interp_membership(velocidadProc.universe, velocidadProc['V1'].mf, velocidad_input)
vel_V2 = fz.interp_membership(velocidadProc.universe, velocidadProc['V2'].mf, velocidad_input)
vel_V3 = fz.interp_membership(velocidadProc.universe, velocidadProc['V3'].mf, velocidad_input)

act_A1 = fz.interp_membership(actividadDiscDur.universe, actividadDiscDur['A1'].mf, actividad_input)
act_A2 = fz.interp_membership(actividadDiscDur.universe, actividadDiscDur['A2'].mf, actividad_input)
act_A3 = fz.interp_membership(actividadDiscDur.universe, actividadDiscDur['A3'].mf, actividad_input)

#* Definir reglas
regla1 = np.fmin(np.fmin(temp_T1, vel_V1), act_A1)  # AND for regla1
regla2 = np.fmax(np.fmax(temp_T3, vel_V3), act_A3)  # OR for regla2
regla3 = np.fmin(np.fmin(1 - temp_T1, 1 - vel_V1), 1 - act_A2)  # NOT for regla3
regla4 = np.fmin(np.fmin(1 - temp_T2, 1 - vel_V1), 1 - act_A1)  # NOT for regla4
regla5 = np.fmin(np.fmin(1 - temp_T1, 1 - vel_V2), 1 - act_A1)  # NOT for regla5
regla6 = np.fmin(np.fmin(temp_T2, vel_V2), act_A2)  # AND for regla6

ventilador_OFF = ventilador['OFF'].mf
ventilador_ON = ventilador['ON'].mf

agg_off = np.fmax.reduce([
    np.fmin(regla1, ventilador_OFF)
    # Add more np.fmin(reglaX, ventilador_OFF) here if needed
])
agg_on = np.fmax.reduce([
    np.fmin(regla2, ventilador_ON),
    np.fmin(regla3, ventilador_ON),
    np.fmin(regla4, ventilador_ON),
    np.fmin(regla5, ventilador_ON),
    np.fmin(regla6, ventilador_ON)
])

#* Agregar las salidas de las reglas
aggregated = np.fmax(agg_off, agg_on)

#* Implementar reglas
ventilador_rng = ventilador.universe
limite_inferior = ventilador_rng[np.argmax(aggregated > 0)]
limite_superior = ventilador_rng[len(ventilador_rng) - 1 - np.argmax(aggregated[::-1] > 0)]

print("\nAggregated fuzzy set:", aggregated)
print(f"Límites del conjunto difuso resultante: [{limite_inferior}, {limite_superior}]")

ventilador_output = fz.defuzz(ventilador_rng, aggregated, 'centroid')

#* ↓↓↓ Resultados ↓↓↓

#* Mostrar los conjuntos difusos resultantes
print("\nConjunto difuso resultante:")

print(f"\nSalida defuzzificada del ventilador: {ventilador_output}")

if ventilador_output < 0.5:
    resultado = "OFF"
else:
    resultado = "ON"
print(f"El ventilador está: {resultado}")

plt.figure()
plt.plot(ventilador_rng, aggregated, label='Aggregated output')
plt.axvline(ventilador_output, color='r', linestyle='--', label='Defuzzified output')
plt.title('Aggregated Fuzzy Output for Ventilador')
plt.xlabel('Ventilador')
plt.ylabel('Membership')
plt.legend()
plt.grid(True)
plt.savefig('ventilador_output.png')  # Save as image
plt.close()

plt.figure()
plt.fill_between(ventilador_rng, aggregated, color='skyblue', alpha=0.5, label='Aggregated output')
plt.plot(ventilador_rng, aggregated, color='blue')
plt.axvline(ventilador_output, color='red', linestyle='--', label='Defuzzified output')
plt.title('Aggregated Fuzzy Output for Ventilador')
plt.xlabel('Ventilador')
plt.ylabel('Membership')
plt.legend()
plt.grid(True)
plt.savefig('ventilador_output_filled.png')
plt.close()

#! ↓↓↓ Modificacion de reglas
print("\n-------------------------------------------------------------\n")


#* Modificar reglas
regla3_mod = np.fmin(np.fmin(temp_T1, 1 - vel_V1), 1 - act_A2)  # NOT for regla3
regla4_mod = np.fmin(np.fmin(1 - temp_T2, 1 - vel_V1), act_A1)  # NOT for regla4
regla5_mod = np.fmin(np.fmin(1 - temp_T1, 1 - vel_V2), 1 - act_A1)  # NOT for regla5


ventilador_OFF = ventilador['OFF'].mf
ventilador_ON = ventilador['ON'].mf

agg_off_mod = np.fmax.reduce([
    np.fmin(regla1, ventilador_OFF),
    np.fmin(regla5_mod, ventilador_OFF)
])
agg_on_mod = np.fmax.reduce([
    np.fmin(regla2, ventilador_ON),
    np.fmin(regla3_mod, ventilador_ON),
    np.fmin(regla4_mod, ventilador_ON),
    np.fmin(regla6, ventilador_ON)
])

#* Agregar las salidas de las reglas
aggregated_mod = np.fmax(agg_off_mod, agg_on_mod)

#* Implementar reglas
limite_inferior_mod = ventilador_rng[np.argmax(aggregated_mod > 0)]
limite_superior_mod = ventilador_rng[len(ventilador_rng) - 1 - np.argmax(aggregated_mod[::-1] > 0)]

print("\nAggregated fuzzy set (modificado):", aggregated_mod)
print(f"Límites del conjunto difuso resultante (modificado): [{limite_inferior_mod}, {limite_superior_mod}]")

ventilador_output_mod = fz.defuzz(ventilador_rng, aggregated_mod, 'centroid')

#* ↓↓↓ Resultados ↓↓↓

#* Mostrar los conjuntos difusos resultantes

print(f"\nSalida defuzzificada del ventilador (modificado): {ventilador_output_mod}")

if ventilador_output_mod < 0.5:
    resultado_mod = "OFF"
else:
    resultado_mod = "ON"
print(f"El ventilador está: {resultado_mod}")

plt.figure()
plt.plot(ventilador_rng, aggregated_mod, label='Aggregated output')
plt.axvline(ventilador_output_mod, color='r', linestyle='--', label='Defuzzified output')
plt.title('Aggregated Fuzzy Output for Ventilador')
plt.xlabel('Ventilador')
plt.ylabel('Membership')
plt.legend()
plt.grid(True)
plt.savefig('ventilador_output_mod.png')  # Save as image
plt.close()

plt.figure()
plt.fill_between(ventilador_rng, aggregated_mod, color='skyblue', alpha=0.5, label='Aggregated output')
plt.plot(ventilador_rng, aggregated_mod, color='blue')
plt.axvline(ventilador_output_mod, color='red', linestyle='--', label='Defuzzified output')
plt.title('Aggregated Fuzzy Output for Ventilador')
plt.xlabel('Ventilador')
plt.ylabel('Membership')
plt.legend()
plt.grid(True)
plt.savefig('ventilador_output_filled_mod.png')
plt.close()
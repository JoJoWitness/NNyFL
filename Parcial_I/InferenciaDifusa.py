import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Definir el rango de los discursos
temperaturaTarMad = ctrl.Antecedent(np.arange(25, 75, 0.1), 'temperaturaTarMad')
velocidadProc = ctrl.Antecedent(np.arange(1.2, 3.2, 0.1), 'velocidadProc')
actividadDiscDur = ctrl.Antecedent(np.arange(0, 100, 0.1), 'actividadDiscDur')
ventilador = ctrl.Consequent(np.arange(0, 1, 0.1), 'ventilador')

# Definir los valores de los conjuntos difusos
temperaturaTarMad['T1'] = fuzz.trapmf(temperaturaTarMad.universe, [4.75, 22.75, 27.25, 45.25])
temperaturaTarMad['T2'] = fuzz.trapmf(temperaturaTarMad.universe, [27.25, 45.25, 49.75, 67.75])
temperaturaTarMad['T3'] = fuzz.trapmf(temperaturaTarMad.universe, [49.75, 67.75, 72.25, 90.25])
temperaturaTarMad.view()
plt.savefig('temperatura_conjunto.png')
plt.close()


velocidadProc['V1'] = fuzz.trapmf(velocidadProc.universe, [0.3, 1.1, 1.3, 2.1])
velocidadProc['V2'] = fuzz.trapmf(velocidadProc.universe, [1.3, 2.1, 2.3, 3.1])
velocidadProc['V3'] = fuzz.trapmf(velocidadProc.universe, [2.3, 3.1, 3.3, 4.1])
velocidadProc.view()
plt.savefig('velocidad_conjunto.png')
plt.close()

actividadDiscDur['A1'] = fuzz.trapmf(actividadDiscDur.universe, [-45, -5, 5, 45])
actividadDiscDur['A2'] = fuzz.trapmf(actividadDiscDur.universe, [5, 45, 55, 95])
actividadDiscDur['A3'] = fuzz.trapmf(actividadDiscDur.universe, [55, 95, 105, 145])
actividadDiscDur.view()
plt.savefig('actividad_conjunto.png')
plt.close()

ventilador['OFF'] = fuzz.trapmf(ventilador.universe, [-0.9, -0.1, 0.1, 0.9])
ventilador['ON'] = fuzz.trapmf(ventilador.universe, [0.1, 0.9, 1.1, 1.9])
ventilador.view()
plt.savefig('ventilador_conjunto.png')
plt.close()

# Definir las reglas difusas
rule1 = ctrl.Rule(
    temperaturaTarMad['T1'] & velocidadProc['V1'] & actividadDiscDur['A1'], 
    ventilador['OFF']
)
rule2 = ctrl.Rule(
    temperaturaTarMad['T3'] | velocidadProc['V3'] | actividadDiscDur['A3'], 
    ventilador['ON']
)
rule3 = ctrl.Rule(
   ~temperaturaTarMad['T1'] & ~velocidadProc['V1'] & ~actividadDiscDur['A2'], 
    ventilador['ON']
)
rule4 = ctrl.Rule(
    ~temperaturaTarMad['T2'] & ~velocidadProc['V1'] & ~actividadDiscDur['A1'], 
    ventilador['ON']
)
rule5 = ctrl.Rule(
    ~temperaturaTarMad['T1'] & ~velocidadProc['V2'] & ~actividadDiscDur['A1'], 
    ventilador['ON']
)
rule6 = ctrl.Rule(
    temperaturaTarMad['T2'] & velocidadProc['V2'] & actividadDiscDur['A2'], 
    ventilador['ON']
)

# Crear el sistema de control difuso
ventilador_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
ventilador_sim = ctrl.ControlSystemSimulation(ventilador_ctrl)

# Entrada de valores para las variables
ventilador_sim.input['temperaturaTarMad'] = 60
ventilador_sim.input['velocidadProc'] = 2.5
ventilador_sim.input['actividadDiscDur'] = 50

# Calcular el sistema difuso
ventilador_sim.compute()
ventilador.view(sim=ventilador_sim)
plt.savefig('ventilador_resultado.png')
plt.close()

#Mostrar la salida
print(f"Salida del ventilador (defuzzificada): {ventilador_sim.output['ventilador']}")
if ventilador_sim.output['ventilador'] < 0.5:
    resultado = "OFF"
else:
    resultado = "ON"
print(f"El ventilador está: {resultado}")


print("\n----------- ↓↓↓ Modificacion de reglas ↓↓↓ -----------\n")

rule4mod = ctrl.Rule(
    ~temperaturaTarMad['T2'] & ~velocidadProc['V1'] & ~actividadDiscDur['A1'], 
    ventilador['OFF']
)
rule5mod = ctrl.Rule(
    ~temperaturaTarMad['T1'] & ~velocidadProc['V2'] & ~actividadDiscDur['A1'], 
    ventilador['OFF']
)
rule6mod = ctrl.Rule(
    temperaturaTarMad['T2'] & velocidadProc['V2'] & actividadDiscDur['A2'], 
    ventilador['OFF']
)

# Crear el sistema de control difuso modificado
ventilador_ctrl_mod = ctrl.ControlSystem([rule1, rule2, rule3, rule4mod, rule5mod, rule6mod])
ventilador_sim_mod = ctrl.ControlSystemSimulation(ventilador_ctrl_mod)

ventilador_sim_mod.input['temperaturaTarMad'] = 60
ventilador_sim_mod.input['velocidadProc'] = 2.5
ventilador_sim_mod.input['actividadDiscDur'] = 50

ventilador_sim_mod.compute()
ventilador.view(sim=ventilador_sim_mod)
plt.savefig('ventilador_resultado_modificado.png')
plt.close()

# Mostrar la salida
print(f"Salida del ventilador modificado (defuzzificada): {ventilador_sim_mod.output['ventilador']}")
if ventilador_sim_mod.output['ventilador'] < 0.5:
    resultado_mod = "OFF"
else:
    resultado_mod = "ON"
print(f"El ventilador está: {resultado_mod}")



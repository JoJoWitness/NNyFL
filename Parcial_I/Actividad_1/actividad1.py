import os
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

folder = "img"
if not os.path.exists(folder):
    os.makedirs(folder)

# Definir los universos de discurso
temp_universe = np.arange(25, 70.1, 0.01)  # Temperatura [25, 70]
vel_universe = np.arange(1.2, 3.21, 0.1)  # Velocidad [1.2, 3.2]
act_universe = np.arange(0, 100.1, 1)    # Actividad [0, 100]
vent_universe = np.arange(0, 1.01, 0.1)   # Ventilador [0, 1]

# Definir los conjuntos difusos para temperatura
temp_T1 = fuzz.trapmf(temp_universe, [4.75, 22.75, 27.25, 45.25])
temp_T2 = fuzz.trapmf(temp_universe, [27.25, 45.25, 49.75, 67.75])
temp_T3 = fuzz.trapmf(temp_universe, [49.75, 67.75, 72.25, 90.25])

# Definir los conjuntos difusos para velocidad
vel_V1 = fuzz.trapmf(vel_universe, [0.3, 1.1, 1.3, 2.1])
vel_V2 = fuzz.trapmf(vel_universe, [1.3, 2.1, 2.3, 3.1])
vel_V3 = fuzz.trapmf(vel_universe, [2.3, 3.1, 3.3, 4.1])

# Definir los conjuntos difusos para actividad
act_A1 = fuzz.trapmf(act_universe, [-45, -5, 5, 45])
act_A2 = fuzz.trapmf(act_universe, [5, 45, 55, 95])
act_A3 = fuzz.trapmf(act_universe, [55, 95, 105, 145])

# OUTPUT
# Definir los conjuntos difusos para ventilador
vent_OFF = fuzz.trapmf(vent_universe, [-0.9, -0.1, 0.1, 0.9])
vent_ON = fuzz.trapmf(vent_universe, [0.1, 0.9, 1.1, 1.9])

# Visualizar los conjuntos difusos y guardarlos
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

# Temperatura
ax[0, 0].plot(temp_universe, temp_T1, 'b', linewidth=1.5, label='T1')
ax[0, 0].plot(temp_universe, temp_T2, 'g', linewidth=1.5, label='T2')
ax[0, 0].plot(temp_universe, temp_T3, 'r', linewidth=1.5, label='T3')
ax[0, 0].set_title('Temperatura')
ax[0, 0].legend()

# Velocidad
ax[0, 1].plot(vel_universe, vel_V1, 'b', linewidth=1.5, label='V1')
ax[0, 1].plot(vel_universe, vel_V2, 'g', linewidth=1.5, label='V2')
ax[0, 1].plot(vel_universe, vel_V3, 'r', linewidth=1.5, label='V3')
ax[0, 1].set_title('Velocidad')
ax[0, 1].legend()

# Actividad
ax[1, 0].plot(act_universe, act_A1, 'b', linewidth=1.5, label='A1')
ax[1, 0].plot(act_universe, act_A2, 'g', linewidth=1.5, label='A2')
ax[1, 0].plot(act_universe, act_A3, 'r', linewidth=1.5, label='A3')
ax[1, 0].set_title('Actividad')
ax[1, 0].legend()

# Ventilador
ax[1, 1].plot(vent_universe, vent_OFF, 'b', linewidth=1.5, label='OFF')
ax[1, 1].plot(vent_universe, vent_ON, 'r', linewidth=1.5, label='ON')
ax[1, 1].set_title('Ventilador')
ax[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(folder, 'conjuntos_difusos.png'))
plt.close()

# Inputs para probar el sistema
temperatura_valor = 60 #valor entre 25 y 70
velocidad_valor = 2.5 #valor entre 1.2 y 3.2 
actividad_valor = 50 #valor entre 0 y 100

print("Valores de entrada: ")
print(f"  Temperatura: {temperatura_valor}")
print(f"  Velocidad: {velocidad_valor}")
print(f"  Actividad: {actividad_valor}")

# PASO 1: Fuzificación - Calcular el grado de pertenencia para cada entrada en cada conjunto difuso
# Temperatura
temp_grado_T1 = fuzz.interp_membership(temp_universe, temp_T1, temperatura_valor)
temp_grado_T2 = fuzz.interp_membership(temp_universe, temp_T2, temperatura_valor)
temp_grado_T3 = fuzz.interp_membership(temp_universe, temp_T3, temperatura_valor)

# Velocidad
vel_grado_V1 = fuzz.interp_membership(vel_universe, vel_V1, velocidad_valor)
vel_grado_V2 = fuzz.interp_membership(vel_universe, vel_V2, velocidad_valor)
vel_grado_V3 = fuzz.interp_membership(vel_universe, vel_V3, velocidad_valor)

# Actividad
act_grado_A1 = fuzz.interp_membership(act_universe, act_A1, actividad_valor)
act_grado_A2 = fuzz.interp_membership(act_universe, act_A2, actividad_valor)
act_grado_A3 = fuzz.interp_membership(act_universe, act_A3, actividad_valor)

print("\nGrados de pertenencia:")
print(f"  Temperatura: T1={temp_grado_T1:.4f}, T2={temp_grado_T2:.4f}, T3={temp_grado_T3:.4f}")
print(f"  Velocidad: V1={vel_grado_V1:.4f}, V2={vel_grado_V2:.4f}, V3={vel_grado_V3:.4f}")
print(f"  Actividad: A1={act_grado_A1:.4f}, A2={act_grado_A2:.4f}, A3={act_grado_A3:.4f}")


# PASO 2: Evaluación de las reglas difusas - Aplicar operaciones lógicas a los grados de pertenencia
# Regla 1: IF T1 AND V1 AND A1 THEN OFF
rule1 = np.fmin(np.fmin(temp_grado_T1, vel_grado_V1), act_grado_A1)
# Regla 2: IF T3 OR V3 OR A3 THEN ON
rule2 = np.fmax(np.fmax(temp_grado_T3, vel_grado_V3), act_grado_A3)
# Regla 3: IF NOT T1 AND NOT V1 AND NOT A2 THEN NOT OFF
rule3 = np.fmin(np.fmin(1 - temp_grado_T1, 1 - vel_grado_V1), 1 - act_grado_A2)
# Regla 4: IF NOT T2 AND NOT V1 AND NOT A1 THEN NOT ON 
rule4 = np.fmin(np.fmin(1 - temp_grado_T2, 1 - vel_grado_V1), 1 - act_grado_A1)
# Regla 5: IF NOT T1 AND NOT V2 AND NOT A1 THEN NOT OFF
rule5 =  np.fmin(np.fmin(1 - temp_grado_T1, 1 - vel_grado_V2), 1 - act_grado_A1)
# Reglas 6: IF T2 AND  V1 AND  A1 THEN ON
rule6 = np.fmin(np.fmin(temp_grado_T2, vel_grado_V2), act_grado_A2)


# PASO 3: Implicación - Recortar los conjuntos de salida según el grado de activación de la regla
vent_activation_OFF =np.fmin(rule1, vent_OFF)
vent_activation_ON = np.fmax.reduce([
    np.fmin(rule2, vent_ON),
    np.fmin(rule3, vent_ON),
    np.fmin(rule4, vent_ON),
    np.fmin(rule5, vent_ON),
    np.fmin(rule6, vent_ON) 
])


# Aplicación a las funciones de salida
fan0 = np.zeros_like(vent_universe)
fan_activation_lo = np.fmin(vent_activation_OFF, vent_OFF)
fan_activation_hi = np.fmin(vent_activation_ON, vent_ON)

# PASO 4: Agregación - Unir los resultados de todas las reglas
agregacion = np.fmax(vent_activation_OFF, vent_activation_ON)

# Visualizar los conjuntos difusos recortados y el resultado de la agregación
plt.figure(figsize=(10, 6))
plt.plot(vent_universe, vent_OFF, 'b--', linewidth=1.5, label='Conjunto OFF')
plt.plot(vent_universe, vent_ON, 'r--', linewidth=1.5, label='Conjunto ON')
plt.fill_between(vent_universe, fan_activation_lo, alpha=0.3, color='blue', label='Implicación OFF')
plt.fill_between(vent_universe, fan_activation_hi , alpha=0.3, color='red', label='Implicación ON')
plt.fill_between(vent_universe, agregacion, alpha=0.7, color='grey', label='Agregación')
plt.title('Conjuntos Difusos Resultantes')
plt.ylabel('Grado de Pertenencia')
plt.xlabel('Ventilador')
plt.legend()
plt.savefig(os.path.join(folder,'conjuntos_resultantes.png'))
plt.close()

# PASO 5: Defuzificación - Calcular el valor crisp usando el método del centroide
if np.any(agregacion):
    defuzz_centroid = fuzz.defuzz(vent_universe, agregacion, 'centroid')
    # Calcular la pertenencia del resultado al conjunto difuso resultante
    defuzz_result = fuzz.interp_membership(vent_universe, agregacion, defuzz_centroid)
else:
    defuzz_centroid = 0
    defuzz_result = 0

print("\nResultados de Defuzificación:")
print(f"  Centroide: {defuzz_centroid:.4f}")
print(f"  Decision: {'ON' if defuzz_centroid > 0.5 else 'OFF'}")

# Calcular los límites del conjunto difuso resultante
if np.any(agregacion > 0.01):
    nonzero_indices = np.nonzero(agregacion > 0.01)[0]
    lower_bound = vent_universe[nonzero_indices[0]]
    upper_bound = vent_universe[nonzero_indices[-1]]
    print(f"  Límites del conjunto difuso resultante: [{lower_bound:.4f}, {upper_bound:.4f}]")
else:
    print("  No se encontraron límites (conjunto difuso vacío)")

# Visualizar la defuzificación
plt.figure(figsize=(10, 6))
plt.plot(vent_universe, vent_OFF, 'b--', linewidth=1.5, label='OFF')
plt.plot(vent_universe, vent_ON, 'r--', linewidth=1.5, label='ON')
plt.fill_between(vent_universe, agregacion, alpha=0.7, color='gray', label='Conjunto Agregado')
plt.axvline(x=defuzz_centroid, color='green', linestyle='-', linewidth=2, label=f'Centroide: {defuzz_centroid:.4f}')
plt.title('Defuzificación por Centroide')
plt.ylabel('Grado de Pertenencia')
plt.xlabel('Ventilador')
plt.legend()
plt.savefig(os.path.join(folder, 'defuzificacion.png'))
plt.close()

# PARTE 2: MODIFICACIÓN DE UNA REGLA
print("\n===== SISTEMA MODIFICADO =====")
print("Modificando regla 6: IF T2 AND V2 AND A2 THEN OFF (cambiado de ON a OFF)")

# La nueva regla 6: IF T2 AND V2 AND A2 THEN OFF (cambiado de ON a OFF)
rule6_mod = np.fmin(np.fmin(temp_grado_T2, vel_grado_V2), act_grado_A2)
print(f"  Regla 6 modificada (T2 & V2 & A2 -> OFF): {rule6_mod:.4f}")

# PASO 3 (modificado): Implicación
# Para reglas que producen OFF (1 -> OFF y ahora también 6 -> OFF)
implicacion_off_mod = np.fmin(np.fmax(rule1, rule6_mod), vent_OFF)

# Para reglas que producen ON (2 -> ON, 6 ya no contribuye a ON)
implicacion_on_mod = np.fmin(rule2, vent_ON)

# Para reglas que producen NOT OFF (3 -> NOT OFF), lo cual implica ON
implicacion_not_off_mod = np.fmin(rule3, vent_ON)  # NOT OFF implica ON

# Para reglas que producen NOT ON (4, 5 -> NOT ON), lo cual implica OFF
implicacion_not_on_mod = np.fmin(np.fmax(rule4, rule5), vent_OFF)  # NOT ON implica OFF

# Combinar todas las implicaciones para OFF
fan_activation_lo_mod = np.fmax(implicacion_off_mod, implicacion_not_on_mod)

# Combinar todas las implicaciones para ON
implicacion_on_combined_mod = np.fmax(implicacion_on_mod, implicacion_not_off_mod)

# PASO 4 (modificado): Agregación
agregacion_mod = np.fmax(fan_activation_lo_mod, implicacion_on_combined_mod)

# Visualizar los conjuntos difusos recortados y el resultado de la agregación para el sistema modificado
plt.figure(figsize=(10, 6))
plt.plot(vent_universe, vent_OFF, 'b--', linewidth=1.5, label='Conjunto OFF')
plt.plot(vent_universe, vent_ON, 'r--', linewidth=1.5, label='Conjunto ON')
plt.fill_between(vent_universe, fan_activation_lo_mod, alpha=0.3, color='blue', label='Implicación OFF')
plt.fill_between(vent_universe, implicacion_on_combined_mod, alpha=0.3, color='red', label='Implicación ON')
plt.fill_between(vent_universe, agregacion_mod, alpha=0.7, color='grey', label='Agregación')
plt.title('Conjuntos Difusos Resultantes (Sistema Modificado)')
plt.ylabel('Grado de Pertenencia')
plt.xlabel('Ventilador')
plt.legend()
plt.savefig(os.path.join(folder, 'conjuntos_resultantes_mod.png'))
plt.close()

# PASO 5 (modificado): Defuzificación
if np.any(agregacion_mod):
    defuzz_centroid_mod = fuzz.defuzz(vent_universe, agregacion_mod, 'centroid')
    defuzz_result_mod = fuzz.interp_membership(vent_universe, agregacion_mod, defuzz_centroid_mod)
else:
    defuzz_centroid_mod = 0
    defuzz_result_mod = 0

print("\nResultados de Defuzificación (sistema modificado):")
print(f"  Centroide: {defuzz_centroid_mod:.4f}")
print(f"  Decisión: {'ON' if defuzz_centroid_mod > 0.5 else 'OFF'}")

# Calcular los límites del conjunto difuso resultante para el sistema modificado
if np.any(agregacion_mod > 0.01):
    nonzero_indices_mod = np.nonzero(agregacion_mod > 0.01)[0]
    lower_bound_mod = vent_universe[nonzero_indices_mod[0]]
    upper_bound_mod = vent_universe[nonzero_indices_mod[-1]]
    print(f"  Límites del conjunto difuso resultante: [{lower_bound_mod:.4f}, {upper_bound_mod:.4f}]")
else:
    print("  No se encontraron límites (conjunto difuso vacío)")

# Visualizar la defuzificación para el sistema modificado
plt.figure(figsize=(10, 6))
plt.plot(vent_universe, vent_OFF, 'b--', linewidth=1.5, label='OFF')
plt.plot(vent_universe, vent_ON, 'r--', linewidth=1.5, label='ON')
plt.fill_between(vent_universe, agregacion_mod, alpha=0.7, color='gray', label='Conjunto Agregado')
plt.axvline(x=defuzz_centroid_mod, color='green', linestyle='-', linewidth=2, label=f'Centroide: {defuzz_centroid_mod:.4f}')
plt.title('Defuzificación por Centroide (Sistema Modificado)')
plt.ylabel('Grado de Pertenencia')
plt.xlabel('Ventilador')
plt.legend()
plt.savefig(os.path.join(folder, 'defuzificacion_mod.png'))
plt.close()
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import os

folder = "img"
if not os.path.exists(folder):
    os.makedirs(folder)

# Definir los universos de discurso
temp_universe = np.arange(25, 70.1, 0.1)  # Temperatura [25, 70]
vel_universe = np.arange(1.2, 3.21, 0.01)  # Velocidad [1.2, 3.2]
act_universe = np.arange(0, 100.1, 0.1)    # Actividad [0, 100]
vent_universe = np.arange(0, 1.01, 0.01)   # Ventilador [0, 1]

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

# Valores de entrada para probar el sistema
temperatura_valor = 30 #valor entre 25 y 70
velocidad_valor = 1.8 #valor entre 1.2 y 3.2 
actividad_valor =70 #valor entre 0 y 100

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
# Regla 1: Si temp es T1 y velocidad es V1 y actividad es A1, entonces ventilador es OFF
regla1 = np.min([temp_grado_T1, vel_grado_V1, act_grado_A1])

# Regla 2: Si temp es T3 y velocidad es V3 y actividad es A3, entonces ventilador es ON
regla2 = np.min([temp_grado_T3, vel_grado_V3, act_grado_A3])

# Regla 3: Si temp es T2 y velocidad es V2 y actividad es A2, entonces ventilador es ON
regla3 = np.min([temp_grado_T2, vel_grado_V2, act_grado_A2])

print("\nActivación de las reglas:")
print(f"  Regla 1 (T1 & V1 & A1 -> OFF): {regla1:.4f}")
print(f"  Regla 2 (T3 & V3 & A3 -> ON): {regla2:.4f}")
print(f"  Regla 3 (T2 & V2 & A2 -> ON): {regla3:.4f}")

# PASO 3: Implicación - Recortar los conjuntos de salida según el grado de activación de la regla
# Utilizamos el método de implicación MIN
implicacion_off = np.fmin(regla1, vent_OFF)
implicacion_on1 = np.fmin(regla2, vent_ON)
implicacion_on2 = np.fmin(regla3, vent_ON)

# PASO 4: Agregación - Combinar los conjuntos de salida recortados
# Utilizamos el método de agregación MAX para combinar las salidas ON
implicacion_on_combined = np.fmax(implicacion_on1, implicacion_on2)

# Agregar todas las salidas
agregacion = np.fmax(implicacion_off, implicacion_on_combined)

# Visualizar los conjuntos difusos recortados y el resultado de la agregación
plt.figure(figsize=(10, 6))
plt.plot(vent_universe, vent_OFF, 'b--', linewidth=1.5, label='Conjunto OFF')
plt.plot(vent_universe, vent_ON, 'r--', linewidth=1.5, label='Conjunto ON')
plt.fill_between(vent_universe, implicacion_off, alpha=0.3, color='blue', label='Implicación OFF')
plt.fill_between(vent_universe, implicacion_on_combined, alpha=0.3, color='red', label='Implicación ON')
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
print(f"  Decisión: {'ON' if defuzz_centroid > 0.5 else 'OFF'}")

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
print("SISTEMA MODIFICADO (Cambiando regla 3)")

# Modificamos la Regla 3: Si temp es T2 y velocidad es V2 y actividad es A2, entonces ventilador es OFF (en lugar de ON)
# La regla 3 modificada contribuye ahora a OFF en lugar de ON
# Los grados de pertenencia son los mismos que calculamos antes

print("\nActivación de las reglas (sistema modificado):")
print(f"  Regla 1 (T1 & V1 & A1 -> OFF): {regla1:.4f}")
print(f"  Regla 2 (T3 & V3 & A3 -> ON): {regla2:.4f}")
print(f"  Regla 3 modificada (T2 & V2 & A2 -> OFF): {regla3:.4f}")

# PASO 3 (modificado): Implicación
implicacion_off_mod = np.fmin(regla1, vent_OFF)  # Regla 1 -> OFF
implicacion_off_mod2 = np.fmin(regla3, vent_OFF)  # Regla 3 modificada -> OFF
implicacion_on_mod = np.fmin(regla2, vent_ON)     # Regla 2 -> ON

# Combinar las dos implicaciones OFF usando MAX
implicacion_off_combined = np.fmax(implicacion_off_mod, implicacion_off_mod2)

# PASO 4 (modificado): Agregación
agregacion_mod = np.fmax(implicacion_off_combined, implicacion_on_mod)

# Visualizar los conjuntos difusos recortados y el resultado de la agregación para el sistema modificado
plt.figure(figsize=(10, 6))
plt.plot(vent_universe, vent_OFF, 'b--', linewidth=1.5, label='Conjunto OFF')
plt.plot(vent_universe, vent_ON, 'r--', linewidth=1.5, label='Conjunto ON')
plt.fill_between(vent_universe, implicacion_off_combined, alpha=0.3, color='blue', label='Implicación OFF')
plt.fill_between(vent_universe, implicacion_on_mod, alpha=0.3, color='red', label='Implicación ON')
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

# Crear una comparación entre los dos sistemas
plt.figure(figsize=(12, 8))

# Sistema original
plt.subplot(2, 1, 1)
plt.plot(vent_universe, vent_OFF, 'b--', linewidth=1.5, label='OFF')
plt.plot(vent_universe, vent_ON, 'r--', linewidth=1.5, label='ON')
plt.fill_between(vent_universe, agregacion, alpha=0.7, color='gray')
plt.axvline(x=defuzz_centroid, color='green', linestyle='-', linewidth=2, label=f'Centroide: {defuzz_centroid:.4f}')
plt.title('Sistema Original')
plt.ylabel('Grado de Pertenencia')
plt.legend()

# Sistema modificado
plt.subplot(2, 1, 2)
plt.plot(vent_universe, vent_OFF, 'b--', linewidth=1.5, label='OFF')
plt.plot(vent_universe, vent_ON, 'r--', linewidth=1.5, label='ON')
plt.fill_between(vent_universe, agregacion_mod, alpha=0.7, color='gray')
plt.axvline(x=defuzz_centroid_mod, color='green', linestyle='-', linewidth=2, label=f'Centroide: {defuzz_centroid_mod:.4f}')
plt.title('Sistema Modificado')
plt.ylabel('Grado de Pertenencia')
plt.xlabel('Ventilador')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(folder, 'comparacion_sistemas.png'))
plt.close()